import json
import multiprocessing as mp
import os
import gc
import random
import re
import sys
import time
import math
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import click
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
import torch
from torch import Tensor
from torch.utils.data import Subset
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
import torch.nn.functional as F
from tqdm import tqdm

torch._dynamo.config.cache_size_limit = 32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(False)

import genmo.mochi_preview.dit.joint_model.lora as lora
import genmo.mochi_preview.dit.joint_model.audio_adapter as audio

from genmo.lib.progress import progress_bar
from genmo.lib.utils import Timer, save_video
from genmo.mochi_preview.vae.vae_stats import vae_latents_to_dit_latents
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    ModelFactory,
    T5ModelFactory,
    cast_dit,
    compute_packed_indices,
    get_conditioning,
    linear_quadratic_schedule,  # used in eval'd Python code in lora.yaml
    load_to_cpu,
    move_to_device,
    sample_model,
    t5_tokenizer,
)
from genmo.mochi_preview.vae.latent_dist import LatentDistribution
from genmo.mochi_preview.vae.models import decode_latents_tiled_spatial

sys.path.append("..")

from dataset import LatentEmbedDataset, MusicInfuserDataset

from moviepy.editor import VideoFileClip, AudioFileClip

def clear_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def set_seed(seed: int = 42):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

def save_video_with_audio(video_frames, output_path, audio_pt_path, wav2vec=True):
    temp_video_path = str(output_path).replace('.mp4', '_temp.mp4')
    save_video(video_frames, temp_video_path)
    audio_dir = os.path.dirname(audio_pt_path)
    audio_filename = os.path.basename(audio_pt_path)
    mp3_filename = audio_filename.replace(f'.audio_{"wav2vec" if wav2vec else ""}.pt', '.mp3')
    mp3_path = os.path.join(audio_dir, mp3_filename)
    try:
        video = VideoFileClip(temp_video_path)
        audio = AudioFileClip(mp3_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(
            str(output_path), 
            codec='libx264',
            audio_codec='mp3',
            temp_audiofile='temp-audio.mp3',
            remove_temp=True
        )
        video.close()
        audio.close()
        os.remove(temp_video_path)
    except Exception as e:
        print(f"Error processing video with audio: {str(e)}")
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, output_path)

class MochiTorchRunEvalPipeline:
    def __init__(
        self,
        *,
        device_id,
        dit,
        text_encoder_factory: ModelFactory,
        decoder_factory: ModelFactory,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.tokenizer = t5_tokenizer()
        t = Timer()
        self.dit = dit
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                world_size=1,
                device_id="cpu",
            )
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id="cpu", world_size=1)
        t.print_stats()  # type: ignore

    def __call__(self, prompt, audio_feat, audio_pt_path, save_path, **kwargs):
        with progress_bar(type="tqdm", enabled=True), torch.inference_mode():
            # Encode prompt with T5 XXL.
            with move_to_device(self.text_encoder, self.device, enabled=True):
                conditioning = get_conditioning(
                    self.tokenizer,
                    self.text_encoder,
                    self.device,
                    batch_inputs=False,
                    prompt=prompt,
                    negative_prompt="",
                )

            # Sample video latents from Mochi.
            with move_to_device(self.dit, self.device, enabled=True):
                latents = sample_model(self.device, self.dit, conditioning, audio_feat=audio_feat, **kwargs)

            # Decode video latents to frames.
            with move_to_device(self.decoder, self.device, enabled=True):
                frames = decode_latents_tiled_spatial(
                    self.decoder, latents, num_tiles_w=2, num_tiles_h=2, overlap=8)
            frames = frames.cpu().numpy()  # b t h w c
            assert isinstance(frames, np.ndarray)

            save_video_with_audio(frames[0], save_path, audio_pt_path)


def map_to_device(x, device: torch.device):
    if isinstance(x, dict):
        return {k: map_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [map_to_device(y, device) for y in x]
    elif isinstance(x, tuple):
        return tuple(map_to_device(y, device) for y in x)
    elif isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    else:
        return x


EPOCH_IDX = 0


def infinite_dl(dl):
    global EPOCH_IDX
    while True:
        EPOCH_IDX += 1
        for batch in dl:
            yield batch


@contextmanager
def timer(description="Task", enabled=True):
    if enabled:
        start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            elapsed = time.perf_counter() - start  # type: ignore
            print(f"{description} took {elapsed:.4f} seconds")


def get_cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True, help="Path to YAML config file")
def main(config_path):
    mp.set_start_method("spawn", force=True)
    cfg = cast(DictConfig, OmegaConf.load(config_path))

    set_seed(getattr(cfg, 'seed', 42))
    audio_mode = getattr(cfg, 'audio_mode', 'cross_attn')
    audio_cross_attn_layers = getattr(cfg, 'audio_cross_attn_layers', [6, 7, 8, 9, 10, 21, 34, 35, 36, 38, 39, 43, 44, 45, 46, 47])
    basic_prompt_ratio = getattr(cfg, 'basic_prompt_ratio', 1.0)
    beta_beta = getattr(cfg, 'beta_beta', 3)
    beta_half = getattr(cfg, 'beta_half', 200)

    device_id = 0
    device_str = f"cuda:0"
    device = torch.device(device_str)

    # Verify checkpoint path exists
    checkpoint_path = Path(cfg.init_checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get step number from checkpoint filename
    pattern = r"model_(\d+)\.(lora|checkpoint)\.(safetensors|pt)"
    match = re.search(pattern, str(checkpoint_path))
    if match:
        start_step_num = int(match.group(1))
        opt_path = str(checkpoint_path).replace("model_", "optimizer_")
    else:
        start_step_num = 0
        opt_path = ""

    print(
        f"model={checkpoint_path}, optimizer={opt_path}, start_step_num={start_step_num}"
    )

    is_audio = audio_mode is not None
    print(f"Basic prompt ratio: {basic_prompt_ratio}")

    wandb_run = None
    sample_prompts = cfg.sample.prompts
    if is_audio:
        sample_audios = cfg.sample.audios

    train_vids = list(sorted(glob(f"{cfg.train_data_dir}/*.mp4")))
    train_vids = [v for v in train_vids if not v.endswith(".recon.mp4")]
    print(f"Found {len(train_vids)} training videos in {cfg.train_data_dir}")
    assert len(train_vids) > 0, f"No training data found in {cfg.train_data_dir}"
    if cfg.single_video_mode:
        train_vids = train_vids[:1]
        sample_prompts = [Path(train_vids[0]).with_suffix(".txt").read_text()]
        print(f"Training on video: {train_vids[0]}")

    train_vids_2 = list(sorted(glob(f"{cfg.train_data_dir_2}/*.mp4")))
    train_vids_2 = [v for v in train_vids_2 if not v.endswith(".recon.mp4")]
    print(f"Found {len(train_vids_2)} training videos in {cfg.train_data_dir_2}")
    
    train_dataset = MusicInfuserDataset(
        train_vids,
        train_vids_2,
        cfg.train_data_dir_2_ratio,
        basic_prompt_ratio=basic_prompt_ratio,
        repeat=1_000 if cfg.single_video_mode else 1,
    )
    
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    train_dl_iter = infinite_dl(train_dl)

    if cfg.get("wandb"):
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}-{int(time.time())}",
            config=OmegaConf.to_container(cfg),  # type: ignore
        )
        print(f"🚀 Weights & Biases run URL: {wandb_run.get_url()}")

    print("Loading model")
    patch_model_fns = []
    model_kwargs = {}
    is_lora = "lora" in cfg.model.type
    is_full = "full" in cfg.model.type
    print(f"{'Enable LoRA' if is_lora else 'Disable LoRA'}")
    print(f"{'Enable audio' if is_audio else 'Disable audio'}")
    if is_lora:
        if is_audio:
            def mark_lora_params(m):
                audio.mark_audio_and_lora_as_trainable(m, bias="none")
                return m
        else:
            def mark_lora_params(m):
                lora.mark_only_lora_as_trainable(m, bias="none")
                return m

        patch_model_fns.append(mark_lora_params)
        model_kwargs = dict(**cfg.model.kwargs)
        # Replace ListConfig with list to allow serialization to JSON.
        for k, v in model_kwargs.items():
            if isinstance(v, ListConfig):
                model_kwargs[k] = list(v)

    elif is_audio and not is_full:
        def mark_lora_params(m):
            audio.mark_only_audio_as_trainable(m, bias="none")
            return m

        patch_model_fns.append(mark_lora_params)

    if cfg.training.get("model_dtype"):
        assert cfg.training.model_dtype == "bf16", f"Only bf16 is supported"
        patch_model_fns.append(lambda m: cast_dit(m, torch.bfloat16))

    model = (
        DitModelFactory(
            model_path=str(checkpoint_path),
            model_dtype="bf16",
            attention_mode=cfg.attention_mode,
            audio_mode=audio_mode,
            audio_cross_attn_layers=audio_cross_attn_layers,
        ).get_model(
            local_rank=0,
            device_id=device_id,
            model_kwargs=model_kwargs,
            patch_model_fns=patch_model_fns,
            world_size=1,
            strict_load=not is_lora and not is_audio,
            fast_init=not is_lora and not is_audio,  # fast_init not supported for LoRA (please someone fix this !!!)
        )
        .train()  # calling train() makes sure LoRA weights are not merged
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    if os.path.exists(opt_path):
        print("Loading optimizer")
        optimizer.load_state_dict(load_to_cpu(opt_path))

    scheduler = get_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=cfg.training.num_steps
    )

    print("Loading eval pipeline ...")
    eval_pipeline = MochiTorchRunEvalPipeline(
        device_id=device_id,
        dit=model,
        text_encoder_factory=T5ModelFactory(),
        decoder_factory=DecoderModelFactory(model_path=cfg.sample.decoder_path),
    )

    def get_batch() -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        nonlocal train_dl_iter
        batch = next(train_dl_iter)  # type: ignore
        latent, embed, audio_embed = cast(Tuple[Dict[str, Any], Dict[str, Any]], batch)
        assert len(embed["y_feat"]) == 1 and len(embed["y_mask"]) == 1, f"Only batch size 1 is supported"

        ldist = LatentDistribution(latent["mean"], latent["logvar"])
        z = ldist.sample()
        assert torch.isfinite(z).all()
        assert z.shape[0] == 1, f"Only batch size 1 is supported"

        eps = torch.randn_like(z)

        if beta_beta is None:
            sigma = torch.rand(z.shape[:1], device="cpu", dtype=torch.float32)
        else:
            # Calculate beta parameters based on current step
            half_life = beta_half
            alpha = torch.ones(z.shape[:1], device="cpu", dtype=torch.float32)
            beta = 1 + (beta_beta - 1) * math.exp(-(step / half_life) * math.log(2))
            
            # Sample from beta distribution
            beta_dist = torch.distributions.Beta(alpha, beta)
            sigma = beta_dist.sample()

        if random.random() < cfg.training.caption_dropout:
            embed["y_mask"][0].zero_()
            embed["y_feat"][0].zero_()
        return embed, z, eps, sigma, audio_embed

    pbar = tqdm(
        range(start_step_num, cfg.training.num_steps),
        total=cfg.training.num_steps,
        initial=start_step_num,
    )
    for step in pbar:
        if cfg.sample.interval and step % cfg.sample.interval == 0 and step > 0:
            sample_dir = Path(cfg.sample.output_dir)
            sample_dir.mkdir(exist_ok=True)
            model.eval()
            
            if not is_audio:
                sample_audios = [None] * len(sample_prompts)

            for eval_idx, sp in enumerate(zip(sample_prompts, sample_audios)):
                prompt, audio_path = sp
                audio_embedding = load_to_cpu(audio_path)["audio_embeddings"].cuda()

                save_path = sample_dir / f"{eval_idx}_{step}.mp4"
                if save_path.exists():
                    print(f"Skipping {save_path} as it already exists")
                    continue

                sample_kwargs = {
                    k.removesuffix("_python_code"): (eval(v) if k.endswith("_python_code") else v)
                    for k, v in cfg.sample.kwargs.items()
                }
                eval_pipeline(
                    prompt=prompt,
                    audio_feat=audio_embedding,
                    audio_pt_path=audio_path,
                    save_path=str(save_path),
                    seed=cfg.sample.seed + eval_idx,
                    **sample_kwargs,
                )
                Path(sample_dir / f"{eval_idx}_{step}.txt").write_text(prompt)
            
                clear_cuda_cache()
            
            model.train()

        with torch.no_grad(), timer("load_batch", enabled=False):
            batch = get_batch()
            embed, z, eps, sigma, audio_embed = map_to_device(batch, device)
            embed = cast(Dict[str, Any], embed)

            num_latent_toks = np.prod(z.shape[-3:])
            indices = compute_packed_indices(device, cast(Tensor, embed["y_mask"][0]), int(num_latent_toks))

            sigma_bcthw = sigma[:, None, None, None, None]  # [B, 1, 1, 1, 1]
            z_sigma = (1 - sigma_bcthw) * z + sigma_bcthw * eps
            ut = z - eps

        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(
                x=z_sigma,
                sigma=sigma,
                packed_indices=indices,
                audio_feat=audio_embed["audio_embeddings"],
                **embed,
                num_ff_checkpoint=cfg.training.num_ff_checkpoint,
                num_qkv_checkpoint=cfg.training.num_qkv_checkpoint,
            )
            assert preds.shape == z.shape

        ut_dit_space = vae_latents_to_dit_latents(ut.float())
        loss = F.mse_loss(preds.float(), ut_dit_space)
        loss.backward()

        log_kwargs = {
            "train/loss": loss.item(),
            "train/epoch": EPOCH_IDX,
            "train/lr": scheduler.get_last_lr()[0],
        }

        if cfg.training.get("grad_clip"):
            assert not is_lora, "Gradient clipping not supported for LoRA"
            gnorm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip)
            log_kwargs["train/gnorm"] = gnorm_before_clip.item()
        pbar.set_postfix(**log_kwargs)

        if wandb_run:
            wandb_run.log(log_kwargs, step=step)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        clear_cuda_cache()
        
        if cfg.training.save_interval and step > 0 and (step + 1) % cfg.training.save_interval == 0:
            with timer("get_state_dict"):
                if is_lora:
                    model_sd = lora.lora_state_dict(model, bias="none")
                elif is_full:
                    # NOTE: Not saving optimizer state dict to save space.
                    model_sd, _optimizer_sd = get_state_dict(
                        model, [], options=StateDictOptions(cpu_offload=True, full_state_dict=True)
                    )
                else:
                    model_sd = {}
                if is_audio:
                    audio_sd = model.audio_projection.get_state_dict()
                    model_sd.update(audio_sd)
                    if model.audio_cross_attn_blocks:
                        audio_ca_sd = model.audio_cross_attn_blocks.state_dict()
                        audio_ca_sd = {f'audio_cross_attn_blocks.{k}': v for k, v in audio_ca_sd.items()}
                        model_sd.update(audio_ca_sd)

            checkpoint_filename = f"model_{step+1}.{'adapter' if is_lora or is_audio else 'checkpoint'}.pt"
            save_path = checkpoint_dir / checkpoint_filename
            if cfg.training.get("save_safetensors", True):
                save_path = save_path.with_suffix(".safetensors")
                save_file(
                    model_sd, save_path,
                    # `safetensors` only supports string-to-string metadata,
                    # so we serialize the kwargs to a JSON string.
                    metadata=dict(kwargs=json.dumps(model_kwargs)),
                )
            else:
                torch.save(model_sd, save_path)


if __name__ == "__main__":
    main()
