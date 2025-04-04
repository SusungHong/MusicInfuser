# MusicInfuser
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://susunghong.github.io/MusicInfuser/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2503.14505)

MusicInfuser adapts a text-to-video diffusion model to align with music, generating dance videos according to the music and text prompts.

## 🚧 Requirements

We have tested on Python 3.10 with `torch>=2.4.1+cu118`, `torchaudio>=2.4.1+cu118`, and `torchvision>=0.19.1+cu118`. This repository requires a single A100 GPU for training and inference.

## 🧱 Installation
```bash
# Clone the repository
git clone https://github.com/SusungHong/MusicInfuser
cd MusicInfuser

# Create and activate conda environment
conda create -n musicinfuser python=3.10
conda activate musicinfuser

# Install dependencies
pip install -r requirements.txt
pip install -e ./mochi --no-build-isolation

# Download model weights
python ./music_infuser/download_weights.py weights/
```

## 🎞️ Inference
To generate videos from music inputs:
```bash
python inference.py --input-file {MP3 or MP4 to extract audio from} \
                    --prompt {prompt} \
                    --num-frames {number of frames}
```

with the following arguments:
- `--input-file`: Input file (MP3 or MP4) to extract audio from.
- `--prompt`: Prompt for the dancer generation. The more specific a prompt is, generally the better the results, but more specificity decreases the effect of audio. Default: `"a professional female dancer dancing K-pop in an advanced dance setting in a studio with a white background, captured from a front view"`
- `--num-frames`: Number of frames to generate. While originally trained with 73 frames, MusicInfuser can extrapolate to longer sequences. Default: `145`

also consider:
- `--seed`: Random seed for generation. The resulting dance also depends on the random seed, so feel free to change it. Default: `None`
- `--cfg-scale`: Classifier-Free Guidance (CFG) scale for the text prompt. Default: `6.0`

## 📀 Dataset
For the AIST dataset, please see the terms of use and download it at [the AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/).

## 🚊 Training
To train the model on your dataset:

1. Preprocess your data:
```bash
bash music_infuser/preprocess.bash -v {dataset path} -o {processed video output dir} -w {path to pretrained mochi} --num_frames {number of frames}
```

2. Run training:
```bash
bash music_infuser/run.bash -c music_infuser/configs/music_infuser.yaml -n 1
```

**Note:** The current implementation only supports single-GPU training, which requires approximately 80GB of VRAM to train with 73-frame sequences.

## 🧑‍⚖️ VLM Evaluation
For evaluating the model using Visual Language Models:

1. Follow the instructions in `vlm_eval/README.md` to set up the VideoLLaMA2 evaluation framework. It is recommended to use a separate python environment from MusicInfuser for the evaluation.
2. Since we parse filenames for automatic evaluation, the names of the outputs should be formatted accordingly. Please refer to `extract_info_from_filename(filename)` in `vlm_eval/eval_quality.py` and `vlm_eval/eval_alignment.py` for details.

## 📚 Citation

```bibtex
@article{hong2025musicinfuser,
  title={MusicInfuser: Making Video Diffusion Listen and Dance},
  author={Hong, Susung and Kemelmacher-Shlizerman, Ira and Curless, Brian and Seitz, Steven M},
  journal={arXiv preprint arXiv:2503.14505},
  year={2025}
}
```

## 🙏 Acknowledgements

This code builds upon the following awesome repositories:
- [Mochi](https://github.com/genmoai/mochi)
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything)

We thank the authors for open-sourcing their code and models, which made this work possible.
