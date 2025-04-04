import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import re
import argparse
import os
from collections import defaultdict
from datetime import datetime
import json

import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_info_from_filename(filename):
    pattern = r"0_(.+?)\.audio_wav2vec\.pt_(.+?)_1\.0_prompt(\d+)_"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None

def get_prompt_alignment_questions(prompt_text):
    questions = [
        f"How well does the dance video capture the specific style mentioned in the prompt: '{prompt_text}'? Rate 0-10 where: 0 means completely missed the style, 5 means some elements of the style are present, and 10 means perfectly captures the style. Output only the number.",

        f"Based on the prompt '{prompt_text}', rate the creativity in interpreting the prompt 0-10 where: 0 means generic/standard interpretation, 5 means moderate creativity, and 10 means highly creative and unique interpretation. Output only the number.",
        
        f"Rate the overall prompt satisfaction 0-10 where: 0 means the video fails to satisfy the prompt '{prompt_text}', 5 means it partially satisfies the prompt, and 10 means it fully satisfies all aspects of the prompt. Output only the number.",
    ]
    return questions

def get_video_paths(base_path):
    video_files = defaultdict(lambda: defaultdict(list))
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if 'prompt' in d.lower()]
        
        for file in files:
            if file.endswith('.mp4'):
                audio_name, model_config, prompt_num = extract_info_from_filename(file)
                if model_config and prompt_num is not None:
                    full_path = os.path.join(root, file)
                    video_files[model_config][prompt_num].append((full_path, audio_name))
    
    for model_config in video_files:
        for prompt_num in video_files[model_config]:
            video_files[model_config][prompt_num].sort()
    
    return dict(video_files)

def parse_score(output):
    try:
        output = output.strip()
        match = re.search(r'(\d+(?:\.\d+)?)', output)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                return score
            else:
                return None
        else:
            return None
    except (ValueError, TypeError):
        return None

def run_prompt_evaluation(video_path, prompt_text, model, processor, tokenizer, args, num_runs=5):
    questions = get_prompt_alignment_questions(prompt_text)
    metrics = [
        'Style Capture',
        'Creative Interpretation',
        'Overall Prompt Satisfaction',
    ]
    
    all_scores = [[] for _ in range(len(questions))]
    valid_runs = 0

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs} for {os.path.basename(video_path)}")
        
        try:
            preprocess = processor['audio' if args.modal_type == "a" else "video"]
            if args.modal_type == "a":
                audio_video_tensor = preprocess(video_path)
            else:
                audio_video_tensor = preprocess(video_path, va=True if args.modal_type == "av" else False)

            run_successful = True
            for i, (question, metric) in enumerate(zip(questions, metrics)):
                output = mm_infer(
                    audio_video_tensor,
                    question,
                    model=model,
                    tokenizer=tokenizer,
                    modal='audio' if args.modal_type == "a" else "video",
                    do_sample=False,
                )
                
                score = parse_score(output)
                if score is not None:
                    all_scores[i].append(score)
                else:
                    print(f"Warning: Could not parse score for metric '{metric}' on run {run+1}. Raw output: '{output}'")
                    run_successful = False
            
            if run_successful:
                valid_runs += 1
        except Exception as e:
            print(f"Error in run {run+1}: {str(e)}")
    
    if valid_runs > 0:
        avg_scores = []
        for i, metric_scores in enumerate(all_scores):
            if metric_scores:
                avg = sum(metric_scores) / len(metric_scores)
                avg_scores.append(avg)
            else:
                print(f"Warning: No valid scores for metric '{metrics[i]}'")
                avg_scores.append(0.0)
        
        print(f"Completed {valid_runs} valid runs out of {num_runs} attempts")
        return avg_scores, valid_runs
    else:
        print(f"Error: No valid runs completed for {os.path.basename(video_path)}")
        return [0.0] * len(questions), 0

def calculate_category_scores(scores, metrics):
    category = "Prompt Alignment"
    category_avg = sum(scores) / len(scores)
    return {category: category_avg}

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def write_report(report_path, results, prompts, metrics, num_runs=5):
    with open(report_path, 'w') as f:
        f.write("Video Dance Prompt Alignment Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Each score is averaged across {num_runs} evaluation runs\n\n")

        f.write("Prompts Used for Evaluation\n")
        f.write("-" * 30 + "\n")
        for prompt_num, prompt_text in enumerate(prompts, 1):
            if prompt_num <= len(prompts):
                f.write(f"Prompt {prompt_num}: {prompt_text}\n")
        f.write("\n\n")
        
        f.write("Overall Model Averages Across All Prompts\n")
        f.write("=" * 50 + "\n\n")
        
        model_averages = {}
        for model_config, prompt_results in results.items():
            all_model_scores = []
            video_count = 0
            
            for prompt_num, videos in prompt_results.items():
                for video_data in videos:
                    video_path, audio_name, scores, valid_runs = video_data
                    if valid_runs > 0:
                        all_model_scores.append(scores)
                        video_count += 1
            
            if all_model_scores:
                avg_scores = [sum(scores) / len(scores) for scores in zip(*all_model_scores)]
                category_avg = calculate_category_scores(avg_scores, metrics)
                overall_avg = sum(avg_scores) / len(avg_scores)
                
                model_averages[model_config] = {
                    'scores': avg_scores,
                    'overall_average': overall_avg,
                    'category_scores': category_avg,
                    'metric_scores': dict(zip(metrics, avg_scores)),
                    'video_count': video_count
                }
                
                f.write(f"Model: {model_config} (based on {video_count} videos)\n")
                f.write("-" * 30 + "\n")
                
                for i, (score, metric) in enumerate(zip(avg_scores, metrics)):
                    f.write(f"{metric}: {score:.1f}/10\n")
                
                for category, score in category_avg.items():
                    f.write(f"{category}: {score:.1f}/10\n")
                
                f.write(f"Overall Average: {overall_avg:.1f}/10\n\n")
        
        if model_averages:
            f.write("\nModel Rankings (Overall Performance)\n")
            f.write("-" * 40 + "\n")
            
            sorted_models = sorted(model_averages.items(), 
                                  key=lambda x: x[1]['overall_average'], reverse=True)
            for rank, (model, data) in enumerate(sorted_models, 1):
                f.write(f"{rank}. {model}: {data['overall_average']:.1f}/10\n")
            
            f.write("\n\n")
        
        f.write("Detailed Results by Model and Prompt\n")
        f.write("=" * 50 + "\n\n")
        
        video_details = defaultdict(list)
        
        for model_config, prompt_results in results.items():
            f.write(f"Model Configuration: {model_config}\n")
            f.write("=" * 30 + "\n\n")
            
            for prompt_num, videos in prompt_results.items():
                prompt_idx = prompt_num - 1 if prompt_num > 0 else 0
                prompt_text = prompts[prompt_idx] if 0 <= prompt_idx < len(prompts) else f"Unknown Prompt {prompt_num}"
                
                f.write(f"Prompt {prompt_num}: {prompt_text}\n")
                f.write("-" * 40 + "\n\n")
                
                prompt_scores = []
                prompt_valid_videos = 0
                
                for video_idx, video_data in enumerate(videos, 1):
                    video_path, audio_name, scores, valid_runs = video_data
                    
                    if valid_runs == 0:
                        f.write(f"Video {video_idx} ({os.path.basename(video_path)}): Evaluation failed\n\n")
                        continue
                    
                    f.write(f"Video {video_idx}: {os.path.basename(video_path)}\n")
                    f.write(f"Directory: {os.path.dirname(video_path)}\n")
                    f.write(f"Audio: {audio_name}\n")
                    f.write(f"Valid Runs: {valid_runs}/{num_runs}\n")
                    
                    category_avg = calculate_category_scores(scores, metrics)
                    overall_score = sum(scores) / len(scores)
                    
                    for i, (score, metric) in enumerate(zip(scores, metrics)):
                        f.write(f"{metric}: {score:.1f}/10\n")
                    
                    for category, score in category_avg.items():
                        f.write(f"{category}: {score:.1f}/10\n")
                    
                    f.write(f"Overall Score: {overall_score:.1f}/10\n\n")
                    
                    prompt_scores.append(scores)
                    prompt_valid_videos += 1
                    
                    video_details[model_config].append({
                        'path': video_path,
                        'scores': scores,
                        'category_scores': category_avg,
                        'overall_score': overall_score,
                        'valid_runs': valid_runs,
                        'prompt_num': prompt_num
                    })
                
                if prompt_valid_videos > 0:
                    prompt_avg_scores = [sum(scores) / len(scores) for scores in zip(*prompt_scores)]
                    prompt_category_avg = calculate_category_scores(prompt_avg_scores, metrics)
                    prompt_overall_avg = sum(prompt_avg_scores) / len(prompt_avg_scores)
                    
                    f.write(f"Prompt {prompt_num} Averages (across {prompt_valid_videos} videos):\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, (score, metric) in enumerate(zip(prompt_avg_scores, metrics)):
                        f.write(f"{metric}: {score:.1f}/10\n")
                        
                    for category, score in prompt_category_avg.items():
                        f.write(f"{category}: {score:.1f}/10\n")
                        
                    f.write(f"Overall Average: {prompt_overall_avg:.1f}/10\n\n")
        
        json_path = report_path.replace('.txt', '.json')
        f.write(f"\n\nComplete evaluation data has been saved as JSON in: {os.path.basename(json_path)}\n")
    
    return video_details

def save_json_results(json_path, results, prompts, video_details, metrics):
    output_data = {
        'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prompts': {i+1: prompt for i, prompt in enumerate(prompts)},
        'metrics': metrics,
        'model_results': {},
        'video_details': {}
    }
    
    for model_config, prompt_results in results.items():
        all_model_scores = []
        video_count = 0
        
        for prompt_num, videos in prompt_results.items():
            for video_data in videos:
                video_path, audio_name, scores, valid_runs = video_data
                if valid_runs > 0:
                    all_model_scores.append(scores)
                    video_count += 1
        
        if all_model_scores:
            avg_scores = [sum(scores) / len(scores) for scores in zip(*all_model_scores)]
            category_avg = calculate_category_scores(avg_scores, metrics)
            overall_avg = sum(avg_scores) / len(avg_scores)
            
            output_data['model_results'][model_config] = {
                'scores': dict(zip(metrics, avg_scores)),
                'overall_average': overall_avg,
                'category_scores': category_avg,
                'video_count': video_count
            }
    
    for model_config, videos in video_details.items():
        output_data['video_details'][model_config] = []
        for video in videos:
            video_data = {
                'path': video['path'],
                'directory': os.path.dirname(video['path']),
                'scores': dict(zip(metrics, video['scores'])),
                'category_scores': video['category_scores'],
                'overall_score': video['overall_score'],
                'valid_runs': video['valid_runs'],
                'prompt_num': video['prompt_num']
            }
            output_data['video_details'][model_config].append(video_data)
    
    output_data['model_rankings'] = []
    if output_data['model_results']:
        sorted_models = sorted(
            output_data['model_results'].items(),
            key=lambda x: x[1]['overall_average'],
            reverse=True
        )
        
        for rank, (model, data) in enumerate(sorted_models, 1):
            output_data['model_rankings'].append({
                'rank': rank,
                'model': model,
                'score': data['overall_average']
            })
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def inference(args):
    prompts = [
        # This is an example
        "a professional female dancer dancing K-pop in an advanced dance setting in a studio with a white background, captured from a front view",
    ]
    
    print("Initializing model...")
    model, processor, tokenizer = model_init(args.model_path)
    
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    
    print("Scanning for videos in directories containing 'prompt'...")
    videos_by_model_and_prompt = get_video_paths(args.video_dir)
    
    results = {}
    metrics = [
        'Style Capture',
        'Creative Interpretation',
        'Overall Prompt Satisfaction',
    ]
    
    for model_config, prompt_videos in videos_by_model_and_prompt.items():
        print(f"\nEvaluating Model Configuration: {model_config}")
        print("=" * 50)
        
        model_results = {}
        
        for prompt_num, videos in prompt_videos.items():
            prompt_idx = prompt_num - 1 if prompt_num > 0 else 0
            
            if not (0 <= prompt_idx < len(prompts)):
                print(f"Warning: No prompt text found for prompt number {prompt_num}, skipping videos.")
                continue
                
            prompt_text = prompts[prompt_idx]
            print(f"\nPrompt {prompt_num}: {prompt_text}")
            print("-" * 50)
            
            prompt_results = []
            
            for video_idx, (video_path, audio_name) in enumerate(videos, 1):
                print(f"\nVideo {video_idx}:")
                print(f"Path: {os.path.basename(video_path)}")
                print(f"Directory: {os.path.dirname(video_path)}")
                print(f"Audio: {audio_name}")
                
                print("\nRunning prompt alignment evaluation...")
                scores, valid_runs = run_prompt_evaluation(
                    video_path, prompt_text, model, processor, tokenizer, 
                    args, num_runs=args.num_runs
                )
                
                if valid_runs == 0:
                    print(f"Skipping video {video_idx} due to evaluation failures")
                    prompt_results.append((video_path, audio_name, scores, valid_runs))
                    continue
                
                category_scores = calculate_category_scores(scores, metrics)
                overall_score = sum(scores) / len(scores)
                
                print("\nPrompt Alignment Scores:")
                print("-" * 30)
                for i, (score, metric) in enumerate(zip(scores, metrics)):
                    print(f"{metric}: {score:.1f}/10")
                
                for category, score in category_scores.items():
                    print(f"{category}: {score:.1f}/10")
                
                print(f"Overall Score: {overall_score:.1f}/10")
                
                prompt_results.append((video_path, audio_name, scores, valid_runs))
            
            model_results[prompt_num] = prompt_results
        
        results[model_config] = model_results
    
    timestamp = get_timestamp()
    
    report_dir = args.output_dir
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, f"prompt_alignment_report_{timestamp}.txt")
    video_details = write_report(report_path, results, prompts, metrics, num_runs=args.num_runs)
    print(f"\nDetailed prompt alignment report has been saved to: {report_path}")
    
    json_path = os.path.join(report_dir, f"prompt_alignment_report_{timestamp}.json")
    save_json_results(json_path, results, prompts, video_details, metrics)
    print(f"JSON results have been saved to: {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the model', 
                       default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
    parser.add_argument('--modal-type', choices=["a", "v", "av"], 
                       help='Modal type (audio, video, or audio-video)', default="av")
    parser.add_argument('--video-dir', help='Directory containing video samples',
                       default="")
    parser.add_argument('--num-runs', type=int, help='Number of evaluation runs per video',
                       default=1)
    parser.add_argument('--output-dir', help='Directory to save output files',
                       default="./")
    
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    set_seed()
    main()