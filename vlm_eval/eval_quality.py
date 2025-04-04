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

def get_evaluation_questions():
    questions = [
        "Rate the imaging quality 0-10 where: 0 means poor imaging quality, 5 means moderate imaging quality, and 10 means perfect imaging quality. Output only the number.",
        "Rate the aesthetic quality 0-10 where: 0 means poor aesthetic quality, 5 means moderate aesthetic quality, and 10 means perfect aesthetic quality. Output only the number.",
        "Rate the overall consistency 0-10 where: 0 means poor consistency, 5 means moderate consistency, and 10 means perfect consistency. Output only the number.",
        "Rate the style alignment of the dance to music 0-10 where: 0 means poor style alignment of the dance to music, 5 means moderate style alignment of the dance to music, and 10 means perfect style alignment of the dance to music. Output only the number.",
        "Rate the beat alignment of the dance to music 0-10 where: 0 means poor beat alignment of the dance to music, 5 means moderate beat alignment of the dance to music, and 10 means perfect beat alignment of the dance to music. Output only the number.",
        "Rate the body representation of the dancer 0-10 where: 0 means unrealistic/distorted proportions of the dancer, 5 means minor anatomical issues of the dancer, and 10 means anatomically perfect representation of the dancer. Output only the number.",
        "Rate the movement realism of the dancer 0-10 where: 0 means poor movement realism of the dancer, 5 means moderate movement realism of the dancer, and 10 means perfect movement realism of the dancer. Output only the number.",
        "Rate the complexity of the choreography 0-10 where: 0 means extremely basic choreography, 5 means intermediate choreography, and 10 means extremely complex/advanced choreography. Output only the number.",
    ]
    return questions

def get_video_paths(base_path):
    video_files = defaultdict(list)
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if "_prompt_" in d.lower()]

        for file in files:
            if file.endswith('.mp4') and not file.endswith('.recon.mp4'):
                
                _, model_config, _ = extract_info_from_filename(file)
                if model_config:
                    full_path = os.path.join(root, file)
                    video_files[model_config].append(full_path)
    
    for model_config in video_files:
        video_files[model_config].sort()
    
    return dict(video_files)

def calculate_category_scores(scores, metrics):
    categories = {
        "Video Quality": metrics[:3],
        "Dance Quality": metrics[3:8],
    }
    
    category_averages = {}
    for category, cat_metrics in categories.items():
        cat_scores = [float(scores[metrics.index(m)]) for m in cat_metrics]
        category_averages[category] = sum(cat_scores) / len(cat_scores)
    
    return category_averages

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def write_report(report_path, model_results, video_details, metrics, num_runs=5):
    with open(report_path, 'w') as f:
        f.write("Video Dance Generation Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Each score is averaged across {num_runs} evaluation runs\n\n")

        f.write("Detailed Video Results\n")
        f.write("-" * 20 + "\n\n")
        for model_config, videos in video_details.items():
            f.write(f"Model Configuration: {model_config}\n")
            f.write("-" * 30 + "\n")
            for video_idx, video_data in enumerate(videos, 1):
                f.write(f"\nVideo {video_idx}: {os.path.basename(video_data['path'])}\n")
                f.write(f"Directory: {os.path.dirname(video_data['path'])}\n")
                f.write(f"Valid Runs: {video_data['valid_runs']}/{num_runs}\n")
                
                for category, score in video_data['category_scores'].items():
                    f.write(f"{category}: {score:.1f}/10\n")
                f.write(f"Overall Score: {video_data['overall_score']:.1f}/10\n\n")

        f.write("\nDetailed Question Averages by Model\n")
        f.write("=" * 40 + "\n\n")
        for model_config, results in model_results.items():
            f.write(f"\nModel: {model_config}\n")
            f.write("-" * 30 + "\n")
            
            categories = {
                "Video Metrics": metrics[:5],
                "Dance Metrics": metrics[5:10],
            }
            
            for category, category_metrics in categories.items():
                f.write(f"\n{category}:\n")
                for metric in category_metrics:
                    if metric in results['metric_scores']:
                        score = results['metric_scores'][metric]
                        f.write(f"  {metric}: {score:.1f}/10\n")
            f.write("\n")

        f.write("\nModel Comparison Summary\n")
        f.write("=" * 40 + "\n\n")
        for model_config, results in model_results.items():
            f.write(f"\nModel: {model_config}\n")
            f.write("-" * 30 + "\n")
            
            for category, score in results['category_averages'].items():
                f.write(f"{category}: {score:.1f}/10\n")
            f.write(f"Overall Average: {results['overall_average']:.1f}/10\n")
        
        json_path = report_path.replace('.txt', '.json')
        f.write(f"\n\nComplete evaluation data has been saved as JSON in: {os.path.basename(json_path)}\n")
    
    json_path = report_path.replace('.txt', '.json')
    
    serializable_video_details = {}
    for model, videos in video_details.items():
        serializable_video_details[model] = []
        for video in videos:
            video_data = {
                'path': video['path'],
                'directory': os.path.dirname(video['path']),
                'scores': video['scores'],
                'category_scores': video['category_scores'],
                'overall_score': video['overall_score'],
                'valid_runs': video['valid_runs']
            }
            serializable_video_details[model].append(video_data)
    
    with open(json_path, 'w') as f:
        json_data = {
            'model_results': model_results, 
            'video_details': serializable_video_details,
            'metadata': {
                'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_runs': num_runs,
                'metrics': metrics
            }
        }
        json.dump(json_data, f, indent=2)

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

def run_evaluation(video_path, model, processor, tokenizer, questions, metrics, args, num_runs=5):
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

def inference(args):
    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    
    num_runs = args.num_runs if hasattr(args, 'num_runs') else 5

    base_path = args.video_dir if hasattr(args, 'video_dir') else ""
    
    report_dir = args.report_dir if hasattr(args, 'report_dir') else "."
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Scanning for videos in {base_path} and all subdirectories...")
    video_paths_by_model = get_video_paths(base_path)
    
    total_videos = sum(len(videos) for videos in video_paths_by_model.values())
    print(f"Found {total_videos} video files across {len(video_paths_by_model)} model configurations")
    
    questions = get_evaluation_questions()
    metrics = [
        'Imaging Quality', 'Aesthetic Quality', 'Overall Consistency',
        'Style Alignment', 'Beat Alignment', 'Body Representation', 'Movement Realism', 'Choreography Complexity',
    ]

    model_results = {}
    video_details = defaultdict(list)

    for model_config, videos in video_paths_by_model.items():
        print(f"\nEvaluating Model Configuration: {model_config}")
        print("=" * 50)
        
        model_scores = []
        
        for video_idx, video_path in enumerate(videos, 1):
            print(f"\nVideo {video_idx}:")
            print(f"Path: {video_path}")
            print(f"Directory: {os.path.dirname(video_path)}")
            
            print("\nRunning multiple evaluations and averaging results...")
            video_scores, valid_runs = run_evaluation(
                video_path, model, processor, tokenizer, 
                questions, metrics, args, num_runs=num_runs
            )
            
            if valid_runs == 0:
                print(f"Skipping video {video_idx} due to evaluation failures")
                continue
                
            print("\nDetailed Scores (averaged across {valid_runs} runs):")
            print("-" * 20)
                
            model_scores.append(video_scores)
            
            category_averages = calculate_category_scores(video_scores, metrics)
            for category, avg_score in category_averages.items():
                print(f"{category}: {avg_score:.1f}/10")
            overall_score = sum(video_scores) / len(video_scores)
            print(f"Video Overall Average: {overall_score:.1f}/10")
            
            video_details[model_config].append({
                'path': video_path,
                'scores': video_scores,
                'category_scores': category_averages,
                'overall_score': overall_score,
                'valid_runs': valid_runs
            })
        
        if model_scores:
            avg_scores = [sum(scores) / len(scores) for scores in zip(*model_scores)]
            model_results[model_config] = {
                'scores': avg_scores,
                'overall_average': sum(avg_scores) / len(avg_scores),
                'category_averages': calculate_category_scores(avg_scores, metrics),
                'metric_scores': dict(zip(metrics, avg_scores)),
                'video_count': len(model_scores)
            }

    print("\nFinal Model Comparison")
    print("=" * 50)
    for model_config, results in model_results.items():
        print(f"\nModel: {model_config} (based on {results['video_count']} videos)")
        print("-" * 30)
        for category, score in results['category_averages'].items():
            print(f"{category}: {score:.1f}/10")
        print(f"Overall Average: {results['overall_average']:.1f}/10")
    
    timestamp = get_timestamp()
    report_path = os.path.join(report_dir, f"dance_evaluation_report_{timestamp}.txt")
    write_report(report_path, model_results, video_details, metrics, num_runs=num_runs)
    json_path = report_path.replace('.txt', '.json')
    print(f"\nDetailed report has been saved to: {report_path}")
    print(f"JSON data has been saved to: {json_path}")

if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the model', 
                       default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
    parser.add_argument('--modal-type', choices=["a", "v", "av"], 
                       help='Modal type (audio, video, or audio-video)', default="av")
    parser.add_argument('--video-dir', help='Directory containing video samples',
                       default="/gscratch/realitylab/susung/mochi-audio/evaluation_samples_final/")
    parser.add_argument('--num-runs', type=int, help='Number of evaluation runs per video',
                       default=1)
    parser.add_argument('--report-dir', help='Directory to save the evaluation report',
                       default=".")
    
    args = parser.parse_args()
    inference(args)