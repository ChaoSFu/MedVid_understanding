"""Skill Assessment Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict
import numpy as np


def detect_dataset_from_video_id(video_id):
    """Detect dataset from video ID patterns."""
    video_id = str(video_id).lower()
    
    # JIGSAWS dataset - patterns like "knot_tying_b001", "suturing_b001", etc.
    if any(pattern in video_id for pattern in ["knot_tying", "suturing", "needle_passing"]) and "_b" in video_id:
        return "jigsaws"
    
    # AVOS dataset - YouTube video IDs
    if len(video_id) == 11 and any(c.isalpha() for c in video_id):
        return "AVOS"
    
    # CoPESD dataset - numerical IDs with parts
    if "_part" in video_id and video_id.replace("_part", "").split("_")[0].isdigit():
        return "CoPESD"
    
    # CholecT50 dataset
    if "video" in video_id.lower() and any(c.isdigit() for c in video_id):
        return "CholecT50"
    
    # NurViD dataset - specific patterns
    if any(keyword in video_id for keyword in ["nur", "nursing", "medical"]):
        return "NurViD"
    
    return "Unknown"


def detect_dataset_from_question(question):
    """Detect dataset from question text patterns."""
    question_lower = question.lower()
    
    # JIGSAWS dataset - look for robotic surgery, bench-top tasks
    if any(pattern in question_lower for pattern in ["robotic bench-top", "knot-tying", "needle-passing", "suturing", "surgical technique"]):
        return "jigsaws"
    
    if "avos" in question_lower:
        return "AVOS"
    elif "copesd" in question_lower:
        return "CoPESD"
    elif "cholect50" in question_lower or "cholec" in question_lower:
        return "CholecT50"
    elif "nurvid" in question_lower or "nursing" in question_lower:
        return "NurViD"
    
    # Check for dataset-specific action patterns
    if any(action in question_lower for action in ["cutting", "tying", "suturing"]):
        return "AVOS"
    elif "forceps" in question_lower and "knife" in question_lower:
        return "CoPESD"
    
    return "Unknown"


def parse_skill_scores(skill_text):
    """Parse skill assessment text into individual scores."""
    import re
    
    # Extract all X/5 patterns
    pattern = r'(\d+)/5'
    scores = re.findall(pattern, skill_text)
    # print("scores in parse_skill_scores", scores)
    if scores:
        # Convert to integers and return average
        numeric_scores = [int(score) for score in scores]
        # print("numeric_scores", numeric_scores)
        return sum(numeric_scores) / len(numeric_scores)
    
    return None


def parse_aspect_scores(skill_text):
    """Parse aspect scores from text like 'Respect for tissue: 2/5, Suture/needle handling: 1/5, ...'"""
    import re
    
    # Split by commas first, then parse each part
    parts = skill_text.split(',')
    aspect_scores = {}
    
    for part in parts:
        # Pattern to match aspect name followed by score within each part
        match = re.search(r'([^:]+?):\s*(\d+)/5', part.strip())
        if match:
            aspect_name = match.group(1).strip()
            score = int(match.group(2))
            aspect_scores[aspect_name] = score
    # print("parts", parts)
    return aspect_scores


def normalize_skill_level(skill_text):
    """Normalize skill level text to standard format for classification."""
    skill_text = skill_text.strip().lower()
    # print("skill_text in normalize_skill_level")
    # print("-"*50)
    # print(skill_text)
    # print("-"*50)
    
    # JIGSAWS skill level mapping - treat as direct classification
    skill_mappings = {
        # Direct skill level names
        "novice": "novice",
        "beginner": "novice", 
        "intermediate": "intermediate",
        "expert": "expert",
        "advanced": "expert",
        
        # Letter codes (JIGSAWS uses N, I, E)
        "n": "novice",
        "i": "intermediate", 
        "e": "expert",
        
        # Numeric mappings (if any)
        "1": "novice",
        "2": "intermediate", 
        "3": "expert",
        
        # Quality descriptors
        "low": "novice",
        "medium": "intermediate", 
        "high": "expert",
        "poor": "novice",
        "good": "intermediate",
        "excellent": "expert"
    }
    
    # Check for exact matches first
    if skill_text in skill_mappings:
        # print("skill_text in skill_mappings", skill_text, "skill_mappings[skill_text]", skill_mappings[skill_text])
        return skill_mappings[skill_text]
    
    # Check for partial matches
    for key, value in skill_mappings.items():
        if key in skill_text:
            return value
    
    # Return original if no mapping found (for debugging)
    print(f"Warning: No mapping found for skill_text: '{skill_text}'")
    return skill_text


def convert_scores_to_skill_level(skill_text):
    """Convert structured skill assessment scores to skill level."""
    # If it contains scores (like "Respect for tissue: 1/5, ..."), parse them
    avg_score = parse_skill_scores(skill_text)
    # print("avg_score in convert_scores_to_skill_level", avg_score)
    if avg_score is not None:
        # Convert average score to skill level
        if avg_score <= 2.0:
            return "novice"
        elif avg_score <= 3.5:
            return "intermediate"
        else:
            return "expert"
    
    # If no scores found, return None
    return None


def calculate_balanced_accuracy(per_class_correct, per_class_total):
    """Calculate balanced accuracy across classes."""
    if not per_class_total:
        return 0.0
    
    # Calculate recall for each class
    recalls = []
    for class_name in per_class_total:
        if per_class_total[class_name] > 0:
            recall = per_class_correct[class_name] / per_class_total[class_name]
            recalls.append(recall)
    
    # Balanced accuracy is the mean of per-class recalls
    if recalls:
        return np.mean(recalls)
    else:
        return 0.0


def group_records_by_dataset(data):
    """Group skill assessment records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        if record.get("qa_type") != "skill_assessment":
            continue
            
        # Get dataset from data_source field if available (preferred method)
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            dataset = detect_dataset_from_video_id(record["metadata"]["video_id"])
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record["question"])
        
        record_data = {
            "question": record["question"],
            "answer": record["answer"],
            "gnd": record["gnd"],
            "video_id": record["metadata"]["video_id"],
            "struc_info": record.get("struc_info", [])
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def evaluate_skill_assessment(records):
    """Evaluate skill assessment using accuracy metric."""
    if not records:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    
    correct = 0
    total = 0
    per_skill_correct = defaultdict(int)
    per_skill_total = defaultdict(int)
    
    # Per-aspect evaluation
    aspect_correct = defaultdict(int)
    aspect_total = defaultdict(int)
    aspect_mae = defaultdict(float)  # Mean Absolute Error for aspects
    
    for record in records:
        # print("record")
        # print(record)
        # print("--------------------------------")
        
        # Get predicted skill level from the answer
        # Parse structured scores (like "Respect for tissue: 1/5, ...")
        pred_skill = convert_scores_to_skill_level(record["answer"])
        
        if pred_skill is None:
            print(f"Warning: Could not parse answer for skill level: '{record['answer']}'. Skipping record.")
            continue
        
        # print("pred_skill", pred_skill)
        # print()
        
        # Get ground truth skill level from struc_info if available, otherwise from gnd text
        gnd_skill = None
        if record.get("struc_info") and len(record["struc_info"]) > 0:
            skill_level_code = record["struc_info"][0].get("skill_level", "")
            if skill_level_code:
                gnd_skill = normalize_skill_level(skill_level_code)
        
        # Fallback to parsing the ground truth text if struc_info not available
        if not gnd_skill:
            gnd_skill = convert_scores_to_skill_level(record["gnd"])
            if gnd_skill is None:
                print(f"Warning: Could not parse ground truth for skill level: '{record['gnd']}'. Skipping record.")
                continue
        
        per_skill_total[gnd_skill] += 1
        total += 1
        
        if pred_skill == gnd_skill:
            correct += 1
            per_skill_correct[gnd_skill] += 1
        
        # Parse aspect scores from text
        pred_aspects = parse_aspect_scores(record["answer"])
        gnd_aspects = None
        
        # Get ground truth aspect scores from struc_info if available
        if record.get("struc_info") and len(record["struc_info"]) > 0:
            gnd_aspects = record["struc_info"][0].get("skill_scores", {})
        
        # Fallback to parsing ground truth text
        if not gnd_aspects:
            gnd_aspects = parse_aspect_scores(record["gnd"])
        
        # Evaluate each aspect
        for aspect_name in gnd_aspects:
            if aspect_name in pred_aspects:
                gnd_score = gnd_aspects[aspect_name]
                pred_score = pred_aspects[aspect_name]
                
                aspect_total[aspect_name] += 1
                
                # Exact match accuracy
                if pred_score == gnd_score:
                    aspect_correct[aspect_name] += 1
                
                # Mean Absolute Error
                aspect_mae[aspect_name] += abs(pred_score - gnd_score)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate per-skill accuracies
    per_skill_accuracies = {}
    for skill in per_skill_total:
        skill_correct = per_skill_correct[skill]
        skill_total = per_skill_total[skill]
        skill_accuracy = skill_correct / skill_total if skill_total > 0 else 0.0
        per_skill_accuracies[skill] = {
            "accuracy": skill_accuracy,
            "correct": skill_correct,
            "total": skill_total
        }
    
    # Calculate balanced accuracy for aspects only
    aspect_balanced_acc = calculate_balanced_accuracy(aspect_correct, aspect_total)
    
    # Calculate per-aspect metrics
    per_aspect_metrics = {}
    for aspect in aspect_total:
        aspect_acc = aspect_correct[aspect] / aspect_total[aspect] if aspect_total[aspect] > 0 else 0.0
        aspect_mae_avg = aspect_mae[aspect] / aspect_total[aspect] if aspect_total[aspect] > 0 else 0.0
        per_aspect_metrics[aspect] = {
            "accuracy": aspect_acc,
            "correct": aspect_correct[aspect],
            "total": aspect_total[aspect],
            "mae": aspect_mae_avg
        }
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_skill": per_skill_accuracies,
        "per_aspect": per_aspect_metrics,
        "aspect_balanced_accuracy": aspect_balanced_acc
    }


def evaluate_dataset_skill_assessment(dataset_name, dataset_records):
    """Evaluate skill assessment for a specific dataset."""
    print(f"\n=== Skill Assessment Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # Evaluate the dataset
    results = evaluate_skill_assessment(dataset_records)
    
    # Print per-aspect results FIRST (main focus)
    if "per_aspect" in results and results["per_aspect"]:
        print(f"\n*** PER-ASPECT PERFORMANCE ***")
        print(f"Aspect Balanced Accuracy: {results.get('aspect_balanced_accuracy', 0.0):.4f}")
        print("\nIndividual Aspect Performance:")
        
        # Sort aspects by name for consistent output
        sorted_aspects = sorted(results["per_aspect"].items())
        for aspect, metrics in sorted_aspects:
            print(f"  {aspect}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"    Mean Absolute Error: {metrics['mae']:.3f}")
    
    # Print overall skill level results (secondary)
    print(f"\n*** OVERALL SKILL LEVEL CLASSIFICATION ***")
    print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    # Print per-skill results
    if "per_skill" in results and results["per_skill"]:
        print("\nPer-skill Level Accuracy:")
        sorted_skills = sorted(results["per_skill"].items())
        for skill, metrics in sorted_skills:
            print(f"  {skill}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    return results


def main():
    """Main evaluation function."""
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        print("Usage: python eval_skill_assessment.py <results_file.json>")
        sys.exit(1)
    
    print(f"Loading results from: {output_file}")
    
    with open(output_file, "r") as f:
        infer_output = json.load(f)
    
    # Group records by dataset
    dataset_records = group_records_by_dataset(infer_output)
    
    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} skill assessment records")
    
    if not any(dataset_records.values()):
        print("No skill assessment records found!")
        return
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_skill_assessment(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*80}")
    print("SKILL ASSESSMENT EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            
            # Show per-aspect summary first
            if "per_aspect" in results and results["per_aspect"]:
                print(f"  Aspect Balanced Accuracy: {results.get('aspect_balanced_accuracy', 0.0):.4f}")
                print("  Per-Aspect Accuracy:")
                sorted_aspects = sorted(results["per_aspect"].items())
                for aspect, metrics in sorted_aspects:
                    print(f"    {aspect}: {metrics['accuracy']:.4f} (MAE: {metrics['mae']:.3f})")
            
            # Show overall skill level accuracy
            print(f"  Overall Skill Level Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    all_bal_acc = [r.get('aspect_balanced_accuracy', 0.0) for r in all_results.values() if r]
    return {
        'per_dataset': all_results,
        'aspect_balanced_accuracy': np.mean(all_bal_acc) if all_bal_acc else 0.0
    }


if __name__ == "__main__":
    main()
