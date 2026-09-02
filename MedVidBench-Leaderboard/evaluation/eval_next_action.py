from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
import os
import numpy as np

# Dataset-specific action lists
AVOS_ACTIONS = ["cutting", "tying", "suturing"]

T50_PHASES = [
    "preparation",
    "carlot-triangle-dissection",
    "clipping-and-cutting",
    "gallbladder-dissection",
    "gallbladder-packaging",
    "cleaning-and-coagulation",
    "gallbladder-extraction"
]

TOTAL_NEW_ACTION_LIST = [
    "adjust camera",
    "position flap with forceps and knife",
    "dissect flap tissue with knife",
    "position flap with forceps only",
    "retract flap edge with forceps only",
    "retract flap edge with forceps and knife",
    "lift flap with forceps",
    "stabilize flap with forceps"
]

# Map old CoPESD actions to new ones for backward compatibility
COPESD_ACTION_MAPPING = {
    "manipulate flap with forceps and knife": "position flap with forceps and knife",
    "dissect flap with knife": "dissect flap tissue with knife",
    "manipulate flap with forceps": "position flap with forceps only",
    "retract flap with forceps": "retract flap edge with forceps only",
    "retract flap with forceps and knife": "retract flap edge with forceps and knife",
    "lift flap with forceps": "lift flap with forceps",
    "hold flap with forceps": "stabilize flap with forceps",
    "retracting mucosa flap with forceps and knife": "retract flap edge with forceps and knife"
}

NURVID_PROCEDURE_ACTIONS = {
    "Administering Oral Medications": [
        "Assist patient taking medicine","Check","Document","Handwashing",
        "Organize the bed unit","Position the patient","Prepare medications"
    ],
    "Aseptic Technique": [
        "Check",
        "Take treatment towels",
    ],
    "Bed Rubbing": [
        "Change upper clothing",
        "Cleanse back",
        "Cleanse chest and abdomen",
        "Cleanse perineum",
        "Handwashing",
        "Rub lower limbs",
        "Rub upper limbs",
        "Soak feet",
        "Wash face",
    ],
    "Bed Shampoo": [
        "Apply shampoo",
        "Comb hair",
        "Dry hair",
        "Moisten hair",
        "Place an underpad",
        "Rinse shampoo",
    ],
    "Blood Glucose Monitoring": [
        "Disinfect skin",
        "Document",
        "Handwashing",
        "Measure blood glucose level",
        "Prepare glucometer",
    ],
    "Cardiopulmonary Resuscitation WIth Manual Resuscitation Bag": [
        "Administer oxygen",
        "Assist with ventilation using a simple respirator",
        "Defibrillate",
        "Identify cardiac arrest",
        "Open airway",
        "Perform chest compressions",
    ],
    "Change Sheets of an Occupied Bed": [
        "Change pillowcase",
        "Handwashing",
        "Prepare operating space",
        "Remove proximal bedsheet",
        "Replace clean bedsheet",
        "Spread the opposite side bed sheet",
        "Spread the proximal bedshee",
        "Withdraw contaminated bed shee",
        "Withdraw the opposite side bed sheet",
    ],
    "Change Wound Dressings": [
        "Cleanse skin",
        "Document",
        "Fill in dressing",
        "Handwashing",
    ],
    "Change a One-Piece Pouching System": [
        "Apply leak prevention ointment",
        "Apply skin protection film",
        "Cleanse skin",
        "Handwashing",
        "Remove ostomy bag",
        "Secure ostomy bag",
        "Trim ostomy bag baseplate",
    ],
    "Change a Two-Piece Pouching System": [
        "Apply leak prevention ointment",
        "Apply skin protection film",
        "Cleanse skin",
        "Handwashing",
        "Remove ostomy bag",
        "Remove the base plate",
        "Secure ostomy bag",
        "Secure the base",
        "Spray stoma care powder",
        "Trim ostomy bag baseplate",
    ],
    "Closed Bed Making": [
        "Cover pillow with pillowcase",
        "Prepare operating space",
        "Spread the large sheet",
    ],
    "Closed Intravenous infusion": [
        "Adjust drip rate",
        "Check",
        "Connect infusion device",
        "Disinfect skin",
        "Document",
        "Handwashing",
        "Release trapped air",
        "Remove needle",
        "Select a vein",
        "Venipuncture",
    ],
    "Closed System Blood Transfusion": [
        "Check",
        "Handwashing",
        "Release trapped air",
        "Transfuse blood",
    ],
    "Defibrillation": [
        "Defibrillate",
        "Observe defibrillation results",
        "Prepare defibrillation device",
    ],
    "Donning and Doffing Isolation Gowns": [
        "Fasten buckle",
        "Handwashing",
        "Loosen isolation gown",
        "Put on isolation gown",
        "Remove isolation gown",
        "Tie waist knot",
    ],
    "Electrocardiogram": [
        "Connect lead wires",
        "Expose the connection sit",
        "Remove the lead wires",
        "Save electrocardiogram (ECG) results",
    ],
    "Female Retention Catheterization": [
        "Disinfect skin",
        "Establish a sterile zone",
        "Insert urinary catheter",
        "Remove urinary catheter",
    ],
    "High-Volume Colonic Enemas": [
        "Check",
        "Inject medication",
        "Insert rectal tube",
        "Place an underpad",
        "Position the patient",
        "Remove rectal tube",
    ],
    "Infusion by Pump": [
        "Connect infusion device",
        "Flush the sealed tube",
        "Release trapped air",
        "Set parameters",
    ],
    "Intramuscular Injection": [
        "Check",
        "Disinfect skin",
        "Handwashing",
        "Inject medication",
        "Position the patient",
        "Prepare medication solution",
    ],
    "Intravenous Blood Sampling": [
        "Blood collection",
        "Check",
        "Disinfect skin",
        "Document",
        "Handwashing",
        "Mix blood sample",
        "Select a vein",
        "Venipuncture",
    ],
    "Intravenous Injection": [
        "Check",
        "Disinfect skin",
        "Document",
        "Handwashing",
        "Inject medication",
        "Prepare medication solution",
        "Release trapped air",
        "Select a vein",
        "Venipuncture",
    ],
    "Logrolling with Draw Sheet": [
        "Check",
        "Check and secure the tubing",
        "Handwashing",
        "Shift to the right side",
        "Turn patient to left lateral position",
    ],
    "Male Retention Catheterization": [
        "Disinfect skin",
        "Establish a sterile zone",
        "Insert urinary catheter",
        "Position the patient",
        "Remove urinary catheter",
    ],
    "Modified Seldinger Technique with Ultrasound for PICC Placement": [
        "Check and secure the tubing",
        "Disinfect skin",
        "Establish a sterile zone",
        "PICC insertion",
        "Withdraw the introducer sheath",
    ],
    "Multi-Parameter Monitoring": [
        "Connect the monitor",
        "Monitor blood oxygen saturation",
    ],
    "Nasogastric Gavage": [
        "Confirm the position of the gastric tube in the stomach",
        "Handwashing",
        "Insert gastric tube",
        "Measure the length of the gastric tube",
        "Nasogastric feeding",
        "Place an underpad",
        "Position the patient",
        "Remove gastric tube",
        "Secure gastric tube",
    ],
    "Nasogastric Tube": [
        "Check the pressure reducer",
        "Document",
        "Insert gastric tube",
        "Measure the length of the gastric tube",
        "Observe drainage situation",
        "Position the patient",
    ],
    "Oral Care for Unconscious Patients": [
        "Check",
        "Cleanse inner surfaces of teeth",
        "Cleanse lips",
        "Cleanse outer surfaces of teeth",
        "Document",
        "Handwashing",
        "Place an underpad",
        "Position the patient",
        "Prepare cotton balls",
    ],
    "Oral and Nasal Suctioning with Central Negative Pressure Device": [
        "Connect suction catheter",
        "Organize the bed unit",
        "Perform endotracheal suctioning",
        "Perform nasopharyngeal and nasotracheal suction",
        "Perform oral-pharyngeal suction",
    ],
    "Oral and Nasal Suctioning with Electric Suction Device": [
        "Adjust negative pressure",
        "Check",
        "Connect suction catheter",
        "Handwashing",
        "Perform nasopharyngeal and nasotracheal suction",
        "Perform oral-pharyngeal suction",
        "Rinse suction catheter",
    ],
    "Oxygen Nebulization": [
        "Adjust oxygen flow rate",
        "Guide nebulization",
        "Install nebulizer",
        "Withdraw nebulizer",
    ],
    "Oxygen Therapy with Central Oxygen Supply": [
        "Adjust oxygen flow rate",
        "Administer oxygen",
        "Handwashing",
        "Install oxygen inhalation device",
        "Withdraw oxygen inhalation device",
    ],
    "Penicillin Skin Testing": [
        "Check",
        "Disinfect skin",
        "Handwashing",
        "Observe results of skin test",
        "Perform intradermal puncture",
        "Prepare skin test solution",
        "Release trapped air",
    ],
    "Perineal Care": [
        "Clean and scrub the perineum",
        "Draw bed curtains",
        "Place an underpad",
        "Position the patient",
    ],
    "Peripheral Venous Indwelled Needle Infusion and Maintaince": [
        "Connect infusion device",
        "Disinfect skin",
        "Flush the sealed tube",
        "Handwashing",
        "Remove needle",
        "Secure the indwelling needle",
        "Venipuncture",
    ],
    "Retention Enema": [
        "Check",
        "Handwashing",
        "Inject medication",
        "Insert rectal tube",
        "Organize the bed unit",
        "Place an underpad",
        "Position the patient",
        "Remove rectal tube",
    ],
    "Skin Preparation": [
        "Cleanse skin",
        "Handwashing",
        "Position the patient",
    ],
    "Sputum Specimen Collection": [
        "Check",
        "Collect sputum specimen",
        "Handwashing",
        "Wear gloves",
    ],
    "Stool Specimen Collection": [
        "Check",
        "Collect stool specimen",
        "Handwashing",
        "Wear gloves",
    ],
    "Subcutaneous Injection": [
        "Aspirate medication",
        "Disinfect skin",
        "Handwashing",
        "Inject medication",
        "Perform subcutaneous puncture",
        "Release trapped air",
        "Remove needle",
    ],
    "Subcutaneous Injection Insulin": [
        "Disinfect skin",
        "Inject medication",
        "Prepare medication solution",
    ],
    "Surgical Hand Scrub": [
        "Dry hands",
        "Perform seven-step handwashing technique",
        "Perform surgical hand disinfection",
        "Perform surgical hand scrub",
        "Rinse with running water",
    ],
    "Throat Swab Collection": [
        "Collect pharyngeal swab specimen",
        "Document",
    ],
    "Transfer with Stretcher": [
        "Move and transfer",
        "Perform four-person transfer",
    ],
    "Urine Specimen Collection": [
        "Check",
        "Collect urine specimen",
        "Handwashing",
    ],
    "Use of Restraints": [
        "Immobilize the shoulder",
    ],
    "Vital Sign Assessment": [
        "Check the blood pressure meter",
        "Check the thermometer",
        "Document",
        "Handwashing",
        "Measure blood pressure",
        "Measure body temperature",
        "Measure pulse",
        "Measure respiration",
    ],
    "Wheelchair Transfer Technique": [
        "Assist with bed rest",
        "Transport in wheelchair",
    ],
}

def detect_dataset_from_file(file_path):
    """
    Detect dataset from file path or name
    """
    file_name = os.path.basename(file_path).lower()
    
    if "avos" in file_name:
        return "AVOS"
    elif "cholect50" in file_name or "t50" in file_name:
        return "CholecT50"
    elif "copesd" in file_name:
        return "CoPESD"
    elif "nurvid" in file_name:
        return "NurViD"
    else:
        # Try to detect from first few records
        return None

def detect_dataset_from_data(data):
    """
    Detect dataset from data content
    """
    # Sample a few records to detect dataset
    sample_records = list(data.values())[:5]
    
    for record in sample_records:
        if "data_source" in record:
            return record["data_source"]
        
        # Check ground truth patterns
        gnd = record.get("gnd", "").strip().lower()
        
        if gnd in [action.lower() for action in AVOS_ACTIONS]:
            return "AVOS"
        elif gnd in [action.lower() for action in T50_PHASES]:
            return "CholecT50"
        elif gnd in [action.lower() for action in TOTAL_NEW_ACTION_LIST]:
            return "CoPESD"
        elif any(gnd in [action.lower() for actions in NURVID_PROCEDURE_ACTIONS.values() for action in actions]):
            return "NurViD"
    
    return None

def get_action_list_for_dataset(dataset, procedure=None):
    """
    Get action list for specific dataset
    """
    if dataset == "AVOS":
        return AVOS_ACTIONS
    elif dataset == "CholecT50":
        return T50_PHASES
    elif dataset == "CoPESD":
        return TOTAL_NEW_ACTION_LIST
    elif dataset == "NurViD":
        if procedure and procedure in NURVID_PROCEDURE_ACTIONS:
            return NURVID_PROCEDURE_ACTIONS[procedure]
        else:
            # Return all unique actions across all procedures
            all_actions = set()
            for actions in NURVID_PROCEDURE_ACTIONS.values():
                all_actions.update(actions)
            return sorted(list(all_actions))
    elif dataset == "EgoSurgery":
        # EgoSurgery uses free-form actions, return empty list
        return []
    else:
        return []

def normalize_action_text(text, dataset):
    """
    Normalize action text based on dataset-specific mappings
    """
    text = text.strip()
    
    if dataset == "CoPESD":
        # Apply CoPESD action mapping for backward compatibility
        if text in COPESD_ACTION_MAPPING:
            return COPESD_ACTION_MAPPING[text]
    
    return text

def create_class_map_for_dataset(actions):
    """
    Create class map for given action list
    """
    return {action: idx for idx, action in enumerate(actions)}


def group_records_by_dataset(data):
    """Group next_action records by dataset for per-dataset evaluation."""
    from dataset_utils import get_dataset_name
    dataset_groups = defaultdict(list)

    for key, record in data.items():
        qa_type = record.get('qa_type', '')
        if 'next_action' not in qa_type.lower():
            continue

        # Detect dataset
        dataset = get_dataset_name(record)

        # Extract procedure for NurViD
        procedure = None
        if dataset == "NurViD":
            question_lower = record.get("question", "").lower()
            for proc_name in NURVID_PROCEDURE_ACTIONS.keys():
                if proc_name.lower() in question_lower:
                    procedure = proc_name
                    break

        # Restructure record to only include needed fields (consistent with Qwen2.5-VL)
        record_data = {
            "answer": record.get("answer", ""),
            "gnd": record.get("gnd", ""),
            "question": record.get("question", ""),
            "video_id": record.get("metadata", {}).get("video_id", record.get("video_id", "")),
            "procedure": procedure
        }

        dataset_groups[dataset].append(record_data)

    return dict(dataset_groups)




def evaluate_dataset_next_action(dataset_name, records):
    """Evaluate next_action for a specific dataset with semantic similarity."""
    print(f"\nEvaluating {dataset_name} ({len(records)} records)...")

    # Group records by procedure (for NurViD)
    procedure_groups = defaultdict(list)
    for record in records:
        procedure = record.get('procedure', 'default')
        procedure_groups[procedure].append(record)

    all_results_by_fps = defaultdict(list)

    # Evaluate each procedure group
    for procedure, proc_records in procedure_groups.items():
        # Get action list for this dataset/procedure
        actions = get_action_list_for_dataset(dataset_name, procedure)

        if not actions:
            # For datasets without predefined action lists (like EgoSurgery),
            # collect unique ground truth actions and use semantic similarity
            unique_actions = set()
            temp_records = []

            for record in proc_records:
                gnd_text = record.get('gnd', '')

                gnd_text = normalize_action_text(gnd_text, dataset_name)
                if gnd_text:
                    unique_actions.add(gnd_text)
                    temp_records.append((record, gnd_text))

            if not unique_actions:
                continue

            # Create action list from unique ground truths
            actions = sorted(list(unique_actions))
            CLASS_MAP = create_class_map_for_dataset(actions)
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            class_embeddings = semantic_model.encode(actions, convert_to_tensor=True)

            # Evaluate with semantic similarity
            for record, gnd_text in temp_records:
                fps = record.get('fps', record.get('metadata', {}).get('fps', 1.0))
                if isinstance(fps, str):
                    fps = float(fps)

                pred_text = normalize_action_text(record.get('answer', ''), dataset_name)

                # Get ground truth index
                gnd_idx = CLASS_MAP[gnd_text]

                # Determine prediction class using semantic similarity
                if pred_text in CLASS_MAP:
                    pred_idx = CLASS_MAP[pred_text]
                else:
                    # Use semantic similarity
                    pred_emb = semantic_model.encode(pred_text, convert_to_tensor=True)
                    sim_scores = util.cos_sim(pred_emb, class_embeddings)[0]
                    pred_idx = sim_scores.argmax().item()

                is_correct = (pred_idx == gnd_idx)
                all_results_by_fps[fps].append(1 if is_correct else 0)
            continue

        # Create class map and embeddings for semantic similarity
        CLASS_MAP = create_class_map_for_dataset(actions)
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        class_embeddings = semantic_model.encode(actions, convert_to_tensor=True)

        # Evaluate each record with semantic similarity
        for record in proc_records:
            fps = record.get('fps', record.get('metadata', {}).get('fps', 1.0))
            if isinstance(fps, str):
                fps = float(fps)

            pred_text = normalize_action_text(record.get('answer', ''), dataset_name)

            # Get ground truth from gnd field only (consistent with Qwen2.5-VL)
            gnd_text = record.get('gnd', '')

            gnd_text = normalize_action_text(gnd_text, dataset_name)

            # Skip if ground truth not in action list
            if not gnd_text or gnd_text not in CLASS_MAP:
                continue

            # Determine prediction class using semantic similarity
            if pred_text in CLASS_MAP:
                pred_idx = CLASS_MAP[pred_text]
            else:
                # Use semantic similarity as fallback
                pred_emb = semantic_model.encode(pred_text, convert_to_tensor=True)
                sim_scores = util.cos_sim(pred_emb, class_embeddings)[0]
                pred_idx = sim_scores.argmax().item()

            gnd_idx = CLASS_MAP[gnd_text]

            # Check if correct
            is_correct = (pred_idx == gnd_idx)
            all_results_by_fps[fps].append(1 if is_correct else 0)

    # Aggregate results by FPS
    aggregated = {}
    all_accuracies = []

    for fps, acc_list in all_results_by_fps.items():
        if acc_list:
            aggregated[f'fps_{fps}'] = {
                'accuracy': np.mean(acc_list),
                'count': len(acc_list)
            }
            all_accuracies.extend(acc_list)

    # Add overall accuracy across all FPS
    if all_accuracies:
        aggregated['overall'] = {
            'accuracy': np.mean(all_accuracies),
            'count': len(all_accuracies)
        }

    return aggregated


def main():
    """Main evaluation function for next_action."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_next_action.py <results_json_file>")
        print("Example: python eval_next_action.py results/model_results.json")
        sys.exit(1)

    output_file = sys.argv[1]
    print(f"Loading results from: {output_file}")

    with open(output_file, "r") as f:
        infer_output = json.load(f)

    dataset_records = group_records_by_dataset(infer_output)

    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} next_action records")

    if not any(dataset_records.values()):
        print("No next_action records found!")
        return

    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:
            results = evaluate_dataset_next_action(dataset_name, records)
            all_results[dataset_name] = results

    print(f"\n{'='*80}")
    print("NEXT ACTION EVALUATION SUMMARY")
    print(f"{'='*80}")

    all_accuracies = []
    total_correct = 0
    total_samples = 0
    for dataset_name, fps_results in all_results.items():
        if fps_results:
            print(f"\n{dataset_name}:")
            for fps_key, metrics in sorted(fps_results.items()):
                print(f"  {fps_key}:")
                for metric_name, value in metrics.items():
                    if metric_name != 'count':
                        print(f"    {metric_name}: {value:.4f}")
                    else:
                        print(f"    samples: {value}")
            if 'overall' in fps_results:
                acc = fps_results['overall'].get('accuracy', 0.0)
                count = fps_results['overall'].get('count', 0)
                all_accuracies.append(acc)
                total_correct += int(acc * count)
                total_samples += count

    return {
        'per_dataset': all_results,
        'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        'macro_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0
    }


if __name__ == "__main__":
    main()
