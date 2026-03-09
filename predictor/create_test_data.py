import json
import argparse
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import extract_model_name

def create_test_data(avg_comparison_results, models_config, validation_models, output_path):
    
    with open(avg_comparison_results, "r") as f:
        results = json.load(f)
        
    with open(models_config, "r") as f:
        models_config_data = json.load(f)
        
    validation_models_list = validation_models.split(",")
    
    data = {
        "train": [],
        "validation": []
    }
    
    for i, l in enumerate(results["labels"]):
        model_name = extract_model_name(l)
        if model_name in validation_models_list:
            split = "validation"
        else:
            split = "train"
        
        entry = {
            "label": model_name,
            "time": results["data"]["actual_total_time"][i],
            "batch_size": 1
        }
        
        # Now add all models_config parameters to entry
        model_config = models_config_data[model_name]
        for key, value in model_config.items():
            entry[key] = value
            
        data[split].append(entry)
    
    # Write to test_data.json
    with open(f"{output_path}/test_data.json", "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"test_data.json is written to {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictor")
    parser.add_argument("--avg_comparison_results", help="Path to avg_comparison_results.json, used to create test_data.json", type=Path, required=True)
    parser.add_argument("--models_config", help="Path to models_config.json", type=Path, required=True)
    parser.add_argument("--output_path", help="Path to where test_data.json is going to be saved", type=Path, required=True)
    parser.add_argument("--validation_models", help="Comma separated models to be used as validation. Rest are used as training samples", type=str, required=True)
    args = parser.parse_args()

    create_test_data(args.avg_comparison_results, args.models_config, args.validation_models, args.output_path)