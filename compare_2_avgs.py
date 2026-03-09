import sys
import json
from tabulate import tabulate

def extract_model_name(label):
    return label.split("model_")[1].split("_")[0].split(".txt")[0]

def read_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compare_step(results1, results2, step, verbosity):
    output_results = []
    
    table_data = []
    sum_speedup = 0
    
    if len(results1["labels"]) != len(results2["labels"]):
        # Take only the common labels
        common_labels = set(results1["labels"]).intersection(set(results2["labels"]))
        print(f"Warning: Different number of labels. Comparing only common labels: {common_labels}")
        results1_indices = [i for i, label in enumerate(results1["labels"]) if label in common_labels]
        results2_indices = [i for i, label in enumerate(results2["labels"]) if label in common_labels]
        results1["labels"] = [results1["labels"][i] for i in results1_indices]
        results2["labels"] = [results2["labels"][i] for i in results2_indices]
        
        for s in results1["data"]:
            results1["data"][s] = [results1["data"][s][i] for i in results1_indices]
            
        for s in results2["data"]:
            results2["data"][s] = [results2["data"][s][i] for i in results2_indices]
        
    # Assuming the labels match
    for i, label in enumerate(results1["labels"]):
        model_name = extract_model_name(label)

        value1 = results1["data"][step][i]
        value2 = results2["data"][step][i]
            
        speedup = value2 / value1
        sum_speedup += speedup
        output_results.append({
            "model_name": model_name,
            "value1": value1,
            "value2": value2,
            "speedup": speedup
        })
        table_data.append([model_name, value1, value2, str(round(speedup, 2)) + "x"])
    
    if verbosity is not None:
        print(f"\nComparing {step}")
        if verbosity == "verbose":
            print(tabulate(table_data, headers=["Model Name", "Value1", "Value2", "SpeedUp"], tablefmt="grid"))
        print(f"Avg Speedup: {(sum_speedup / len(results1['labels'])):.2f}x")
        if verbosity == "verbose":
            print("\n\n")
    
    return output_results

def compare_files(file_path1, file_path2, verbosity="normal"):
    if verbosity not in [None, "normal", "verbose"]:
        raise ValueError("Verbosity should one of None, 'normal' or 'verbose'")
    
    output_results = {}
    
    results1 = read_json_file(file_path1)
    results2 = read_json_file(file_path2)
    
    for step in ["tokenizer_init", "model_init", "load_weights", "dynamo_transform_time", "graph_compile_cached", "kv_cache_profiling", "graph_capturing", "total_time", "actual_total_time"]:
        output_results[step] = compare_step(results1, results2, step, verbosity)
        
    return output_results
        
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <avg_comparison_results1.json> <avg_comparison_results2.json>")
        sys.exit(1)

    avg_comparison_results1_path = sys.argv[1]    
    avg_comparison_results2_path = sys.argv[2]
    
    
    compare_files(avg_comparison_results1_path, avg_comparison_results2_path)
    