import sys
import os
import json
from profile_log import main as analyze_main

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <log1.txt> <log2.txt> <output_path> ...")
        sys.exit(1)

    log_paths = sys.argv[1:-1]    
    output_path = sys.argv[-1]

    all_dicts = []
    column_labels = []

    for path in log_paths:
        try:
            parsed = analyze_main(path)
            all_dicts.append(parsed)
            column_labels.append(os.path.basename(path))
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            continue

    if not all_dicts:
        print("No logs were successfully parsed.")
        return


    custom_order = [
        "detect_platform",
        "llm_imports",
        "get_model_info",
        "worker_init",
        "framework_bootstrap",
        "load_weights",
        "model_init",
        "model_loading",
        "dynamo_transform_time",
        "graph_compile_general_shape",
        "graph_compile_cached",
        "torch.compile",
        "kv_cache_profiling",
        "graph_capturing",
        "init_engine",
        "tokenizer_init",
        "total_time",
        "actual_total_time",
    ]
    
    table = []
    for key in custom_order:
        row = [key]
        for d in all_dicts:
            val = d.get(key, None)
            if val is None:
                v = "--"
            else:
                v = round(val, 2)
            row.append(v)
        table.append(row)
    
    # Save as json
    results = {
        "labels": column_labels,
        "data": {},
    }
    for key in custom_order:
        results["data"][key] = [d.get(key, None) for d in all_dicts]
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Comparison results saved to {output_path}")

if __name__ == "__main__":
    main()
