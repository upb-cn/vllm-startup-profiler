import json
import os
import subprocess
import shlex
import sys
from itertools import product
from time import sleep

def clear_cache():
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()

def is_array(val):
    return isinstance(val, list)

def build_command(base_params, config_dir):
    args = []
    output_suffix = []
    for key, value in base_params.items():
        flag = f"--{key}"
        # if key in combination:
        # value = combination[key]
        val_str = str(value)
        # If value is a path, use only the basename
        if '/' in val_str:
            val_str = os.path.basename(val_str.rstrip("/"))
        output_suffix.append(f"{key}_{val_str.replace(' ', '_')}")
        if value is None:
            args.append(flag)
        elif isinstance(value, str):
            args.append(flag)
            if value.strip().startswith("{"):  # detect JSON value
                args.append(value)
            else:
                args.extend(shlex.split(value))
        else:
            args.extend([flag, str(value)])

    command = ["python3", "engine.py"] + args
    if len(output_suffix) == 0:
        output_file = "output.txt"
    else:
        output_file = f"output_{'_'.join(output_suffix)}.txt"
    output_file = os.path.join(config_dir, output_file)
    full_cmd = f"{' '.join(shlex.quote(arg) for arg in command)} | tee {shlex.quote(output_file)}"
    return full_cmd, output_file

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <config_file.json>")
        sys.exit(1)
        
    clear_cache()
        
    output_files = []

    config_file = sys.argv[1]
    config_dir = os.path.dirname(os.path.abspath(config_file))
    with open(config_file, "r") as f:
        config = json.load(f)

    # Export env vars
    for k, v in config.get("env", {}).items():
        os.environ[k] = str(v)

    default_params = config.get("default_params", {})
    configs = config.get("configs", [])

    for cfg in configs:
        # Merge default_params with cfg, cfg overrides defaults
        merged_params = {**default_params, **cfg}

        static_params = {k: v for k, v in merged_params.items() if not is_array(v)}
        tuned_params = {k: v for k, v in merged_params.items() if is_array(v)}

        if not tuned_params:
            # No arrays: run once
            cmd, output_file = build_command(static_params, config_dir)
            output_files.append(output_file)
            print("Running:", cmd)
            subprocess.run(cmd, shell=True, check=True)
            continue

        # Cartesian product of all array-valued params
        tuned_keys = list(tuned_params.keys())
        tuned_values = list(product(*[tuned_params[k] for k in tuned_keys]))

        for values in tuned_values:
            combination = dict(zip(tuned_keys, values))
            cmd, output_file = build_command({**static_params, **combination}, config_dir)
            output_files.append(output_file)
            print("Running:", cmd)
            subprocess.run(cmd, shell=True, check=True)


    # Run compare_logs.py on all output files
    compare_script = os.path.join(os.path.dirname(__file__), "compare_logs.py")
    compare_cmd = ["python3", compare_script] + output_files + [os.path.join(config_dir, "comparison_results.json")]
    if not os.path.exists(os.path.dirname(compare_cmd[-1])):
        os.makedirs(os.path.dirname(compare_cmd[-1]))
    print("Comparing logs...")
    sleep(3)  # Give some time for the previous commands to finish
    subprocess.run(compare_cmd, check=True)

if __name__ == "__main__":
    main()
