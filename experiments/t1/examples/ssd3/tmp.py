import os
import json

models_map = {
    "falcon-7b": "falcon-7b",
    "Llama-2-7b-hf": "llama2-7b-hf",
    "Llama-2-13b-hf": "llama2-13b-hf",
    "Llama-3.2-3B": "llama3-3b",
    "Qwen1.5-0.5B": "qwen-0.5b",
    "Qwen1.5-1.8B": "qwen-1.8b",
    "Qwen1.5-4B": "qwen-4b",
    "Qwen1.5-7B": "qwen-7b",
    "Qwen1.5-14B": "qwen-14b",
    "Yi-6B": "yi-6b",
    "Yi-9B": "yi-9b",
}

for i in range(1, 6):
    dir_path = f"./iterations/{i}"
    for file in os.listdir(dir_path):
        
        if not file.startswith("output_"):
            continue
        old_filepath = os.path.join(dir_path, file)
        
        if file == "output_gpu-memory-utilization_0.9_model_Qwen1.5-14B_max_model_len_16752.txt":
            new_filepath = os.path.join(dir_path, "output_model_qwen-14b.txt")
            os.rename(old_filepath, new_filepath)
        else:
            old_model_name = file.split("_")[-1].split(".txt")[0]
            if old_model_name in models_map:
                new_model_name = models_map[old_model_name]
                new_file = f"output_model_{new_model_name}.txt"
                new_filepath = os.path.join(dir_path, new_file)
                
                os.rename(old_filepath, new_filepath)
            
    with open(f"{dir_path}/comparison_results.json", "r") as f:
        content = json.load(f)
        
    for i,l in enumerate(content["labels"]):
        
        if l == "output_gpu-memory-utilization_0.9_model_Qwen1.5-14B_max_model_len_16752.txt":
            new_file = "output_model_qwen-14b.txt"
        else:
            old_model_name = l.split("_")[-1].split(".txt")[0]
            new_model_name = models_map[old_model_name]
            new_file = f"output_model_{new_model_name}.txt"
            
        content["labels"][i] = new_file
        
    with open(f"{dir_path}/comparison_results.json", "w") as f:
        json.dump(content, f, indent=4)