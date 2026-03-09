from huggingface_hub import snapshot_download
import os
import sys
import subprocess

def download_model_data(model_name, local_dir_path, token = None):
    os.makedirs(local_dir_path, exist_ok=True)
    
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir_path,
        local_dir_use_symlinks=False,
        token=token
    )
    print("Model has been downloaded!")



if __name__ == '__main__':
    configs = [
        {"model_name": "meta-llama/Llama-2-7b-hf", "local_dir_path": "./models/llama2-7b-hf"},
        {"model_name": "meta-llama/Llama-2-13b-hf", "local_dir_path": "./models/llama2-13b-hf"},
        {"model_name": "meta-llama/Llama-3.2-3B", "local_dir_path": "./models/llama3-3b"},
        {"model_name": "tiiuae/falcon-7b", "local_dir_path": "./models/falcon-7b"},
        {"model_name": "Qwen/Qwen1.5-0.5B", "local_dir_path": "./models/qwen-0.5b"},
        {"model_name": "Qwen/Qwen1.5-14B", "local_dir_path": "./models/qwen-14b"},
        {"model_name": "Qwen/Qwen1.5-1.8B", "local_dir_path": "./models/qwen-1.8b"},
        {"model_name": "Qwen/Qwen1.5-4B", "local_dir_path": "./models/qwen-4b"},
        {"model_name": "Qwen/Qwen1.5-7B", "local_dir_path": "./models/qwen-7b"},
        {"model_name": "01-ai/Yi-6B", "local_dir_path": "./models/yi-6b"},
        {"model_name": "01-ai/Yi-9B", "local_dir_path": "./models/yi-9b"},
        {"model_name": "tiiuae/falcon-11B", "local_dir_path": "./models/falcon-11b"},
        {"model_name": "mistralai/Mistral-7B-v0.1", "local_dir_path": "./models/mistral-7b"},
        {"model_name": "Qwen/Qwen1.5-MoE-A2.7B", "local_dir_path": "./models/qwen-14.3b"},
        {"model_name": "google/gemma-7b", "local_dir_path": "./models/gemma-7b"},
        {"model_name": "openai/gpt-oss-20b", "local_dir_path": "./models/gpt-oss-20b"},
        {"model_name": "ibm-granite/granite-3.3-8b-instruct", "local_dir_path": "./models/granite3.3-8b-instruct"},
        {"model_name": "ibm-granite/granite-4.0-h-small", "local_dir_path": "./models/granite4.0-h-32b"},
        {"model_name": "ibm-granite/granite-4.0-h-micro", "local_dir_path": "./models/granite4.0-h-3b"},
        {"model_name": "deepseek-ai/DeepSeek-V2-Lite", "local_dir_path": "./models/deepseek-v2-lite-16b"},
        {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "local_dir_path": "./models/deepseek-r1-distill-llama-8b"},
        {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "local_dir_path": "./models/deepseek-r1-distill-qwen-7b"},
        {"model_name": "facebook/opt-6.7b", "local_dir_path": "./models/opt-6.7b"},
    ]
    
    os.makedirs("./models", exist_ok=True)
    
    token = sys.argv[1] if len(sys.argv) > 1 else None
    if token is None:
        raise ValueError("Please enter a hugging face token. Usage: python3 download_models.py <token>")
    for config in configs:
        download_model_data(config["model_name"], config["local_dir_path"], token)
        
    # Download 4 tensorized models for Figure 13
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    
    models = {
        "meta-llama/Llama-2-13b-hf": "./models/llama2-13b-hf-tensorized",
        "meta-llama/Llama-2-7b-hf": "./models/llama2-7b-hf-tensorized",
        "01-ai/Yi-6B": "./models/yi-6b-tensorized",
        "tiiuae/falcon-7b": "./models/falcon-7b-tensorized",
    }

    for repo_id,dir_path in models.items():
        cmd = [
            "python3",
            "tensorize.py",
            "--model", repo_id,
            "serialize",
            "--serialized-directory", dir_path
        ]

        subprocess.run(cmd, env=env, check=True)
    
    