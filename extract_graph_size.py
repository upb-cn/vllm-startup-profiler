import sys
import os
import json
from utils import extract_model_name

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <iteration_output_dir_path> <models_config.json_path>")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    models_config_path = sys.argv[2]
    
    sizes_map = {}
    
    for file in os.listdir(output_dir):
        if not file.endswith(".txt"):
            continue
        
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            graph_paths = []
            for line in lines:
                if "from inductor via handle" in line:
                    handle_value = line.split("handle ")[1]
                    graph_path = handle_value.split(", '")[1][:-3]
                    graph_paths.append(graph_path)
                    
        
        # Extract size of each graph file
        graph_sizes = []
        for p in [graph_paths[0], graph_paths[1], graph_paths[-1]]:
            size_bytes = os.path.getsize(p)
            size_kb = size_bytes / 1024
            graph_sizes.append(round(size_kb, 2))
        
        sizes_map[file] = graph_sizes
        
    
    with open(models_config_path, "r") as f:
        models_config = json.load(f)
    
    for label, sizes in sizes_map.items():
        model_name = extract_model_name(label)
        if model_name in models_config:
            # There is 3 different compiled graph
            # One for the first layer, one for the last layer
            # and one for all the middle layers
            nb_layers = models_config[model_name]["layers"]
            total_size = sizes[0] + sizes[2] + sizes[1] * (nb_layers - 2)
            models_config[model_name]["compiled_graph_sizes"] = round(total_size)
        else:
            print(f"Warning: Model {model_name} not found in models_config.json")
        
    with open(models_config_path, "w") as f:
        json.dump(models_config, f, indent=4)
        
if __name__ == "__main__":
    main()