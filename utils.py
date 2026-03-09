import torch
import subprocess
import matplotlib.pyplot as plt
import math
import json
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as stats
import numpy as np
import os
from compare_2_avgs import compare_files

script_dir = Path(__file__).parent

def clear_cache():
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Execute a shell command to clear system caches
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
    
def num_of_batch_sizes(cuda_graph_sizes):
    return len([1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)])

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extract_version(label):
    # Extract version numbers like v1.2.3
    match = re.search(r'v(\d+(?:\.\d+)*)', label)
    if match:
        return tuple(map(int, match.group(1).split(".")))
    return ()

def extract_label_value(label, key):
    return int(label.split(key)[1].split(".txt")[0].split("_")[1])

def get_model_info(label, key): 
    model_name = extract_model_name(label)
    with open(script_dir / "models_config.json", "r") as f:
        info = json.load(f)
    
    return info[model_name].get(key, 0)


def extract_nb_layers(label):
    return get_model_info(label, "layers")

def extract_vocab_size(label):
    return get_model_info(label, "vocab_size")

def extract_tokenizer_size(label):
    return get_model_info(label, "tokenizer_size")

def extract_head_size_x_layers(label):
    num_key_value_heads = get_model_info(label, "num_key_value_heads")
    num_attention_heads = get_model_info(label, "num_attention_heads")
    hidden_size = get_model_info(label, "hidden_size")
    num_layers = get_model_info(label, "layers")
    ffn_dim = get_model_info(label, "ffn_dim")
    
    delta = 1
    if num_key_value_heads == 1:
        delta = 1.4
    elif num_key_value_heads != num_attention_heads:
        delta = 1.1
    
    return 5e-8 * num_layers * (hidden_size * num_attention_heads + 0.5 * ffn_dim) * delta

def extract_model_name(label):
    return label.split("model_")[1].split("_")[0].split(".txt")[0]

def extract_cuda_graph_size(label):
    return extract_label_value(label, "cuda-graph-sizes")

def extract_max_seq_len_to_capture(label):
    return extract_label_value(label, "max-seq-len-to-capture")

def extract_batch_size(label):
    return num_of_batch_sizes(extract_cuda_graph_size(label))

def extract_batch_size_x_model_size(label):
    return extract_batch_size(label) * extract_model_size(label)

def extract_compiled_graph_size(label):
    return get_model_info(label, "compiled_graph_sizes")

def extract_kv_cache_profiling_overhead(label):
    is_moe = get_model_info(label, "moe")
    size = extract_model_size(label)
    if is_moe:
        size += size * 0.5
    return size

def extract_model_size(label):
    model_name = extract_model_name(label)
    pattern = re.compile(r'(\d+(?:\.\d+)?[bm])')
    match = pattern.search(model_name)
    if not match:
        raise ValueError(f"Model name {model_name} does not match the expected pattern.")
    size = match.group(1)
    num = float(size[:-1])
    if "m" in size:
        num /= 1000
    
    # Special Use cases
    if "gpt-oss-20b" in model_name:
        # This model is by default mxfp4 quantized
        # Dividing by 3.5 instead of 4 because of some overheads
        num /= 3.5
    
    return num

def get_sort_indices(labels, sort_by="model_size"):

    if sort_by == "model_size":
        sizes = [extract_model_size(label) for label in labels]
        return sorted(range(len(labels)), key=lambda i: sizes[i])
    elif sort_by == "alphabetical":
        return sorted(range(len(labels)), key=lambda i: natural_sort_key(labels[i]))
    elif sort_by == "version":
        return sorted(range(len(labels)), key=lambda i: extract_version(labels[i]))
    else:
        raise ValueError(f"Unsupported sort_by value: {sort_by}")
    
def get_labels_matrics(json_filepath, sort_by):
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)

    labels = json_data['labels']
    metrics = json_data['data']
        
    sort_indices = get_sort_indices(labels, sort_by)
    sorted_labels = [labels[i] for i in sort_indices]
    
    return metrics, sorted_labels, sort_indices

# Helper function
def draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename, func=None, excluded_labels=[]):
    # Color Map for each model:

    cbf_colors = [
        "#4477AA", "#66CCEE", "#228833", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499",
        "#332288", "#EE7733", "#BBBBBB", "#77AADD", "#EEDD88",
        "#FFAABB", "#99DDFF", "#55CC77", "#FFDD44", "#AA3377",
        "#1177AA", "#66AA77", "#CCBB44", "#BB5566", "#DDDDDD"
    ]

    models = [
        'qwen-0.5b', 'qwen-1.8b', 'qwen-4b', 'qwen-7b', 'qwen-14b',
        'qwen-14.3b', 'llama2-7b-hf', 'llama2-13b-hf', 'llama3-3b', 'llama3.1-8b-instruct',
        'falcon-7b', 'falcon-11b', 'yi-6b', 'yi-9b', 'mpt-7b',
        'mistral-7b', 'gemma-7b', 'gpt-oss-20b', 'deepseek-v2-lite-16b', 'deepseek-r1-distill-llama-8b',
        'deepseek-r1-distill-qwen-7b', 'granite3.3-8b-instruct', 'granite4.0-h-3b', 'granite4.0-h-32b', 'olmoe-7b'
    ]

    color_map = {model: cbf_colors[i % len(cbf_colors)] for i, model in enumerate(models)}

    all_values = []
    labels = []
    matric2_values = []
    colors = []

    for i, iter_dir in enumerate(os.listdir(iterations_path)):
        json_filepath = os.path.join(iterations_path, iter_dir, "comparison_results.json")

        metrics, sorted_labels, sort_indices = get_labels_matrics(json_filepath, sort_by)
        metric_values = [metrics[metric1][sort_indices[i]] for i in range(len(sorted_labels))]
        model_names = [extract_model_name(label) for label in sorted_labels]
        if func:
            model_metric_nb = [func(label) for label in sorted_labels]
        else:
            model_metric_nb = [get_model_info(label, metric2) for label in sorted_labels]

        # Now sort the values and model names based on the number of metric
        sorted_indices = sorted(range(len(model_metric_nb)), key=lambda i: model_metric_nb[i])
        metric_values_sorted = [metric_values[i] for i in sorted_indices]
        model_names_sorted = [model_names[i] for i in sorted_indices]
        model_metric_nb_sorted = [model_metric_nb[i] for i in sorted_indices]

        values = []
        for j, model_name in enumerate(model_names_sorted):
            if model_name in model_names_map:
                values.append(metric_values_sorted[j])
                if i == 0:
                    matric2_values.append(model_metric_nb_sorted[j])
                    labels.append(model_names_map[model_name])
                    
                    # Get colors from the color map using the "labels"
                    colors.append(color_map[model_name])
                
        all_values.append(values)
        
        
    # Compute mean and standard error for each bar
    all_values_np = np.array(all_values)  # shape: (num_runs, num_models)
    means = np.mean(all_values_np, axis=0)
    stderrs = stats.sem(all_values_np, axis=0)
    
    if len(model_names_map.keys()) < 2:
        colors = [cbf_colors[i] for i in range(len(labels))]


    # Main bar chart with error bars
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"{label} ({matric2_values[i]})" for i, label in enumerate(labels)],
        means,
        yerr=stderrs,
        capsize=10,
        color=colors,
        ecolor='black'
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(range(len(labels)))
    
    xtick_labels = []
    rotation = 30
    if metric2 != "size":
        xtick_labels = [f"{label} ({matric2_values[i]})" for i, label in enumerate(labels)]
    else:
        xtick_labels = labels
    
    if func and metric1 != "kv_cache_profiling":
        xtick_labels = matric2_values
        rotation = 0
    
    ax.set_xticklabels(xtick_labels, rotation=rotation, ha='right', fontsize=13)
    ax.set_ylim(0, max(means + stderrs) * ylim_multiplier)
    
    # Add value of each bar on top
    for bar, value in zip(bars, means):
        text_height = bar.get_height() / 2
        if bar.get_height() < 0.15 * max(means + stderrs):
            text_height = (bar.get_height() / 2) - 0.02 * max(means + stderrs)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_height,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=11,
            color='white',
        )
        
    # --- Inset plot with excluded points ---
    ax_inset = inset_axes(
        ax, width="30%", height="30%", 
        loc='upper left', 
        bbox_to_anchor=(0.1, -0.04, 1, 1), 
        bbox_transform=ax.transAxes
    )

    X = np.array(matric2_values).reshape(-1, 1)
    y = means

    # Exclude selected points before fitting
    mask = np.array([lbl not in excluded_labels for lbl in labels])
    X_fit = X[mask]
    y_fit = y[mask]

    reg = LinearRegression()
    reg.fit(X_fit, y_fit)
    y_pred = reg.predict(X)

    # Scatter + regression line
    for i in range(len(matric2_values)):
        ax_inset.scatter(matric2_values[i], means[i], color=colors[i], s=15)
    ax_inset.plot(X, y_pred, color='black', linestyle='--', linewidth=1)

    ax_inset.set_xlabel(x2label, fontsize=10)
    ax_inset.set_ylabel(y2label, fontsize=10)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    
    # --- Pearson Correlation Coefficient ---
    mask = np.array([lbl not in excluded_labels for lbl in labels])
    X_flat = np.array(matric2_values)[mask]
    y_flat = np.array(means)[mask]

    pcc, _ = stats.pearsonr(X_flat, y_flat)
    
    ax.text(
        0.6, 0.95,
        f"PCC = {pcc:.2f}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
    )

    plt.savefig(
        f"{filename}.pdf",
        format="pdf",
        bbox_inches="tight",
        transparent=True
    )
    plt.show()
    
def compare_2_archs(filepath1, filepath2, verbosity, title, pdf_name, custom_bars=None):
    
    if not os.path.exists(filepath1):
        print(f"File {filepath1} does not exist")
        return
    
    if not os.path.exists(filepath2):
        print(f"File {filepath2} does not exist")
        return
        
    steps_map = {
        "framework_bootstrap": "Framework Bootstrap",
        "tokenizer_init": "Tokenizer Init",
        "model_init": "Model Init",
        "load_weights": "Load Weights",
        "dynamo_transform_time": "Dynamo Transform",
        "graph_compile_cached": "Load Compiled Graphs",
        "kv_cache_profiling": "KV Cache Profiling",
        "graph_capturing": "Graph Capturing",
        "actual_total_time": "Total Time",
    }
    
    output_results = compare_files(filepath1, filepath2, verbosity)
    if custom_bars:
        for key,values in custom_bars.items():
            output_results[key] = values
            
    # Order output_results based on steps_map order
    output_results = {k: output_results[k] for k in steps_map.keys() if k in output_results}

    steps = list(output_results.keys())
    avg_speedups = []
    stderr_speedups = []
    xticklabels = []
    colors = []
    for step in steps:
        if step in steps_map:
            speedups = [entry["speedup"] for entry in output_results[step]]
            avg_speedups.append(np.mean(speedups))
            stderr_speedups.append(stats.sem(speedups))
            xticklabels.append(steps_map[step])
            # Make the last bar (Total Time) red, others blue
            colors.append('#666666' if steps_map[step] == 'Total Time' else '#1f77b4')

    x = np.arange(len(xticklabels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, avg_speedups, yerr=stderr_speedups, capsize=10, color=colors, ecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontsize=13)
    ax.set_ylabel(title, fontsize=15)

    # Annotate each bar with its value
    for bar, value in zip(bars, avg_speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.2f}", ha='center', va='bottom', fontsize=10, color='white')

    plt.tight_layout()
    plt.savefig(
        pdf_name,
        format="pdf",
        bbox_inches="tight",
        transparent=True
    )
    plt.show()
    

def draw_graph(json_file_path, sort_by, included_keys=None):
    
    metrics, sorted_labels, sort_indices = get_labels_matrics(json_file_path, sort_by)
    
    if included_keys is not None:
        metrics = {k: v for k, v in metrics.items() if k in included_keys}
    else: # Default
        # No need for model_loading (it's just a sum of load_weights and model_init)
        del metrics['model_loading']
        
        # No need for torch.compile (it's just a sum of dynamo_transfer_time and graph_compile_general_shape)
        del metrics['torch.compile']
        
        # No need for init_engine (it's just a sum of kv_cache_profiling and graph_capturing)
        del metrics['init_engine']
        
        # No need for kv_cache_init (it's just constant time for all models ~0.01s)
        if 'kv_cache_init' in metrics:
            del metrics['kv_cache_init']
        
        # No need for actual_total_time (it's just same as total_time + 14s platform detection overhead)
        del metrics['actual_total_time']

    # Number of metrics (keys in 'data')
    num_metrics = len(metrics)
    cols = 3  # number of columns in the subplot grid
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    # Custom naming for batch size
    for i, label in enumerate(sorted_labels):
        if "cuda-graph-sizes" in label:
            cuda_graph_sizes = extract_cuda_graph_size(label)
            num_batches = num_of_batch_sizes(cuda_graph_sizes)
            sorted_labels[i] = label.replace(f"cuda-graph-sizes_{cuda_graph_sizes}", f"batch-size_{num_batches}")

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        cleaned_values = [v if v is not None else 0 for v in values]
        sorted_values = [cleaned_values[i] for i in sort_indices]
        bars = ax.bar(range(len(sorted_labels)), sorted_values)
        ax.set_title(metric_name)
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(
            [label.replace("output_", "").replace("model_", "").replace(".txt", "") for label in sorted_labels],
            rotation=90,
            fontsize=10
        )
        ax.set_ylabel("Time (s)")
        max_val = max(cleaned_values)
        upper = max_val * 1.15 if max_val > 0 else 1
        ax.set_ylim(0, upper)
        
        # Annotate each bar with its height
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
            offset = 0.05 * max(cleaned_values)  # 5% of max value
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset if height > 0 else offset,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Increase vertical spacing
    fig.tight_layout(h_pad=2.5)

    plt.show()
    

def draw_multiple_relationships(values_list, sorted_labels_list, titles, xlabel, y_axis_func):
    assert len(values_list) == len(sorted_labels_list) == len(titles), "All input lists must have the same length"

    num_plots = len(values_list)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, (values, sorted_labels, title) in enumerate(zip(values_list, sorted_labels_list, titles)):
        ax = axes[idx]

        # Data from the table
        models = [extract_model_name(sorted_labels[i]) for i in range(len(sorted_labels))]
        param_sizes = np.array([y_axis_func(sorted_labels[i]) for i in range(len(sorted_labels))]).reshape(-1, 1)
        times = np.array(values)

        # Linear regression
        reg = LinearRegression().fit(param_sizes, times)
        predicted_times = reg.predict(param_sizes)

        # Plotting
        ax.scatter(param_sizes, times, color='blue', label='Actual Data')
        ax.plot(param_sizes, predicted_times, color='red', linestyle='--', label='Linear Fit')
        for i, label in enumerate(models):
            ax.text(param_sizes[i], times[i] + 0.2, label, ha='center', fontsize=8)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def draw_relationship(json_filepath, order_by, keys, xlabel, y_axis_func):
    metrics, sorted_labels, sort_indices = get_labels_matrics(json_filepath, order_by)

    values_list = []
    sorted_labels_list = []
    titles = []

    for k in keys:
        values_list.append([metrics[k][sort_indices[i]] for i in range(len(sorted_labels))])
        sorted_labels_list.append(sorted_labels)
        titles.append(f"{k}")

    draw_multiple_relationships(
        values_list=values_list,
        sorted_labels_list=sorted_labels_list,
        titles=titles,
        xlabel=xlabel,
        y_axis_func=y_axis_func
    )
    

def draw_metric_wrt_metric(configs, x_axis):
    num_plots = len(configs)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, config in enumerate(configs):
        json_filepath = config['json_filepath']
        metric = config['metric']

        metrics, sorted_labels, sort_indices = get_labels_matrics(json_filepath, "model_size")

        metric_values = [metrics[metric][sort_indices[i]] for i in range(len(sorted_labels))]
        model_names = [extract_model_name(label) for label in sorted_labels]
        model_metric_nb = [get_model_info(label, x_axis["key"]) for label in sorted_labels]

        # Now sort the values and model names based on the number of metric
        sorted_indices = sorted(range(len(model_metric_nb)), key=lambda i: model_metric_nb[i])
        metric_values_sorted = [metric_values[i] for i in sorted_indices]
        model_names_sorted = [model_names[i] for i in sorted_indices]
        model_metric_nb_sorted = [model_metric_nb[i] for i in sorted_indices]

        labels = [f"{model_names_sorted[i]} ({model_metric_nb_sorted[i]})" for i in range(len(model_names_sorted))]

        ax = axes[idx]
        ax.bar(labels, metric_values_sorted)
        ax.set_xlabel("Model Names")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {x_axis['title']}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()