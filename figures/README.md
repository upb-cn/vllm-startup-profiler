## Setup

1) Clone the repo:
```bash
git clone https://github.com/upb-cn/vllm-startup-profiler
cd vllm-startup-profiler
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Apply custom vllm changes (addition of profiling logs):
```bash
python3 apply_vllm_changes.py
```

4) Download all models used in our experiments. This will download multiple LLMs. This will need around 600GB of disk space.
```bash
python3 download_models.py <hf_token>
# `<hf_token>` is your HuggingFace token. Note that some models require applying to access before downloading.
```

## System Specifications

- In the following table, you can find the specification of our four nodes n1-n4.
- All experiments were run on node n1, with the following exceptions:
    - In Figure 10, we compare the results between n1 and n2.
    - In Figure 11 & 12, we compare the results between n1 and n3.

|        | node1 (n1)               | node2 (n2)               | node3 (n3)                                 | node4 (n4)                                 |
|--------|--------------------------|--------------------------|--------------------------------------------|--------------------------------------------|
| CPU    | AMD EPYC 9354 (32C)      | AMD EPYC 9354 (32C)      | 2× Intel Xeon Platinum 8568Y+ (2×48C)      | 2× Intel Xeon Gold 5520+ (2×28C)           |
| GPU    | H100 NVL                 | L40S                     | H100                                       | L40S                                       |
| DRAM   | DDR5 251GB               | DDR5 251GB               | DDR5 2TB                                   | DDR5 2TB                                   |
| OS     | Debian 12                | Debian 12                | Red Hat Enterprise Linux (RHEL) 9          | Red Hat Enterprise Linux (RHEL) 9          |
| Kernel | 6.1.0-40-amd64           | 6.1.0-40-amd64           | 5.14.0-503.34.1.el9_5                      | 5.14.0-503.34.1.el9_5                      |
| Python | 3.11.2                   | 3.11.2                   | 3.12                                       | 3.12                                       |
| CUDA   | 12.6 (Driver 580.82.07)  | 12.6 (Driver 580.82.07)  | 12.8 (Driver 570.124.06)                   | 12.8 (Driver 570.124.06)                   |
| PyTorch| 2.7.1+cu126              | 2.7.1+cu126              | 2.7.1+cu126                                | 2.7.1+cu126                                |
| vLLM   | v0.10.1.1                | v0.10.1.1                | v0.10.1.1                                  | v0.10.1.1                                  |
| SSD    | --                       | --                       | 4× PCIe 5.0 SSDs                           | --                                         |
| FS     | --                       | --                       | XFS/LVM with RAID-0 mirror                 | --                                         |


## Reproduce Figures

Note that even with identical environments, the results may still encounter small variations

In order to reproduce any figure:
```bash
cd figures
bash run_figure.sh <num>
```
Where `<num>` is one of the following: `1`, `2`, `7`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `17`, `rest`.
Where `rest` include Figures 3, 4, 5, 6, 8 and 17

You will find the figure in `figures/figure-<num>/figure<num>.pdf`

### Important Notes

- Figure 1 is a little bit tricky. In order to reproduce, we have to install vllm in different versions. In order to manage this, we will use python virtual environment and create a new environment for every version.
- Figure 10 requires running the experiments on 2 different GPUs. 
    - If you have 2 GPUs on the same machine, then you can do this by setting the environment variable `CUDA_VISIBLE_DEVICES` before running the `run_figure.sh` script. You also need to pass the index of the GPU to the script. For example:
    
        ```bash
        # Run on first GPU
        CUDA_VISIBLE_DEVICES="0" bash run_figure.sh 10 0
        # Run on second GPU
        CUDA_VISIBLE_DEVICES="1" bash run_figure.sh 10 1
        ```
    - If you have 2 GPUs on different machines, then:
        - Run the command `bash run_figure.sh 10 0` on the first machine.
        - Run the command `bash run_figure.sh 10 1` on the second machine.
        - Move the output files from the second machine (e.g. `./figures/figure-10/gpu1`) to the same path in the first machine.
        - Then to plot the figure, you can simply run `python3 figures/figure-10/plot.py`
    
- Figure 11 follows a similar path to Figure 10. It requires running the experiments on 2 different CPUs.
    - You should first run the first experiment on the first machine using this command: `bash run_figure.sh 11 0`
    - Then do the same thing on the second machine: `bash run_figure.sh 11 1`
    - Now move the output files from the second machine (e.g. `./figures/figure-11/cpu1`) to the same path in the first machine
    - Then to plot the figure, you can simply run `python3 figures/figure-11/plot.py`
- Again, for Figure 13, it's the same process as before:
    - First run the experiments on the first machine: `bash run_figure.sh 13 0`. This will represent the case where all weights are retrieved from RAM.
    - Then run the experiments on the second machine: `bash run_figure.sh 13 1`. This will represent the case where all weights are retrieved from the SSD. However, note that now we have clear RAM memory after each run. To do so, the script will automatically run the following command: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, which requires sudo access. **NOTE: This might affect other running processes. Therefore, it's better to run this command alone.**
    - Now move the output files from the second machine (e.g. `./figures/figure-13/case1`) to the same path in the first machine
    - Then to plot the figure, you can simply run `python3 figures/figure-13/plot.py`

