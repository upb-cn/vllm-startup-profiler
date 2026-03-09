import argparse
import time
import gc
import os
import psutil
import threading
import json
from pathlib import Path

script_dir = Path(__file__).parent

def monitor(stop_event, interval=1.0):
    process = psutil.Process(os.getpid())
    usage_log = []

    while not stop_event.is_set():
        process.cpu_percent(interval=interval)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        usage_log.append(per_core)

    with open(script_dir / "cpu_usage.json", "w") as jf:
        json.dump(usage_log, jf)

def initialize_engine(args: argparse.Namespace) -> "LLMEngine":
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor, daemon=True, args=(stop_event, 0.1))
    monitor_thread.start()
   
    start = time.time()
    engine = initialize_engine(args)
    end = time.time()
    print(f"initialize_engine took {end - start:.2f} seconds")
    
    stop_event.set()
    monitor_thread.join()
    print("Report written to cpu_report.txt")
    
    del engine
    gc.collect()


if __name__ == "__main__":
    from vllm import EngineArgs, LLMEngine
    from vllm.utils import FlexibleArgumentParser
    
    args = parse_args()
    main(args)