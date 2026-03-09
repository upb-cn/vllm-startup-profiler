import argparse
import time
import gc

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
   
    start = time.time()
    engine = initialize_engine(args)
    end = time.time()
    print(f"initialize_engine took {end - start:.2f} seconds")
    
    del engine
    gc.collect()


if __name__ == "__main__":
    from vllm import EngineArgs, LLMEngine
    
    # Backward-compatible import
    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        # Fallback for older vLLM versions
        FlexibleArgumentParser = argparse.ArgumentParser
    
    args = parse_args()
    main(args)