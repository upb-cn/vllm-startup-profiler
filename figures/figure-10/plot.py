from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import compare_2_archs

script_dir = Path(__file__).parent


gpu0_filepath = script_dir / "gpu0/avg_comparison_results.json"
gpu1_filepath = script_dir / "gpu1/avg_comparison_results.json"
title = "Normalized to node gpu0 values"
pdf_name = script_dir / "figure10.pdf"
compare_2_archs(gpu0_filepath, gpu1_filepath, None, title, pdf_name)