from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import compare_2_archs

script_dir = Path(__file__).parent

case0_filepath = script_dir / "case0/avg_comparison_results.json"
case1_filepath = script_dir / "case1/avg_comparison_results.json"
verbosity = None
title = "Normalized to DRAM loading"
pdf_name = script_dir / "figure13.pdf"

compare_2_archs(case0_filepath, case1_filepath, verbosity, title, pdf_name)