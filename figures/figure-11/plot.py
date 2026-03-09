from pathlib import Path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import compare_2_archs

script_dir = Path(__file__).parent

cpu0_filepath = script_dir / "cpu0/avg_comparison_results.json"
cpu1_filepath = script_dir / "cpu1/avg_comparison_results.json"
verbosity = None
title = "Normalized to node n1 values"
pdf_name = script_dir / "figure11.pdf"

compare_2_archs(cpu0_filepath, cpu1_filepath, verbosity, title, pdf_name)
