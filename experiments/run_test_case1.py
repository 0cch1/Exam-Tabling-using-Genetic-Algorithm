import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ga_timetabling import read_instance, run_test_case1_baseline

inst = read_instance("instances/test_case1.txt")
run_test_case1_baseline(inst)
