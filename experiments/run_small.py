import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ga_timetabling import read_instance, compare_settings

inst = read_instance("instances/small-2.txt")
compare_settings(inst)
