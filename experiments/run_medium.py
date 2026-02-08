import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ga_timetabling import read_instance, compare_settings_for_medium

inst = read_instance("instances/medium-1.txt")
compare_settings_for_medium(inst)
