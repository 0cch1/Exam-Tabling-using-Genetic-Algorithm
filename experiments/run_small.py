import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ga_timetabling import read_instance, compare_settings

inst = read_instance("instances/small-2.txt")
compare_settings(inst)

# Output fitness-over-generations plot (assignment requirement)
from src.ga_timetabling import run_ga, plot_curve
_, _, _, _, curve = run_ga(inst, seed=42)
plot_curve(curve)
