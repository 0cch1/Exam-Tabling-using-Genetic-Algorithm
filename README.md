# Exam Timetabling using Genetic Algorithms

Genetic Algorithm for exam timetabling. Assigns exams to timeslots so that no student has two exams in the same slot (hard constraint) and the number of consecutive exams per student is minimised (soft constraint).

---

## Requirements

- Python 3.9+
- numpy, matplotlib

```bash
pip install -r requirements.txt
```

---

## Code Structure

### Directory layout

| Directory / File | Purpose |
|------------------|--------|
| **src/** | Core GA implementation (single module). |
| **src/ga_timetabling.py** | All algorithm logic: instance parsing, fitness, GA operators, and main loop. |
| **instances/** | Input instance files (plain text: first line `N K M`, then M×N enrolment matrix). |
| **experiments/** | Scripts to run the GA on each instance and reproduce results. |
| **results/** | Optional output directory for logs and plots (small/, medium/, test_case1/). |
| **requirements.txt** | Python dependencies. |

### Main components in `src/ga_timetabling.py`

- **Data & I/O**
  - `ExamInstance` — Dataclass holding N, K, M, enrolment matrix, conflict matrix, and per-student exam lists.
  - `read_instance(path)` — Parses an instance file and returns an `ExamInstance`.

- **Fitness**
  - `hard_violations(inst, sol)` — Counts student exam conflicts (same timeslot).
  - `soft_consecutive_exams(inst, sol)` — Counts consecutive exams per student.
  - `evaluate(inst, sol, alpha)` — Returns combined fitness (α·H + S), H, and S.

- **Repair & GA operators**
  - `repair_solution(inst, sol, rng)` — Resolves hard conflicts by moving exams to less conflicting slots.
  - `init_population(inst, pop_size, rng)` — Random initial solutions (optionally repaired).
  - `tournament_select(pop, fit, rng, k)` — Tournament selection (default k=3).
  - `uniform_crossover(p1, p2, rng)` — Uniform crossover; offspring may be repaired after.
  - `mutate(sol, K, rng, pm)` — Per-gene mutation with probability pm.

- **Main loop & experiments**
  - `run_ga(inst, pop_size, generations, pc, pm, ...)` — Single GA run; supports elitism (`n_elite`) and repair; returns best solution, fitness, H, S, and fitness-per-generation curve.
  - `run_multiple(inst, runs, ...)` — Multiple runs with different seeds; prints summary statistics (best, mean, worst, std) and optional convergence plot.
  - `plot_curve(best_curve, title)` — Plots best fitness vs generation.

- **Experiment helpers** (call `run_multiple` with different parameters)
  - `compare_settings(inst)` — Baseline on small instance.
  - `compare_settings_for_medium(inst)` — Two parameter configs on medium instance.
  - `run_test_case1_baseline(inst)` — Baseline on test_case1.

---

## How to run

From the repository root:

```bash
# Small instance (10 runs + convergence plot)
python experiments/run_small.py

# Medium instance (two parameter settings, 10 runs each)
python experiments/run_medium.py

# Test case 1 (baseline, 10 runs)
python experiments/run_test_case1.py
```

Each script loads the corresponding instance from `instances/`, runs the GA (with multiple seeds where applicable), and prints best/mean/worst/std of the soft constraint cost. `run_small.py` also displays a fitness-over-generations plot.

---

## Instance file format

First line: `N K M` (number of exams, timeslots, students).  
Next M lines: N space-separated 0/1 values (enrolment matrix).  
Example: `instances/small-2.txt`, `instances/medium-1.txt`, `instances/test_case1.txt`.
