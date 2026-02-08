from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ExamInstance:
    N: int
    K: int
    M: int
    enroll: np.ndarray              # (M, N)
    conflict: np.ndarray            # (N, N) 0/1
    student_exams: List[np.ndarray] # per student: list of exams


def read_instance(path: str) -> ExamInstance:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 3:
            raise ValueError(f"Invalid header in {path}. Expected 'N K M', got: {header}")
        N, K, M = map(int, header)

        rows = []
        for i in range(M):
            line = f.readline()
            if not line:
                raise ValueError(f"File ended early: expected {M} rows, got {i}.")
            vals = line.strip().split()
            if len(vals) != N:
                raise ValueError(f"Row {i} has {len(vals)} entries, expected {N}.")
            rows.append([int(x) for x in vals])

    enroll = np.array(rows, dtype=np.int8)
    co = (enroll.T @ enroll).astype(np.int32)
    conflict = (co > 0).astype(np.int8)
    np.fill_diagonal(conflict, 0)

    student_exams = [np.flatnonzero(enroll[s]).astype(np.int32) for s in range(M)]
    return ExamInstance(N=N, K=K, M=M, enroll=enroll, conflict=conflict, student_exams=student_exams)


def hard_violations(inst: ExamInstance, sol: np.ndarray) -> int:
    H = 0
    for t in range(1, inst.K + 1):
        exams = np.flatnonzero(sol == t)
        if exams.size <= 1:
            continue
        sub = inst.conflict[np.ix_(exams, exams)]
        H += int(np.triu(sub, k=1).sum())
    return H


def soft_consecutive_exams(inst: ExamInstance, sol: np.ndarray) -> int:
    S = 0
    for exams in inst.student_exams:
        if exams.size <= 1:
            continue
        slots = np.sort(sol[exams])
        S += int(np.sum((slots[1:] - slots[:-1]) == 1))
    return S


def evaluate(inst: ExamInstance, sol: np.ndarray, alpha: int = 10_000) -> Tuple[int, int, int]:
    H = hard_violations(inst, sol)
    S = soft_consecutive_exams(inst, sol)
    return alpha * H + S, H, S


# ---------- Repair (important) ----------
def repair_solution(inst: ExamInstance, sol: np.ndarray, rng: np.random.Generator, max_passes: int = 50) -> np.ndarray:
    """
    Try to eliminate hard conflicts by moving conflicting exams to least-conflicting slots.
    Stops early if no conflicts remain.
    """
    N, K = inst.N, inst.K
    sol = sol.copy()

    for _ in range(max_passes):
        moved_any = False

        # Find conflicts: any pair (a,b) with conflict[a,b]=1 and same slot.
        # We'll scan exams and see if each exam conflicts within its slot.
        for e in range(N):
            t = sol[e]
            # exams in same timeslot
            same_slot = np.flatnonzero(sol == t)
            if same_slot.size <= 1:
                continue
            # count conflicts of e with others in same slot
            conflicts_here = inst.conflict[e, same_slot].sum()
            if conflicts_here == 0:
                continue

            # Move e to best slot (least conflicts)
            best_t = t
            best_cost = None
            for cand_t in range(1, K + 1):
                if cand_t == t:
                    continue
                exams_in_cand = np.flatnonzero(sol == cand_t)
                cost = int(inst.conflict[e, exams_in_cand].sum()) if exams_in_cand.size > 0 else 0
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_t = cand_t
                    if best_cost == 0:
                        break

            # If multiple equal-best slots exist, this picks the first found; ok for now.
            if best_t != t:
                sol[e] = best_t
                moved_any = True

        if not moved_any:
            break

        # Early exit if already feasible
        if hard_violations(inst, sol) == 0:
            break

    return sol


# ---------- GA operators ----------
def init_population(inst: ExamInstance, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    pop = rng.integers(1, inst.K + 1, size=(pop_size, inst.N), dtype=np.int32)
    return pop


def tournament_select(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    idx = rng.integers(0, pop.shape[0], size=k)
    best = idx[np.argmin(fit[idx])]
    return pop[best].copy()


def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(size=p1.shape[0]) < 0.5
    c1 = p1.copy()
    c2 = p2.copy()
    c1[mask] = p2[mask]
    c2[mask] = p1[mask]
    return c1, c2


def mutate(sol: np.ndarray, K: int, rng: np.random.Generator, pm: float = 0.05) -> np.ndarray:
    sol = sol.copy()
    for i in range(sol.shape[0]):
        if rng.random() < pm:
            new_t = rng.integers(1, K + 1)
            while new_t == sol[i]:
                new_t = rng.integers(1, K + 1)
            sol[i] = new_t
    return sol


# ---------- GA loop ----------
def run_ga(
    inst: ExamInstance,
    pop_size: int = 100,
    generations: int = 500,
    pc: float = 0.8,
    pm: float = 0.05,
    tourn_k: int = 3,
    alpha: int = 10_000,
    seed: int = 42,
    use_repair: bool = True
) -> Tuple[np.ndarray, int, int, int, List[int]]:
    rng = np.random.default_rng(seed)

    pop = init_population(inst, pop_size, rng)
    if use_repair:
        pop = np.array([repair_solution(inst, ind, rng) for ind in pop], dtype=np.int32)

    fit = np.array([evaluate(inst, ind, alpha=alpha)[0] for ind in pop], dtype=np.int64)

    best_curve: List[int] = []

    best_idx = int(np.argmin(fit))
    best_sol = pop[best_idx].copy()
    best_F, best_H, best_S = evaluate(inst, best_sol, alpha=alpha)

    for g in range(generations):
        new_pop = []

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit, rng, k=tourn_k)
            p2 = tournament_select(pop, fit, rng, k=tourn_k)

            if rng.random() < pc:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate(c1, inst.K, rng, pm=pm)
            c2 = mutate(c2, inst.K, rng, pm=pm)

            if use_repair:
                c1 = repair_solution(inst, c1, rng)
                c2 = repair_solution(inst, c2, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=np.int32)
        fit = np.array([evaluate(inst, ind, alpha=alpha)[0] for ind in pop], dtype=np.int64)

        gen_best_idx = int(np.argmin(fit))
        gen_best_sol = pop[gen_best_idx]
        gen_best_F, gen_best_H, gen_best_S = evaluate(inst, gen_best_sol, alpha=alpha)

        # Track global best
        if gen_best_F < best_F:
            best_sol = gen_best_sol.copy()
            best_F, best_H, best_S = gen_best_F, gen_best_H, gen_best_S

        best_curve.append(best_F)

    return best_sol, best_F, best_H, best_S, best_curve

def run_multiple(
    inst: ExamInstance,
    runs: int = 10,
    base_seed: int = 1000,
    pop_size: int = 100,
    generations: int = 500,
    pc: float = 0.8,
    pm: float = 0.05,
    tourn_k: int = 3,
    alpha: int = 10_000,
    use_repair: bool = True,
    plot_best_run: bool = False,
) -> Dict[str, object]:
    results = []  # list of dicts
    best_overall = None

    for r in range(runs):
        seed = base_seed + r
        best_sol, best_F, best_H, best_S, curve = run_ga(
            inst,
            pop_size=pop_size,
            generations=generations,
            pc=pc,
            pm=pm,
            tourn_k=tourn_k,
            alpha=alpha,
            seed=seed,
            use_repair=use_repair
        )

        if best_H != 0:
            print(f"[WARN] Run {r+1}/{runs} seed={seed} produced infeasible solution: H={best_H}")

        rec = {"run": r + 1, "seed": seed, "F": best_F, "H": best_H, "S": best_S, "curve": curve}
        results.append(rec)

        if best_overall is None or best_F < best_overall["F"]:
            best_overall = rec | {"sol": best_sol}

        print(f"Run {r+1:02d}/{runs} seed={seed} -> F={best_F}, H={best_H}, S={best_S}")

    # Summaries (focus on S since with H=0, F==S)
    S_vals = np.array([x["S"] for x in results], dtype=np.float64)
    H_vals = np.array([x["H"] for x in results], dtype=np.int32)

    feasible_rate = float(np.mean(H_vals == 0)) * 100.0

    print("\n=== 10-run Summary ===")
    print(f"Feasible runs (H=0): {np.sum(H_vals==0)}/{runs} ({feasible_rate:.1f}%)")
    print(f"S (soft consecutive) best  : {int(S_vals.min())}")
    print(f"S (soft consecutive) worst : {int(S_vals.max())}")
    print(f"S (soft consecutive) mean  : {S_vals.mean():.2f}")
    print(f"S (soft consecutive) std   : {S_vals.std(ddof=1):.2f}")

    print("\nBest overall solution:")
    print("Timeslots:", best_overall["sol"].tolist())
    print(f"F={best_overall['F']}  H={best_overall['H']}  S={best_overall['S']}")

    if plot_best_run and best_overall is not None:
        plot_curve(best_overall["curve"], title="Best run fitness curve")

    return best_overall

def run_test_case1_baseline(inst: ExamInstance) -> None:
    print("\n" + "="*60)
    print("Test Case 1 - Baseline (pop=100, pm=0.05)")
    print("="*60)

    run_multiple(
        inst,
        runs=10,
        base_seed=4000,
        pop_size=100,
        generations=300,
        pc=0.8,
        pm=0.05,
        tourn_k=3,
        alpha=10_000,
        use_repair=True,
        plot_best_run=False
    )


def compare_settings(inst: ExamInstance) -> None:
    """Baseline runs for small instance (e.g. small-2)."""
    run_multiple(
        inst,
        runs=10,
        base_seed=1000,
        pop_size=100,
        generations=500,
        pc=0.8,
        pm=0.05,
        tourn_k=3,
        alpha=10_000,
        use_repair=True,
        plot_best_run=False
    )


def compare_settings_for_medium(inst: ExamInstance) -> None:
    settings = [
        {"name": "A_baseline", "pop_size": 100, "pm": 0.05, "generations": 300},
        {"name": "B_explore",  "pop_size": 150, "pm": 0.10, "generations": 300},
    ]

    for cfg in settings:
        print("\n" + "="*60)
        print(f"Setting: {cfg['name']}  pop={cfg['pop_size']}  pm={cfg['pm']}  gen={cfg['generations']}")
        print("="*60)

        run_multiple(
            inst,
            runs=10,
            base_seed=3000,
            pop_size=cfg["pop_size"],
            generations=cfg["generations"],
            pc=0.8,
            pm=cfg["pm"],
            tourn_k=3,
            alpha=10_000,
            use_repair=True,
            plot_best_run=False
        )


def plot_curve(best_curve: List[int], title: str = "Best Fitness per Generation") -> None:
    plt.figure()
    plt.plot(best_curve)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
