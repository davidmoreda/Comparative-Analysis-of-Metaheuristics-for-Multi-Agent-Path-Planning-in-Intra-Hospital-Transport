import itertools
import multiprocessing as mp
import sqlite3
import numpy as np
import time
from pymoo.indicators.hv import HV
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
# ===========================================

from algorithms.ga_runner_multi import run_ga_multi
from algorithms.ga_core import detect_conflicts, MIN_SEP


# ============================================================
# DB UTILITIES
# ============================================================

def init_db(path):
    """Crea tablas si no existen."""
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pop_size INTEGER,
            ngen INTEGER,
            cxpb REAL,
            mutpb REAL,
            seed INTEGER,
            hv REAL,
            clean REAL,
            penalized REAL,
            conflicts INTEGER,
            mindist REAL,
            feasible INTEGER,
            time_sec REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pareto_fronts (
            run_id INTEGER,
            f1 REAL,
            f2 REAL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
    """)

    con.commit()
    con.close()


def save_run_to_db(db_path, run_data, pareto):
    """Inserta una ejecución individual + su Pareto en la DB."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        INSERT INTO runs
        (pop_size, ngen, cxpb, mutpb, seed, hv,
         clean, penalized, conflicts, mindist, feasible, time_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_data["params"]["pop_size"],
        run_data["params"]["ngen"],
        run_data["params"]["cxpb"],
        run_data["params"]["mutpb"],
        run_data["seed"],
        run_data["hv"],
        run_data["clean"],
        run_data["penalized"],
        run_data["conflicts"],
        run_data["mindist"],
        int(run_data["feasible"]),
        run_data["time_sec"]
    ))

    run_id = cur.lastrowid

    # Pareto front
    cur.executemany(
        "INSERT INTO pareto_fronts (run_id, f1, f2) VALUES (?, ?, ?)",
        [(run_id, float(f[0]), float(f[1])) for f in pareto]
    )

    con.commit()
    con.close()


# ============================================================
# NSGA-II RUN → HV + Pareto
# ============================================================

def run_single_nsga(args):
    base_params, seed, ref_point, db_path = args

    params = dict(base_params)
    params["seed"] = seed

    t0 = time.time()
    out = run_ga_multi(**params)
    t1 = time.time()

    best = out["best_tradeoff"]
    penal, clean = best.fitness.values
    conflicts, mindist = detect_conflicts(best)
    feasible = (len(conflicts) == 0 and mindist >= MIN_SEP)

    # Pareto array
    pareto = np.array([ind.fitness.values for ind in out["pareto_front"]])

    # Hypervolume
    hv = HV(ref_point=ref_point)(pareto) if len(pareto) > 0 else 0.0

    run_data = {
        "seed": seed,
        "params": base_params,
        "hv": float(hv),
        "clean": float(clean),
        "penalized": float(penal),
        "conflicts": int(len(conflicts)),
        "mindist": float(mindist),
        "feasible": feasible,
        "time_sec": float(t1 - t0),
    }

    # Guardamos directamente en DB
    save_run_to_db(db_path, run_data, pareto)

    return run_data


# ============================================================
# GRID SEARCH (MULTIPROCESS)
# ============================================================

def generate_param_grid(param_dict):
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def aggregate_config(results):
    """Agrega resultados de una config sobre todas las seeds."""
    hv = np.array([r["hv"] for r in results])
    time_vals = np.array([r["time_sec"] for r in results])

    return {
        "params": results[0]["params"],
        "hv_mean": float(hv.mean()),
        "hv_std": float(hv.std()),
        "time_mean": float(time_vals.mean()),
        "time_std": float(time_vals.std()),
    }


def grid_search_nsga2(param_grid, seeds, ref_point, n_jobs, db_path):
    init_db(db_path)

    all_configs = list(generate_param_grid(param_grid))

    # Preparar tareas
    tasks = []
    for cfg in all_configs:
        for s in seeds:
            tasks.append((cfg, s, ref_point, db_path))

    # Exec parallel
    with mp.Pool(processes=n_jobs) as pool:
        all_results = pool.map(run_single_nsga, tasks)

    # Agrupar por configuración
    cfg_map = {}
    for cfg, seed, _, _ in tasks:
        cfg_key = tuple(cfg.items())
        cfg_map.setdefault(cfg_key, [])

    for res in all_results:
        cfg_key = tuple(res["params"].items())
        cfg_map[cfg_key].append(res)

    # Agregación por configuración
    aggregated = [aggregate_config(v) for v in cfg_map.values()]

    # Ordenar por HV descendente
    aggregated_sorted = sorted(aggregated, key=lambda x: -x["hv_mean"])
    return aggregated_sorted


# ============================================================
# MAIN (EJEMPLO)
# ============================================================

if __name__ == "__main__":
    param_grid_nsga = {
        "pop_size": [50, 80, 100, 120],
        "ngen":     [300, 500,800,1000],
        "cxpb":     [0.6,0.7,0.8],
        "mutpb":    [0.4,0.3,0.2],
        "show_plots": [False],
        "show_anim":  [False],
    }

    seeds = [0, 1, 2]
    ref_point = np.array([5000, 5000])
    db_path = "nsga2_grid.db"

    best_configs = grid_search_nsga2(
        param_grid_nsga,
        seeds=seeds,
        ref_point=ref_point,
        n_jobs=8,
        db_path=db_path
    )

    # No prints — solo retornamos
