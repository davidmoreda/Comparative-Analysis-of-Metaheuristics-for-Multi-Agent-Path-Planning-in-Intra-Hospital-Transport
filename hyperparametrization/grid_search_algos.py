import itertools
import multiprocessing as mp
import sqlite3
import json
import time
import numpy as np
import os
import sys

# ===========================================
# AÑADIR CARPETA RAÍZ AL PATH
# ===========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# ===========================================
# IMPORTAR ALGORITMOS REALES
# ===========================================
from algorithms.ga_runner import run_ga
from algorithms.mulambda_runner import run_mulambda
from algorithms.sa_runner import run_sa

# ===========================================
# DATABASE
# ===========================================
def init_db(path):
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algo TEXT,
            seed INTEGER,
            params TEXT,
            score REAL,
            time_sec REAL
        );
    """)

    con.commit()
    con.close()


def save_run(db_path, algo, seed, params, score, time_sec):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        INSERT INTO runs (algo, seed, params, score, time_sec)
        VALUES (?, ?, ?, ?, ?)
    """, (
        algo,
        seed,
        json.dumps(params),
        float(score),
        float(time_sec)
    ))

    con.commit()
    con.close()

# ===========================================
# WRAPPERS CORRECTOS (NO COLISIONAN)
# ===========================================

def run_ga_wrapper(params, seed):
    p = dict(params)
    p["seed"] = seed
    out = run_ga(**p)
    return float(out["best_penalized"])   # GA devuelve best_penalized


def run_mulambda_wrapper(params, seed):
    p = dict(params)
    p["seed"] = seed
    out = run_mulambda(**p)
    return float(out["best_penalized"])   # μ+λ devuelve best_penalized


def run_sa_wrapper(params, seed):
    p = dict(params)
    p["seed"] = seed
    out = run_sa(**p)
    return float(out["best_penalized"])   # SA devuelve best_penalized


RUNNERS = {
    "ga": run_ga_wrapper,
    "mulambda": run_mulambda_wrapper,
    "sa": run_sa_wrapper
}

# ===========================================
# GRID GENERATOR CON AUTO-MUTPB
# ===========================================

def expand_grid(grid_dict, algo_name):
    is_mulambda = (algo_name == "mulambda")

    keys = list(grid_dict.keys())
    values = list(grid_dict.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        if is_mulambda:
            cx = params["cxpb"]
            params["mutpb"] = round(1.0 - cx, 6)

        yield params

# ===========================================
# WORKER
# ===========================================

def _worker(args):
    algo, params, seed, db_path = args
    runner = RUNNERS[algo]

    t0 = time.time()
    score = runner(params, seed)
    t1 = time.time()

    save_run(db_path, algo, seed, params, score, t1 - t0)

    return score

# ===========================================
# GRID SEARCH
# ===========================================

def grid_search(algo, param_grid, seeds, db_path, n_jobs=8):

    init_db(db_path)

    all_params = list(expand_grid(param_grid, algo))

    tasks = []
    for p in all_params:
        for s in seeds:
            tasks.append((algo, p, s, db_path))

    with mp.Pool(processes=n_jobs) as pool:
        pool.map(_worker, tasks)

    return analyze_results(db_path, algo)

# ===========================================
# ANALYSIS
# ===========================================

def analyze_results(db_path, algo):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        SELECT params, score, time_sec
        FROM runs
        WHERE algo = ?
    """, (algo,))

    rows = cur.fetchall()
    con.close()

    groups = {}
    for p_json, score, t in rows:
        groups.setdefault(p_json, []).append((score, t))

    results = []
    for p_json, vals in groups.items():
        scores = np.array([v[0] for v in vals])
        times  = np.array([v[1] for v in vals])

        results.append({
            "params": json.loads(p_json),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "time_mean": float(times.mean()),
            "time_std": float(times.std())
        })

    return sorted(results, key=lambda x: (x["score_mean"], x["time_mean"]))

# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":

    grid_ga = {
        "pop_size": [80, 120, 150],
        "ngen": [300, 600, 800],
        "cxpb": [0.5, 0.7, 0.8],
        "mutpb": [0.1, 0.3, 0.5]
    }

    grid_mulambda = {
        "mu": [50, 80, 120],
        "lambda_": [50, 80, 120],
        "ngen": [300, 600, 800],
        "cxpb": [0.5, 0.7, 0.8]
    }

    grid_sa = {
        "n_iter": [5000, 8000, 12000],
        "start_temp": [5, 10, 20],
        "end_temp": [0.01]
    }

    DB_PATH = "grid_results.db"
    SEEDS = [0, 1, 2]
    N_JOBS = 8

    algo = "mulambda"       # "ga", "mulambda", "sa"
    param_grid = grid_mulambda

    results = grid_search(algo, param_grid, SEEDS, DB_PATH, N_JOBS)

    print("\nTop 3 mejores configuraciones:")
    for r in results[:3]:
        print("\nParams:", r["params"])
        print("Score:", r["score_mean"])
        print("Time:", r["time_mean"])
