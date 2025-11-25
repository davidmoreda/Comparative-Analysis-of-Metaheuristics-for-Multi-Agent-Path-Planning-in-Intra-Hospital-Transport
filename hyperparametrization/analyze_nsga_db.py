import sqlite3
import numpy as np
import matplotlib.pyplot as plt


DB_PATH = "nsga2_grid.db"


# ============================================================
# 1) LEER TODAS LAS CONFIGURACIONES (AGREGADO POR HV)
# ============================================================

def get_best_configs(limit=10):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT pop_size, ngen, cxpb, mutpb,
               AVG(hv) AS hv_mean,
               AVG(time_sec) AS time_mean,
               COUNT(*) AS runs_count
        FROM runs
        GROUP BY pop_size, ngen, cxpb, mutpb
        ORDER BY hv_mean DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    con.close()
    return rows


# ============================================================
# 2) OBTENER EL MEJOR RUN INDIVIDUAL
# ============================================================

def get_best_run():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT id, pop_size, ngen, cxpb, mutpb, seed,
               hv, clean, penalized, conflicts, mindist, feasible, time_sec
        FROM runs
        ORDER BY hv DESC
        LIMIT 1
    """)

    row = cur.fetchone()
    con.close()
    return row


# ============================================================
# 3) OBTENER EL FRENTE DE PARETO DE UN RUN
# ============================================================

def get_pareto(run_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT f1, f2 FROM pareto_fronts WHERE run_id = ?
    """, (run_id,))

    pts = np.array(cur.fetchall())
    con.close()
    return pts


# ============================================================
# 4) PLOTEAR UN FRENTE DE PARETO
# ============================================================

def plot_pareto(run_id):
    pareto = get_pareto(run_id)

    plt.figure(figsize=(7, 6))
    plt.scatter(pareto[:, 0], pareto[:, 1], s=30, alpha=0.8)
    plt.xlabel("Penalized (minimizar)")
    plt.ylabel("Clean (minimizar)")
    plt.title(f"Pareto front (run_id={run_id})")
    plt.grid(True)
    plt.show()


# ============================================================
# 5) MAIN DE ANÁLISIS
# ============================================================

if __name__ == "__main__":

    print("\n===== TOP CONFIGURACIONES POR HV =====")
    configs = get_best_configs(limit=5)
    for c in configs:
        pop, ngen, cxpb, mutpb, hv_mean, time_mean, count = c
        print(f"pop={pop}, ngen={ngen}, cxpb={cxpb}, mutpb={mutpb}")
        print(f"  HV medio = {hv_mean:.2f}")
        print(f"  Tiempo medio = {time_mean:.2f}s")
        print(f"  Runs = {count}\n")

    print("\n===== MEJOR RUN INDIVIDUAL =====")
    best = get_best_run()
    run_id, pop, ngen, cxpb, mutpb, seed, hv, clean, penal, conf, mind, feas, t = best
    print(f"run_id={run_id}")
    print(f"pop={pop}, ngen={ngen}, cxpb={cxpb}, mutpb={mutpb}, seed={seed}")
    print(f"HV={hv:.2f}, clean={clean:.2f}, penal={penal:.2f}")
    print(f"conflicts={conf}, mindist={mind:.2f}, feasible={feas}")
    print(f"time={t:.2f}s")

    print("\nMostrando Pareto…")
    plot_pareto(run_id)
