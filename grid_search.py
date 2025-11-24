

import itertools
import multiprocessing as mp
import time
import csv
import numpy as np

from ga_runner_multi import run_ga_multi
from ga_core import detect_conflicts, MIN_SEP


# ============================================================
# 1) DEFINE AQUÍ EL GRID DE NSGA-II
# ============================================================

# ============================================================
# PARAM GRIDS DEFINITIVOS
# ============================================================

param_grid_ga = {
    "pop_size": [80, 120, 160],
    "ngen":     [150, 250],
    "cxpb":     [0.5, 0.7, 0.9],
    "mutpb":    [0.2, 0.3, 0.5],
    "show_plots": [False],
    "show_anim":  [False]
}

param_grid_sa = {
    "n_iter":     [8000, 10000, 12000, 15000],
    "start_temp": [5, 10, 20, 30],
    "end_temp":   [0.01],
    "show_plots": [False],
    "show_anim":  [False]
}

param_grid_mulambda = {
    "mu":     [80, 120, 160],
    "lambda_":[80, 120, 160],
    "ngen":   [500, 1000],
    "cxpb":   [0.4, 0.7],
    "mutpb":  [0.2, 0.4],
    "show_plots": [False],
    "show_anim":  [False]
}

param_grid_nsga = {
    "pop_size": [50,100,150, 200],
    "ngen":     [300,500,800],
    "cxpb":     [0.5,0.7,0.9],
    "mutpb":    [0.2,0.3,0.4],
    "show_plots": [False],
    "show_anim":  [False]
}



# ============================================================
# 2) GENERADOR DE TODAS LAS COMBINACIONES DEL GRID
# ============================================================

def generate_param_grid(param_dict):
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    for combo in itertools.product(*values):
        base_params = dict(zip(keys, combo))
        yield base_params


# ============================================================
# 3) UNA EJECUCIÓN DE NSGA-II (UNA CONFIG + UNA SEED)
# ============================================================

def run_single_nsga(args):
    """
    Ejecuta NSGA-II para una combinación concreta de hiperparámetros y una seed.

    Devuelve un dict con:
      - clean, penalized (fitness de best_tradeoff)
      - nº de conflictos y distancia mínima
      - factible o no
      - tiempo en segundos
    """
    base_params, seed = args

    params = dict(base_params)
    params["seed"] = seed

    t0 = time.time()
    out = run_ga_multi(**params)
    t1 = time.time()

    best_trade = out["best_tradeoff"]
    penal, clean = best_trade.fitness.values

    conflicts, mindist = detect_conflicts(best_trade)
    feasible = (len(conflicts) == 0 and mindist >= MIN_SEP)

    return {
        "seed": seed,
        "params": base_params,
        "clean": float(clean),
        "penalized": float(penal),
        "conflicts": int(len(conflicts)),
        "mindist": float(mindist),
        "feasible": bool(feasible),
        "time_sec": float(t1 - t0),
    }


# ============================================================
# 4) AGREGA LOS RESULTADOS DE VARIAS SEEDS PARA UNA CONFIG
# ============================================================

def aggregate_multiseed(results_one_config):
    """
    results_one_config: lista de dicts devueltos por run_single_nsga
                        (misma config, distintas seeds).

    Devuelve un dict con métricas agregadas por configuración.
    """
    clean_vals = np.array([r["clean"] for r in results_one_config], dtype=float)
    penal_vals = np.array([r["penalized"] for r in results_one_config], dtype=float)
    conf_vals  = np.array([r["conflicts"] for r in results_one_config], dtype=int)
    mind_vals  = np.array([r["mindist"] for r in results_one_config], dtype=float)
    feas_flags = np.array([r["feasible"] for r in results_one_config], dtype=bool)
    times      = np.array([r["time_sec"] for r in results_one_config], dtype=float)

    feasible_rate = feas_flags.mean()  # entre 0 y 1

    if feasible_rate > 0:
        penal_feas = float(penal_vals[feas_flags].mean())
        clean_feas = float(clean_vals[feas_flags].mean())
        conf_feas  = float(conf_vals[feas_flags].mean())
        mind_feas  = float(mind_vals[feas_flags].mean())
    else:
        penal_feas = float("inf")
        clean_feas = float("inf")
        conf_feas  = float(conf_vals.mean())
        mind_feas  = float(mind_vals.mean())

    return {
        "params": results_one_config[0]["params"],

        # Factibilidad
        "feasible_rate": float(feasible_rate),
        "conf_mean_all": float(conf_vals.mean()),
        "mind_mean_all": float(mind_vals.mean()),

        # Métricas solo en runs factibles
        "penal_mean_feas": penal_feas,
        "clean_mean_feas": clean_feas,
        "conf_mean_feas": conf_feas,
        "mind_mean_feas": mind_feas,

        # Métricas en todos los runs (útil para ver tendencia)
        "penal_mean_all": float(penal_vals.mean()),
        "clean_mean_all": float(clean_vals.mean()),

        # Tiempo
        "time_mean": float(times.mean()),
        "time_std": float(times.std()),
    }


# ============================================================
# 5) CLAVE DE ORDENACIÓN: QUÉ CONSIDERAMOS "MEJOR"
# ============================================================

def sort_key(result):
    """
    Orden de prioridad:

      1) Mayor tasa de factibilidad (feasible_rate)
      2) Menor penalizado medio en las seeds factibles
      3) Menor distancia limpia media en factibles
      4) Menor tiempo medio
    """
    return (
        -result["feasible_rate"],       # más alto mejor → signo menos
        result["penal_mean_feas"],
        result["clean_mean_feas"],
        result["time_mean"],
    )


# ============================================================
# 6) GRID SEARCH MULTISEED SOLO PARA NSGA-II
# ============================================================

def grid_search_nsga(param_grid, seeds, n_jobs=8, csv_path=None):
    """
    param_grid : dict con listas de valores (sin 'seed')
    seeds      : lista de semillas, p.ej. [0, 1, 2]
    n_jobs     : nº de procesos en paralelo
    csv_path   : si no es None, guarda resultados en CSV

    Devuelve: lista de dicts agregados y ordenados por calidad.
    """
    all_configs = list(generate_param_grid(param_grid))
    print(f"\n=== GRID SEARCH NSGA-II ===")
    print(f"Total configuraciones: {len(all_configs)}")
    print(f"Seeds por configuración: {len(seeds)}")
    print(f"Procesos en paralelo: {n_jobs}\n")

    # 1) Preparamos lista de tareas: una por (config, seed)
    tasks = []
    cfg_ids = []  # id de configuración para agrupar luego
    for cfg_id, base_params in enumerate(all_configs):
        for s in seeds:
            tasks.append((base_params, s))
            cfg_ids.append(cfg_id)

    t_global0 = time.time()

    # 2) Ejecutamos todas las runs en paralelo
    with mp.Pool(processes=n_jobs) as pool:
        all_results = pool.map(run_single_nsga, tasks)

    t_global1 = time.time()
    print(f"\n✔ Grid search completado en {t_global1 - t_global0:.2f} segundos (tiempo de pared).\n")

    # 3) Agrupamos por configuración
    grouped = {}
    for cfg_id, res in zip(cfg_ids, all_results):
        grouped.setdefault(cfg_id, []).append(res)

    agg_results = []
    for cfg_id, res_list in grouped.items():
        agg = aggregate_multiseed(res_list)
        agg_results.append(agg)

    # 4) Ordenamos según la clave de calidad
    agg_results_sorted = sorted(agg_results, key=sort_key)

    # 5) Guardado opcional a CSV
    if csv_path is not None:
        fieldnames = [
            "pop_size", "ngen", "cxpb", "mutpb",
            "feasible_rate",
            "conf_mean_all", "mind_mean_all",
            "penal_mean_feas", "clean_mean_feas",
            "conf_mean_feas", "mind_mean_feas",
            "penal_mean_all", "clean_mean_all",
            "time_mean", "time_std",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in agg_results_sorted:
                row = {
                    "feasible_rate": r["feasible_rate"],
                    "conf_mean_all": r["conf_mean_all"],
                    "mind_mean_all": r["mind_mean_all"],
                    "penal_mean_feas": r["penal_mean_feas"],
                    "clean_mean_feas": r["clean_mean_feas"],
                    "conf_mean_feas": r["conf_mean_feas"],
                    "mind_mean_feas": r["mind_mean_feas"],
                    "penal_mean_all": r["penal_mean_all"],
                    "clean_mean_all": r["clean_mean_all"],
                    "time_mean": r["time_mean"],
                    "time_std": r["time_std"],
                }
                # Añadimos los hiperparámetros básicos
                params = r["params"]
                row["pop_size"] = params["pop_size"]
                row["ngen"]     = params["ngen"]
                row["cxpb"]     = params["cxpb"]
                row["mutpb"]    = params["mutpb"]

                writer.writerow(row)

        print(f"✔ Resultados guardados en: {csv_path}")

    return agg_results_sorted


# ============================================================
# 7) MAIN: EJEMPLO DE EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    # Ajusta aquí tus seeds y nº de procesos
    seeds = [0, 1, 2]
    n_jobs = 8

    results = grid_search_nsga(
        param_grid_nsga,
        seeds=seeds,
        n_jobs=n_jobs,
        csv_path="nsga_grid_results.csv"
    )

    # Mostramos el TOP 5 por consola
    print("\n===== TOP 5 CONFIGURACIONES (NSGA-II, MULTISEED, FACTIBILIDAD-AWARE) =====")
    for r in results[:5]:
        p = r["params"]
        print("\nParams:",
              f"pop_size={p['pop_size']}, ngen={p['ngen']}, "
              f"cxpb={p['cxpb']:.3f}, mutpb={p['mutpb']:.3f}")
        print(f"  feasible_rate   = {r['feasible_rate']:.2f}")
        print(f"  penal_mean_feas = {r['penal_mean_feas']:.2f}")
        print(f"  clean_mean_feas = {r['clean_mean_feas']:.2f}")
        print(f"  penal_mean_all  = {r['penal_mean_all']:.2f}")
        print(f"  clean_mean_all  = {r['clean_mean_all']:.2f}")
        print(f"  conf_mean_all   = {r['conf_mean_all']:.2f}")
        print(f"  mind_mean_all   = {r['mind_mean_all']:.2f}")
        print(f"  time_mean       = {r['time_mean']:.2f}s ± {r['time_std']:.2f}")
