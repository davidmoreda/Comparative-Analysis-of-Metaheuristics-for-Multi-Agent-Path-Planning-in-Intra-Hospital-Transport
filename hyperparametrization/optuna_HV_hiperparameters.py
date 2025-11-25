import numpy as np
import optuna
from pymoo.indicators.hv import HV
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

from ga_runner_multi import run_ga_multi


# ================================================
# CONFIGURACIÓN GLOBAL
# ================================================
REF_POINT = np.array([5000, 5000])
SEEDS = [0, 1, 2]

N_JOBS = 8              # nº de trials paralelos
SEED_JOBS = len(SEEDS)  # nº de procesos por trial
STORAGE = "sqlite:///optuna_ga.db"


# ================================================
# RUNNER DE UNA SEED (PARALELO)
# ================================================
def run_single_seed(args):
    pop_size, ngen, cxpb, mutpb, seed = args

    try:
        out = run_ga_multi(
            pop_size=pop_size,
            ngen=ngen,
            cxpb=cxpb,
            mutpb=mutpb,
            seed=seed,
            show_plots=False,
            show_anim=False,
            save_anim=False,
            debug_interval=ngen-2
        )

        front = np.array([ind.fitness.values for ind in out["pareto_front"]])
        return front

    except Exception as e:
        print(f"[ERROR Seed {seed}] {e}")
        traceback.print_exc()
        # devolvemos un front muy malo para no romper Optuna
        return np.array([[999999, 999999]])


# ================================================
# FUNCIÓN OBJETIVO (UN TRIAL)
# ================================================
def objective(trial):

    pop_size = trial.suggest_int("pop_size", 50, 200)
    ngen     = trial.suggest_int("ngen", 200, 1000)
    cxpb     = trial.suggest_float("cxpb", 0.5, 0.9)
    mutpb    = trial.suggest_float("mutpb", 0.1, 0.5)

    args_list = [(pop_size, ngen, cxpb, mutpb, sd) for sd in SEEDS]

    # Semillas en paralelo
    fronts = []
    with ProcessPoolExecutor(max_workers=SEED_JOBS) as executor:
        future_map = {executor.submit(run_single_seed, args): args for args in args_list}

        for fut in as_completed(future_map):
            res = fut.result()
            fronts.append(res)

    # Hypervolume
    hv_metric = HV(ref_point=REF_POINT)
    hvs = [hv_metric(front) for front in fronts]

    return -float(np.mean(hvs))   # Optuna minimiza


# ================================================
# EJECUCIÓN DEL ESTUDIO PRO
# ================================================
def run_optuna_pro(n_trials=100):

    sampler = optuna.samplers.TPESampler(multivariate=True)

    study = optuna.create_study(
        study_name="GA_HV_OPT",
        direction="minimize",
        storage=STORAGE,
        load_if_exists=True,
        sampler=sampler
    )

    # N_JOBS = nº de trials en paralelo
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=N_JOBS,
        catch=(Exception,)
    )

    print("\n==============================")
    print("        RESULTADOS")
    print("==============================")
    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor valor (neg-HV):", study.best_value)
    print("Hypervolume estimado:", -study.best_value)


# ================================================
# MAIN OBLIGATORIO
# ================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # estable y multiproceso
    print("=== OPTUNA HV MULTI-PRO ===")
    run_optuna_pro(n_trials=50)
