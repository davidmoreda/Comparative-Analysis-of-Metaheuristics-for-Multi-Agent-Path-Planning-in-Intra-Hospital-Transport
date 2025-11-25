from ga_runner_multi import run_ga_multi
import numpy as np

def objective(trial):
    pop_size = trial.suggest_int("pop_size", 40, 140)
    ngen     = trial.suggest_int("ngen", 80, 250)
    cxpb     = trial.suggest_float("cxpb", 0.5, 0.9)
    mutpb    = trial.suggest_float("mutpb", 0.1, 0.5)

    seeds = [0, 1, 2]   # mínimo 3 semillas
    values = []

    for sd in seeds:
        out = run_ga_multi(
            pop_size=pop_size,
            ngen=ngen,
            cxpb=cxpb,
            mutpb=mutpb,
            seed=sd,
            show_plots=False,
            show_anim=False,
            save_anim=False,
            debug_interval=ngen-1
        )

        penal, clean = out["best_tradeoff"].fitness.values
        values.append(penal + clean)

    return float(np.mean(values))

import optuna

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Mejores hiperparámetros:", study.best_params)
print("Mejor valor objetivo:", study.best_value)
