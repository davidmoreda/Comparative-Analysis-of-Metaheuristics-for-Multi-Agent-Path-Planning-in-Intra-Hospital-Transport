import time
import random
import numpy as np
from deap import tools, algorithms

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

from ga_core import (
    prepare_environment,
    ga_setup,
    detect_conflicts
)


# ==========================================================
# CURVAS DE CONVERGENCIA MULTIOBJETIVO
# ==========================================================
def plot_multi_convergence(logbook):
    gens = logbook.select("gen")
    mins = logbook.select("min")  # lista de tuplas (min_pen, min_clean)
    avgs = logbook.select("avg")  # lista de tuplas (avg_pen, avg_clean)

    min_pen  = [m[0] for m in mins]
    min_clean = [m[1] for m in mins]
    avg_pen  = [a[0] for a in avgs]
    avg_clean = [a[1] for a in avgs]

    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    # Penalized
    axs[0].plot(gens, min_pen, label="Best Penalized", lw=2)
    axs[0].plot(gens, avg_pen, label="Avg Penalized", ls='--')
    axs[0].set_title("Penalized Objective Convergence")
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Penalized Fitness")
    axs[0].grid(True)
    axs[0].legend()

    # Clean distance
    axs[1].plot(gens, min_clean, label="Best Clean", lw=2)
    axs[1].plot(gens, avg_clean, label="Avg Clean", ls='--')
    axs[1].set_title("Clean Distance Convergence")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Clean Distance")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# PARETO FRONT
# ==========================================================
def plot_pareto_front(hof):
    xs = [ind.fitness.values[0] for ind in hof]  # penalized
    ys = [ind.fitness.values[1] for ind in hof]  # clean distance

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='red', s=40)
    plt.xlabel("Penalized Fitness")
    plt.ylabel("Clean Distance")
    plt.title("Pareto Front (NSGA-II)")
    plt.grid(True)
    plt.show()

# ==========================================================
# PLOT THREE ROUTES
# ==========================================================
def plot_routes_three(env, starts, picks, drops,
                      inds, titles):
    """
    Dibuja 3 soluciones (individuos) en una sola figura:
    inds = [ind1, ind2, ind3]
    titles = [t1, t2, t3]
    """
    H, W = env.shape
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ind, title in zip(axes, inds, titles):
        ax.imshow(env, cmap="gray", origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        for k, r in enumerate(ind):
            ys = [p[0] for p in r]
            xs = [p[1] for p in r]
            c = colors[k % len(colors)]
            ax.plot(xs, ys, '-', color=c, lw=2, label=f"Agent {k+1}")

            sy, sx = starts[k]
            py, px = picks[k]
            dy, dx = drops[k]

            ax.plot(sx, sy, 'o', color=c, markersize=6, markerfacecolor='none')
            ax.plot(px, py, 's', color=c, markersize=6, markerfacecolor='none')
            ax.plot(dx, dy, 'X', color=c, markersize=6)

        ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.show()

# ==========================================================
# IMPORTAMOS TUS FUNCIONES DEL RUNNER NORMAL
# ==========================================================
from ga_runner import visualize_routes_timed, FFMpegWriterWithProgress


# ==========================================================
# NSGA-II RUNNER
# ==========================================================
def run_ga_multi(
    pop_size=100,
    ngen=80,
    cxpb=0.7,
    mutpb=0.3,
    seed=42,
    show_plots=True,
    show_anim=True,
    save_anim = False,
    anim_file="routes_animation.mp4",
    debug_interval=10,      
):
    random.seed(seed)
    np.random.seed(seed)

    env, starts, picks, drops = prepare_environment()

    # === MULTI = True → usa evaluate_multi ===
    tb, base_routes = ga_setup(env, starts, picks, drops, multi=True)

    print("\n=== Running NSGA-II Multiobjective GA ===")

    # ------------------------------------------------------
    # INICIALIZACIÓN NSGA-II
    # ------------------------------------------------------
    pop = tb.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen"] + stats.fields

    # Evaluación inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(tb.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Primera ordenación NSGA-II
    pop = tools.selNSGA2(pop, pop_size)
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(logbook.stream)

    # ------------------------------------------------------
    # BUCLE PRINCIPAL NSGA-II
    # ------------------------------------------------------
    for gen in range(1, ngen + 1):
        # 1) Variación (crossover + mutación)
        offspring = algorithms.varAnd(pop, tb, cxpb=cxpb, mutpb=mutpb)

        # 2) Evaluar solo los inválidos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(tb.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 3) Combinar población actual + descendencia
        combined = pop + offspring

        # 4) Selección NSGA-II (no dominated sorting + crowding distance)
        pop = tools.selNSGA2(combined, pop_size)

        # 5) Actualizar Pareto front y estadísticas
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        #print(logbook.stream)

        # --------------------------------------------------
        # DEBUG CADA X GENERACIONES (como en ga_runner)
        # --------------------------------------------------
        if gen % debug_interval == 0:
            # Penalizado y limpio de toda la población
            penal_vals = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
            clean_vals = np.array([ind.fitness.values[1] for ind in pop], dtype=float)

            best_pen = float(np.min(penal_vals))
            avg_pen  = float(np.mean(penal_vals))

            best_clean = float(np.min(clean_vals))
            avg_clean  = float(np.mean(clean_vals))
            std_clean  = float(np.std(clean_vals))

            # Conflictos y distancia mínima del mejor penalizado
            best_pen_ind = min(pop, key=lambda ind: ind.fitness.values[0])
            conflicts, mindist = detect_conflicts(best_pen_ind)

            print(
                f"[GEN {gen:4d}] "
                f"PenalBest={best_pen:.1f} | "
                f"PenalAvg={avg_pen:.1f} | "
                f"CleanBest={best_clean:.1f} | "
                f"CleanAvg={avg_clean:.1f} | "
                f"StdClean={std_clean:.1f} | "
                f"MinDist={mindist:.2f} | "
                f"Conflicts={len(conflicts)}"
            )

    print("\n=== NSGA-II COMPLETED ===")
    print(f"Solutions in Pareto Front: {len(hof)}")

    # Extract best solutions
    best_pen = min(hof, key=lambda ind: ind.fitness.values[0])
    best_clean = min(hof, key=lambda ind: ind.fitness.values[1])
    best_trade = min(hof, key=lambda ind: ind.fitness.values[0] + ind.fitness.values[1])

    print("\nBest Penalized:", best_pen.fitness.values)
    print("Best Clean:", best_clean.fitness.values)
    print("Best Tradeoff:", best_trade.fitness.values)

    # --- Conflictos de las soluciones finales ---
    for name, ind in [
        ("Best Penalized", best_pen),
        ("Best Clean", best_clean),
        ("Best Tradeoff", best_trade),
    ]:
        conflicts, mindist = detect_conflicts(ind)
        print(
            f"{name} -> Conflicts={len(conflicts)}, "
            f"MinDist={mindist:.2f}"
        )

    # ======================================================
    # VISUALIZACIONES
    # ======================================================
    if show_plots:
        # Pareto front
        plot_pareto_front(hof)

        # Convergencia multiobjetivo
        plot_multi_convergence(logbook)
        """
        # Rutas
        plot_routes(env, best_pen,   starts, picks, drops,
                    title=f"Best Penalized {best_pen.fitness.values}")
        plot_routes(env, best_clean, starts, picks, drops,
                    title=f"Best Clean Distance {best_clean.fitness.values}")
        plot_routes(env, best_trade, starts, picks, drops,
                    title=f"Best Tradeoff {best_trade.fitness.values}")
        """
            # --- una figura con las tres soluciones ---
        plot_routes_three(
            env, starts, picks, drops,
            inds=[best_pen, best_clean, best_trade],
            titles=[
                f"Best Penalized\n{best_pen.fitness.values}",
                f"Best Clean\n{best_clean.fitness.values}",
                f"Best Tradeoff\n{best_trade.fitness.values}",
            ]
        )

    # ======================================================
    # ANIMATION → SOLO BEST TRADEOFF
    # ======================================================
    if show_anim:
        print("\nShowing animation of best tradeoff...")

        # Si vamos a guardar a disco, no hace falta mostrar interactivo
        ani = visualize_routes_timed(
            env, best_trade, starts, picks, drops
        )

        if save_anim:
            # n_frames = longitud máxima de las rutas
            total_frames = max(len(r) for r in best_trade)

            writer = FFMpegWriterWithProgress(
                total_frames=total_frames,
                fps=15
            )
            print(f"Saving animation to: {anim_file}")
            ani.save(anim_file, writer=writer)
            plt.close(ani._fig)

    return {
        "pareto_front": hof,
        "best_penalized": best_pen,
        "best_clean": best_clean,
        "best_tradeoff": best_trade,
        "final_population": pop,
        "logbook": logbook
    }




if __name__ == "__main__":
    run_ga_multi(
        pop_size=57,
        ngen=248, #248 optuna
        cxpb=0.5475809615019229,
        mutpb=0.20546606617312818,
        seed=0,
        show_plots=True,
        show_anim=True,
        save_anim=False,
        anim_file="routes_animation_ga_multi_57_1000_0547_0205_s0.mp4",
        debug_interval=50
    )
