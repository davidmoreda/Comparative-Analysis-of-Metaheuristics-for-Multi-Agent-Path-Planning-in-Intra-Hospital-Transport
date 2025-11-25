import time
import random
import numpy as np
from deap import tools, algorithms

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import math 

from algorithms.ga_core import (
    prepare_environment,
    ga_setup,
    detect_conflicts,
    evaluate_clean_distance,   
)


# ==========================================================
# VISUALIZACIÓN DE RUTAS
# ==========================================================
def plot_routes(env, routes, starts, picks, drops, title=None):
    H, W = env.shape
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink"]

    plt.figure(figsize=(8, 8))
    plt.imshow(env, cmap="gray", origin="upper")

    # título configurable
    if title is None:
        plt.title("Optimized Multi-Agent Routes (GA)")
    else:
        plt.title(title)

    for k, r in enumerate(routes):
        ys = [p[0] for p in r]
        xs = [p[1] for p in r]
        c = colors[k % len(colors)]
        plt.plot(xs, ys, '-', color=c, lw=2, label=f"Agent {k+1}")

        sy, sx = starts[k]
        py, px = picks[k]
        dy, dx = drops[k]

        plt.plot(sx, sy, 'o', color=c, markersize=9, markerfacecolor='none')
        plt.plot(px, py, 's', color=c, markersize=9, markerfacecolor='none')
        plt.plot(dx, dy, 'X', color=c, markersize=9)

    plt.legend(loc="upper right")
    plt.grid(False)
    plt.show()



# ==========================================================
# CURVA DE CONVERGENCIA (SOLO DISTANCIA REAL)
# ==========================================================
def plot_convergence(clean_best, clean_avg, clean_std,title):
    clean_best = np.array(clean_best, dtype=float)
    clean_avg  = np.array(clean_avg, dtype=float)
    clean_std  = np.array(clean_std, dtype=float)

    gens = np.arange(1, len(clean_best) + 1)

    plt.figure(figsize=(10,5))
    plt.yscale("log")

    plt.plot(gens, clean_best, label="Best Distance", linewidth=2)
    plt.plot(gens, clean_avg,  label="Average Distance", linestyle='--')
    plt.fill_between(
        gens,
        clean_avg - clean_std,
        clean_avg + clean_std,
        alpha=0.25,
        label="±1 std"
    )

    plt.xlabel("Generation")
    plt.ylabel("Real Distance")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ==========================================================
# ANIMACIÓN TEMPORAL
# ==========================================================

def visualize_routes_timed(env, routes, starts, picks, drops,
                           safe_dist=6.0, show=True):
    colors = ['lime', 'red', 'cyan', 'yellow', 'magenta', 'orange']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(env, cmap="gray", origin="upper")
    ax.set_title("Spatiotemporal Evolution of Multi-Agent Routes")

    for i, c in enumerate(colors[:len(routes)]):
        sy, sx = starts[i]
        py, px = picks[i]
        dy, dx = drops[i]
        ax.plot(sx, sy, 'o', color=c, markersize=8, markerfacecolor='none')
        ax.plot(px, py, 's', color=c, markersize=8, markerfacecolor='none')
        ax.plot(dx, dy, 'X', color=c, markersize=8)

    plots = [ax.plot([], [], 'o', color=c, lw=2)[0] for c in colors[:len(routes)]]

    txt_frame = ax.text(2, 5, "", color='white', fontsize=11)
    txt_dist  = ax.text(2, 15, "", color='white', fontsize=11)
    txt_wait  = ax.text(2, 25, "", color='yellow', fontsize=11)
    txt_warn  = ax.text(2, 35, "", color='red', fontsize=13, fontweight="bold")

    n_frames = max(len(r) for r in routes)

    def update(frame):
        waits = []
        pos = []

        for i, r in enumerate(routes):
            idx = min(frame, len(r)-1)
            sub = np.array(r[:idx+1])
            plots[i].set_data(sub[:, 1], sub[:, 0])
            pos.append(r[idx])

            if frame > 0 and idx > 0 and r[idx] == r[idx-1]:
                waits.append(f"A{i+1}")

        dmin = float("inf")
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                y1, x1 = pos[i]
                y2, x2 = pos[j]
                d = math.hypot(y1-y2, x1-x2)
                dmin = min(dmin, d)

        txt_frame.set_text(f"Frame: {frame}")
        txt_dist.set_text(f"Min Distance: {dmin:.2f}")
        txt_wait.set_text("Waiting: " + ", ".join(waits) if waits else "")

        if dmin < safe_dist:
            txt_warn.set_text(f"⚠ TOO CLOSE (< {safe_dist:.1f})")
        else:
            txt_warn.set_text("")

        return plots + [txt_frame, txt_dist, txt_wait, txt_warn]

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=70,
        blit=False,
        repeat=True
    )

    if show:
        plt.show()

    return ani


# ==========================================================
# WRITER MP4
# ==========================================================
class FFMpegWriterWithProgress(FFMpegWriter):
    def __init__(self, total_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbar = tqdm(total=total_frames, desc="Renderizando vídeo", unit="frame")

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self.pbar.update(1)

    def finish(self):
        super().finish()
        self.pbar.close()


def run_ga(
    pop_size=100,
    ngen=200,
    cxpb=0.8,
    mutpb=0.2,
    seed=0,
    show_plots=True,
    show_anim=True,
    save_anim=False,
    anim_file="routes_animation.mp4",
    debug_interval=20,
    metric="clean"   
):
    random.seed(seed)
    np.random.seed(seed)

    env, starts, picks, drops = prepare_environment()
    tb, base_routes = ga_setup(env, starts, picks, drops, multi=False)

    pop = tb.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # evaluar población inicial
    fits = list(map(tb.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)

    # MÉTRICAS LIMPIAS
    clean_best = []
    clean_avg  = []
    clean_std  = []
    penalized_best = []
    penalized_avg  = []
    penalized_std  = []

    start_t = time.time()

    for gen in range(1, ngen + 1):

        offspring = algorithms.varAnd(pop, tb, cxpb=cxpb, mutpb=mutpb)
        fits = list(map(tb.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = tb.select(offspring, len(pop))
        hof.update(pop)

        # ==================================================
        # FITNESS LIMPIO
        # ==================================================
        clean_vals = [evaluate_clean_distance(ind) for ind in pop]
        best_clean = np.min(clean_vals)

        clean_best.append(best_clean)
        clean_avg.append(np.mean(clean_vals))
        clean_std.append(np.std(clean_vals))
        penal_vals = [ind.fitness.values[0] for ind in pop]
        penalized_best.append(np.min(penal_vals))
        penalized_avg.append(np.mean(penal_vals))
        penalized_std.append(np.std(penal_vals))
        # ==================================================
        # DEBUG: fitness limpio + fitness penalizado
        # ==================================================
        if gen % debug_interval == 0:
            best_pen = penalized_best[-1]
            avg_pen  = penalized_avg[-1]
            conflicts, mindist = detect_conflicts(hof[0])

            print(
                f"[GEN {gen:4d}] "
                f"PenalizedBest={best_pen:.1f} | "
                f"AvgPenalized={avg_pen:.1f} | "
                f"CleanBest={best_clean:.1f} | "
                f"AvgClean={clean_avg[-1]:.1f} | "
                f"Std={clean_std[-1]:.1f} | "
                f"MinDist={mindist:.2f} | "
                f"Conflicts={len(conflicts)}"
            )

    end_t = time.time()

    # ==================================================
    # FIN — selección de métrica para visualización
    # ==================================================
    best = hof[0]
    best_penalized = best.fitness.values[0]
    best_clean = evaluate_clean_distance(best)

    print("\n===== GA COMPLETADO =====")
    print(f"Mejor fitness penalizado = {best_penalized:.3f}")
    print(f"Mejor distancia real     = {best_clean:.3f}")
    print(f"Tiempo = {end_t - start_t:.2f} s")

    # === VISUALIZACIÓN SEGÚN MÉTRICA ===
    if show_plots:
        plot_routes(env, best, starts, picks, drops)

        if metric == "clean":
            plot_convergence(clean_best, clean_avg, clean_std,
                 title="Clean Fitness Convergence")

        elif metric == "penalized":
            plot_convergence(penalized_best, penalized_avg, penalized_std,
                 title="Penalized Fitness Convergence")
        else:
            raise ValueError("metric debe ser 'clean' o 'penalized'")
        
        if show_anim:
            print("\nShowing animation of best solution...")

            # Si vamos a guardar a disco, no hace falta mostrar interactivo
            ani = visualize_routes_timed(
                env, best, starts, picks, drops
            )

            if save_anim:
                # n_frames = longitud máxima de las rutas
                total_frames = max(len(r) for r in best)

                writer = FFMpegWriterWithProgress(
                    total_frames=total_frames,
                    fps=15
                )
                print(f"Saving animation to: {anim_file}")
                ani.save(anim_file, writer=writer)
                plt.close(ani._fig)


    return {
        "best_penalized": best_penalized,
        "best_clean": best_clean,
        "routes": best,
    }




if __name__ == "__main__":
    run_ga(
        pop_size=100,
        ngen=400,
        cxpb=0.7,
        mutpb=0.2,
        seed=42,
        show_plots=True,
        show_anim=True,
        save_anim=False,
        anim_file="routes_animation.mp4",
        debug_interval=50,
        metric="penalized"
    )
