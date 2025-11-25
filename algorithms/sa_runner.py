import time
import random
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from algorithms.ga_core import (
    prepare_environment,
    create_graph,
    build_route,
    mutate_macro_detour,
    mutate_segment,
    mutate_long_wait,
    mutate_wait,
    mutate_shift_start,
    mutate_conflict,
    detect_conflicts,
    evaluate,               # fitness penalizado
    evaluate_clean_distance # métrica limpia: sólo distancia real
)


# ==========================================================
# VISUALIZACIÓN DE RUTAS
# ==========================================================
def plot_routes_sa(env, routes, starts, picks, drops):
    H, W = env.shape
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink"]

    plt.figure(figsize=(8, 8))
    plt.imshow(env, cmap="gray", origin="upper")
    plt.title("Optimized Multi-Agent Routes (SA)")

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
#  - clean_best: mejor distancia real encontrada hasta cada iter
#  - clean_curr: distancia real de la solución actual en cada iter
# ==========================================================
def plot_convergence_sa(clean_best, clean_curr, clean_std=None):
    clean_best = np.array(clean_best, dtype=float)
    clean_curr = np.array(clean_curr, dtype=float)

    if clean_std is None:
        clean_std = np.zeros_like(clean_best)
    clean_std = np.array(clean_std, dtype=float)

    iters = np.arange(1, len(clean_best) + 1)

    plt.figure(figsize=(10, 5))
    plt.yscale("log")

    plt.plot(iters, clean_best, label="Best Distance (so far)", linewidth=2)
    plt.plot(iters, clean_curr, label="Current Distance", linestyle='--')

    # Relleno con ± std (aquí es opcional, puede ser todo 0)
    plt.fill_between(
        iters,
        clean_curr - clean_std,
        clean_curr + clean_std,
        alpha=0.25,
        label="±1 std (if used)"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Real Distance")
    plt.title("SA Convergence (Real Distance Only)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ==========================================================
# ANIMACIÓN TEMPORAL
# ==========================================================
def visualize_routes_timed_sa(env, routes, starts, picks, drops, safe_dist=6.0):
    colors = ['lime', 'red', 'cyan', 'yellow', 'magenta', 'orange']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(env, cmap="gray", origin="upper")
    ax.set_title("Spatiotemporal Evolution of Multi-Agent Routes (SA)")

    for i, c in enumerate(colors[:len(routes)]):
        sy, sx = starts[i]
        py, px = picks[i]
        dy, dx = drops[i]
        ax.plot(sx, sy, 'o', color=c, markersize=8, markerfacecolor='none')
        ax.plot(px, py, 's', color=c, markersize=8, markerfacecolor='none')
        ax.plot(dx, dy, 'X', color=c, markersize=8)

    plots = [ax.plot([], [], 'o', color=c, lw=2)[0]
             for c in colors[:len(routes)]]

    txt_frame = ax.text(2, 5, "", color='white', fontsize=11)
    txt_dist  = ax.text(2, 15, "", color='white', fontsize=11)
    txt_wait  = ax.text(2, 25, "", color='yellow', fontsize=11)
    txt_warn  = ax.text(2, 35, "", color='red', fontsize=13, fontweight="bold")

    import math as _math
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
                d = _math.hypot(y1-y2, x1-x2)
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

    plt.show()
    return ani


# ==========================================================
# WRITER MP4
# ==========================================================
class FFMpegWriterWithProgress(FFMpegWriter):
    def __init__(self, total_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbar = tqdm(total=total_frames, desc="Renderizando vídeo (SA)",
                         unit="frame")

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self.pbar.update(1)

    def finish(self):
        super().finish()
        self.pbar.close()


# ==========================================================
# OPERADOR DE VECINDAD PARA SA
#   - Reutiliza las mismas mutaciones que el GA
# ==========================================================
def sa_neighbor(ind, env, G):
    """
    Genera un vecino de 'ind' aplicando mutaciones similares al GA.
    Se trabaja sobre una copia profunda de las rutas.
    """
    new_ind = [r[:] for r in ind]  # copia poco profunda por ruta, suficiente

    conflicts, _ = detect_conflicts(new_ind)

    for k in range(len(new_ind)):
        r = new_ind[k]

        if random.random() < 0.25:
            r = mutate_macro_detour(r, G, env)

        if random.random() < 0.30:
            r = mutate_segment(r, G)

        if random.random() < 0.30:
            r = mutate_long_wait(r, G)

        if random.random() < 0.30:
            r = mutate_wait(r, G)

        if random.random() < 0.25:
            r = mutate_shift_start(r, G)

        if conflicts and random.random() < 0.40:
            r = mutate_conflict(r, G, conflicts)

        new_ind[k] = r

    return new_ind


# ==========================================================
#      EJECUCIÓN DEL SIMULATED ANNEALING COMPLETO
# ==========================================================
def run_sa(
    n_iter=5000,
    start_temp=10.0,
    end_temp=0.1,
    seed=42,
    show_plots=False,
    show_anim=False,
    save_anim=False,
    anim_file="routes_animation_sa.mp4",
    debug_interval=500
):
    """
    Optimización de las rutas multi-agente usando Simulated Annealing.

    Devuelve un diccionario con:
      - best_penalized: mejor fitness penalizado
      - best_distance: mejor distancia real (métrica limpia)
      - routes: rutas del mejor individuo
      - clean_best: historial de mejor distancia limpia
      - clean_avg: aquí se usa como 'distancia de la solución actual'
      - time_sec: tiempo total de cómputo
    """

    random.seed(seed)
    np.random.seed(seed)

    # ======= Preparar entorno y rutas base =================
    env, starts, picks, drops = prepare_environment()
    G = create_graph(env)

    base_routes = []
    for k in range(len(starts)):
        r = build_route(G, starts[k], picks[k], drops[k])
        if r is None:
            raise RuntimeError("No base route for agent", k)
        base_routes.append(r)

    # Solución inicial = rutas base
    current = [br[:] for br in base_routes]

    def penalized_cost(ind):
        # evaluate devuelve una tupla (coste,)
        return evaluate(ind, picks, drops, base_routes)[0]

    curr_cost = penalized_cost(current)

    best = [r[:] for r in current]
    best_cost = curr_cost
    best_clean = evaluate_clean_distance(best)

    # Historiales para la convergencia (métrica limpia)
    clean_best_hist = []
    clean_curr_hist = []
    clean_std_hist  = []  # aquí 0 siempre, pero lo dejamos por compatibilidad

    # Schedule geométrico: T_k = start_temp * alpha^k
    if end_temp <= 0.0:
        end_temp = 1e-6
    alpha = (end_temp / start_temp) ** (1.0 / max(1, n_iter))

    start_t = time.time()

    for it in range(1, n_iter + 1):
        T = start_temp * (alpha ** it)

        candidate = sa_neighbor(current, env, G)
        cand_cost = penalized_cost(candidate)

        # Aceptación de Metropolis
        accept = False
        if cand_cost < curr_cost:
            accept = True
        else:
            delta = cand_cost - curr_cost
            prob = math.exp(-delta / T) if T > 0 else 0.0
            if random.random() < prob:
                accept = True

        if accept:
            current = candidate
            curr_cost = cand_cost

        # Actualizar mejor solución global
        if curr_cost < best_cost:
            best = [r[:] for r in current]
            best_cost = curr_cost
            best_clean = evaluate_clean_distance(best)

        # Métrica limpia en la solución actual
        curr_clean = evaluate_clean_distance(current)

        clean_best_hist.append(best_clean)
        clean_curr_hist.append(curr_clean)
        clean_std_hist.append(0.0)

        # Debug periódicamente
        if debug_interval is not None and debug_interval > 0:
            if it % debug_interval == 0 or it == 1:
                conflicts, mindist = detect_conflicts(best)
                print(
                    f"[IT {it:5d}] "
                    f"T={T:.4f}  "
                    f"BestPenalized={best_cost:.2f}  "
                    f"BestCleanDist={best_clean:.2f}  "
                    f"CurrCleanDist={curr_clean:.2f}  "
                    f"MinDist={mindist:.2f}  "
                    f"Conflicts={len(conflicts)}"
                )

    end_t = time.time()

    # Resultados finales
    best_penalized = best_cost
    best_distance  = best_clean

    print("\n===== SA COMPLETADO =====")
    print(f"Mejor fitness penalizado = {best_penalized:.3f}")
    print(f"Mejor distancia real     = {best_distance:.3f}")
    print(f"Tiempo = {end_t - start_t:.2f} s")

    if show_plots:
        plot_routes_sa(env, best, starts, picks, drops)
        plot_convergence_sa(clean_best_hist, clean_curr_hist, clean_std_hist)

    ani = None
    if show_anim or save_anim:
        ani = visualize_routes_timed_sa(env, best, starts, picks, drops)

    if save_anim and ani is not None:
        print(f"Guardando vídeo SA en: {anim_file}")
        total_frames = max(len(r) for r in best)
        writer = FFMpegWriterWithProgress(
            total_frames=total_frames,
            fps=20,
            bitrate=1800
        )
        ani.save(anim_file, writer=writer)
        print("✔ Vídeo guardado.")

    return {
        "best_penalized": best_penalized,
        "best_distance": best_distance,
        "routes": best,
        "clean_best": clean_best_hist,
        "clean_avg": clean_curr_hist,   # aquí 'avg' = solución actual
        "clean_std": clean_std_hist,
        "time_sec": end_t - start_t
    }


if __name__ == "__main__":
    run_sa(
        n_iter=5000,
        start_temp=5.0,
        end_temp=0.01,
        seed=42,
        show_plots=False,
        show_anim=False,
        save_anim=False,
        anim_file="routes_animation_sa_15000_10_001_42.mp4",
        debug_interval=500
    )

"""
Top 3 mejores configuraciones:

Params: {'n_iter': 12000, 'start_temp': 5, 'end_temp': 0.01}
Score: 2170.2597637085464
Time: 58.89351360003153

Params: {'n_iter': 5000, 'start_temp': 20, 'end_temp': 0.01}
Score: 2275.522967144143
Time: 23.348228295644123

Params: {'n_iter': 8000, 'start_temp': 10, 'end_temp': 0.01}
Score: 2294.497834896004
Time: 36.90306027730306
"""