# ============================================================
# ACO MULTI-AGENTE + ANIMACIÓN
# Usando las funciones reales de ga_core.py
# ============================================================

import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ga_core import (
    create_graph,
    detect_conflicts,
    prepare_environment,
    MOVE_DIAG,
    MOVE_ORTH,
)


# ============================================================
# VISUALIZACIÓN ANIMADA
# ============================================================

def animate_aco_routes(env, rutas, interval=200):
    """
    Animación del movimiento simultáneo de todos los agentes.
    Cada frame corresponde al timestep t.
    """

    # --------------------------
    # Preprocesar rutas por timestep
    # --------------------------
    max_t = 0
    rutas_por_agente = []

    for ruta in rutas:
        d = {}
        for (t, nodo) in ruta:
            d[t] = nodo
            if t > max_t:
                max_t = t
        rutas_por_agente.append(d)

    num_agents = len(rutas)

    # --------------------------
    # Crear figura
    # --------------------------
    H, W = env.shape
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(env, cmap="gray_r", origin="upper")
    ax.set_title("Evolución temporal de rutas ACO")
    ax.set_xticks([])
    ax.set_yticks([])

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink",
              "tab:olive", "tab:cyan", "tab:gray"]

    puntos = []
    for k in range(num_agents):
        p, = ax.plot([], [], "o", color=colors[k % len(colors)], markersize=10)
        puntos.append(p)

    # Texto timestep
    time_text = ax.text(5, H + 2, "", fontsize=16, color="white")

    # --------------------------
    # Frame update
    # --------------------------
    def update(frame_t):
        for i in range(num_agents):
            pos_dict = rutas_por_agente[i]
            if frame_t in pos_dict:
                y, x = pos_dict[frame_t]
                puntos[i].set_data([x], [y])

        time_text.set_text(f"t = {frame_t}")
        return puntos + [time_text]

    # --------------------------
    # Animación
    # --------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=range(0, max_t + 1),
        interval=interval,
        blit=True,
    )

    plt.show()
    return anim


# ============================================================
# ACO MULTI-AGENTE USANDO ga_core.py
# ============================================================

class ACO_MultiAgent:

    def __init__(
        self,
        env,
        starts,
        pickups,
        drops,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        Q=10.0,
        num_ants=30,
        iterations=80,
        MIN_SEP=6.0,
        w_dist=1.0,
        w_conf=1000.0,
        verbose=True,
    ):

        self.env = env
        self.starts = starts
        self.pickups = pickups
        self.drops = drops

        # usa tu propio create_graph
        self.G = create_graph(env)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.num_ants = num_ants
        self.iterations = iterations
        self.MIN_SEP = MIN_SEP

        self.w_dist = w_dist
        self.w_conf = w_conf
        self.verbose = verbose

        # feromonas
        self.pheromone = defaultdict(lambda: 0.01)
        for u in self.G:
            for v in self.G[u]:
                self.pheromone[(u, v)] = 0.01

        self.num_agents = len(starts)

    # --------------------------------------------------------

    def dist(self, a, b):
        dy = abs(a[0] - b[0])
        dx = abs(a[1] - b[1])
        return MOVE_DIAG if dy + dx == 2 else MOVE_ORTH

    # --------------------------------------------------------

    def choose_next(self, u, timestep, ocupacion):
        vecinos = list(self.G[u])
        total = 0
        probs = []

        for v in vecinos:
            # penal temporal
            penal = 0.0001 if (timestep, v) in ocupacion else 1.0

            tau = self.pheromone[(u, v)]
            eta = 1.0 / (self.dist(u, v) + 1e-9)

            val = penal * ((tau ** self.alpha) * (eta ** self.beta))
            probs.append(val)
            total += val

        if total == 0:
            return random.choice(vecinos)

        probs = [p / total for p in probs]
        r = random.random()
        acc = 0
        for v, p in zip(vecinos, probs):
            acc += p
            if r <= acc:
                return v

        return vecinos[-1]

    # --------------------------------------------------------

    def build_route_single(self, start, mid, end, ocupacion):
        ruta = []
        t = 0

        curr = start
        while curr != mid:
            nxt = self.choose_next(curr, t, ocupacion)
            ruta.append((t, nxt))
            ocupacion[(t, nxt)] = True
            curr = nxt
            t += 1

        curr = mid
        while curr != end:
            nxt = self.choose_next(curr, t, ocupacion)
            ruta.append((t, nxt))
            ocupacion[(t, nxt)] = True
            curr = nxt
            t += 1

        return ruta

    # --------------------------------------------------------

    def build_solution_multi(self):
        rutas = []
        ocupacion = {}

        for i in range(self.num_agents):
            ruta_i = self.build_route_single(
                self.starts[i],
                self.pickups[i],
                self.drops[i],
                ocupacion
            )
            rutas.append(ruta_i)

        return rutas

    # --------------------------------------------------------

    def score_solution(self, rutas):
        total_dist = 0
        for r in rutas:
            nodos = [n for _, n in r]
            for a, b in zip(nodos[:-1], nodos[1:]):
                total_dist += self.dist(a, b)

        # usa tu detect_conflicts real
        conflicts, min_dist = detect_conflicts([ [n for _, n in ruta] for ruta in rutas ])
        penalty = self.w_conf * len(conflicts)

        return total_dist + penalty, total_dist, len(conflicts)

    # --------------------------------------------------------

    def evaporate(self):
        for k in self.pheromone:
            self.pheromone[k] *= (1 - self.rho)

    def reinforce(self, rutas, score):
        dep = self.Q / (score + 1e-9)
        for ruta in rutas:
            nodos = [n for _, n in ruta]
            for a, b in zip(nodos[:-1], nodos[1:]):
                self.pheromone[(a, b)] += dep

    # --------------------------------------------------------

    def run(self):
        best_routes = None
        best_score = float("inf")

        for it in range(1, self.iterations + 1):
            candidatos = []

            for _ in range(self.num_ants):
                rutas = self.build_solution_multi()
                score, dist, conf = self.score_solution(rutas)
                candidatos.append((score, rutas))

            candidatos.sort(key=lambda x: x[0])
            score_best, rutas_best = candidatos[0]

            self.evaporate()
            self.reinforce(rutas_best, score_best)

            if score_best < best_score:
                best_score = score_best
                best_routes = rutas_best

            if self.verbose:
                print(f"[ITER {it}] Best={score_best:.2f}")

        return best_routes, best_score


# ============================================================
# EJECUCIÓN REAL
# ============================================================

if __name__ == "__main__":

    env, starts, picks, drops = prepare_environment(show_grid=False)  # usa tu mapa real

    aco = ACO_MultiAgent(
        env,
        starts,
        picks,
        drops,
        num_ants=25,
        iterations=60,
        verbose=True,
    )

    rutas, score = aco.run()

    print("ACO terminado. Mejor score:", score)
    animate_aco_routes(env, rutas, interval=150)
