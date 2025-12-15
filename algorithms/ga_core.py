import numpy as np
import networkx as nx
import math, random
from deap import base, creator, tools
import cv2
import matplotlib.pyplot as plt


# ==============================================================
# CONFIG
# ==============================================================

MOVE_ORTH = 1.0
MOVE_DIAG = math.sqrt(2)

MIN_SEP = 6.0          # distancia mínima requerida

WAIT_PENAL        = 0.05    
WAIT_BLOCK_WEIGHT = 1.5      
W_BACK            = 3.0       
BIG_PENALTY       = 5000      # para cosas "muy mal" (sin ruta, pick/drop mal, etc.)
COLLISION_PENALTY = 150       # penalización por conflicto temporal (ajustable)



random.seed(42)

# ==============================================================
# NEAREST FREE
# ==============================================================

def nearest_free_black(env, y, x, max_r=40):
    H, W = env.shape
    if 0 <= y < H and 0 <= x < W and env[y, x] == 0:
        return (y, x)

    for r in range(1, max_r):
        for dy in range(-r, r+1):
            for dx in (-r, r):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and env[yy, xx] == 0:
                    return (yy, xx)

        for dx in range(-r, r+1):
            for dy in (-r, r):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and env[yy, xx] == 0:
                    return (yy, xx)
    return None

# ==============================================================
# GRAPH CREATION
# ==============================================================

def create_graph(env):
    H, W = env.shape
    G = nx.Graph()
    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    for y in range(H):
        for x in range(W):
            if env[y, x] != 0:
                continue
            G.add_node((y, x))
            for dy, dx in moves:
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and env[yy, xx] == 0:
                    w = MOVE_DIAG if abs(dy)+abs(dx) == 2 else MOVE_ORTH
                    G.add_edge((y, x), (yy, xx), weight=w)
    return G

# ==============================================================
# SHORTEST
# ==============================================================

def shortest(G, a, b):
    try:
        return nx.shortest_path(G, a, b, weight="weight")
    except:
        return None

# ==============================================================
# BUILD ROUTE
# ==============================================================

def build_route(G, start, pick, drop):
    s1 = shortest(G, start, pick)
    s2 = shortest(G, pick, drop)
    s3 = shortest(G, drop, start)
    if s1 is None or s2 is None or s3 is None:
        return None
    return s1[:-1] + s2[:-1] + s3

# ==============================================================

 #MUTATION AUX
# ==============================================================

def mutate_wait(route, G):
    if len(route) < 5:
        return route
    i = random.randint(1, len(route)-2)
    new = route[:i] + [route[i]] + route[i:]
    if not G.has_edge(new[i], new[i+1]):
        return route
    return new

def mutate_segment(route, G):
    if len(route) < 25:
        return route
    seg_len = random.randint(4, 7)
    i = random.randint(1, len(route)-seg_len-2)
    A = route[i]
    B = route[i+seg_len]
    mid = shortest(G, A, B)
    if mid is None:
        return route
    new = route[:i] + mid + route[i+seg_len+1:]
    if i > 0 and not G.has_edge(route[i-1], mid[0]):
        return route
    if (i+seg_len+1 < len(route)) and not G.has_edge(mid[-1], route[i+seg_len+1]):
        return route
    return new

def detect_conflicts(routes):
    K = len(routes)
    maxT = max(len(r) for r in routes)
    conflicts = []
    min_dist = float("inf")

    for t in range(maxT):
        pos = [r[min(t, len(r)-1)] for r in routes]
        for i in range(K):
            for j in range(i+1, K):
                y1,x1 = pos[i]; y2,x2 = pos[j]
                d = math.hypot(y1-y2, x1-x2)
                min_dist = min(min_dist, d)
                if d < MIN_SEP:
                    conflicts.append(t)
    return conflicts, min_dist



# ==============================================================
# NEW MUTATIONS (MACRO, WAIT, SHIFT)
# ==============================================================

def mutate_macro_detour(route, G, env, radius=12):
    if len(route) < 30:
        return route
    H, W = env.shape
    i = random.randint(5, len(route)-6)
    A = route[i]
    B = route[i+4]

    ay, ax = A
    candidates = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            yy = ay + dy
            xx = ax + dx
            if 0 <= yy < H and 0 <= xx < W and env[yy, xx] == 0:
                candidates.append((yy, xx))
    if not candidates:
        return route

    C = random.choice(candidates)
    p1 = shortest(G, A, C)
    p2 = shortest(G, C, B)
    if p1 is None or p2 is None:
        return route

    new = route[:i] + p1 + p2 + route[i+5:]
    if i > 0 and not G.has_edge(route[i-1], p1[0]):
        return route
    if i+5 < len(route) and not G.has_edge(p2[-1], route[i+5]):
        return route
    return new

def mutate_long_wait(route, G):
    if len(route) < 10:
        return route
    i = random.randint(3, len(route)-3)
    wlen = random.randint(3, 12)
    new = route[:i] + [route[i]]*wlen + route[i:]
    if i > 0 and not G.has_edge(new[i-1], new[i]):
        return route
    return new

def mutate_shift_start(route, G):
    delay = random.randint(10, 30)
    return [route[0]]*delay + route

# ==============================================================
# COST FUNCTIONS
# ==============================================================
def cost_distance(route):
    total = 0.0
    for (y1,x1),(y2,x2) in zip(route[:-1], route[1:]):
        total += MOVE_DIAG if abs(y1-y2)+abs(x1-x2) == 2 else MOVE_ORTH
    return total

def cost_waits(route):
    waits = [i for i in range(1, len(route)) if route[i] == route[i-1]]
    return sum((k+1)**2 for k,_ in enumerate(waits))

def cost_wait_blocks(route):
    blocks = 0
    i = 1
    while i < len(route):
        if route[i] == route[i-1]:
            blocks += 1
            while i < len(route) and route[i] == route[i-1]:
                i += 1
        i += 1
    return blocks * WAIT_BLOCK_WEIGHT

def cost_backtracking(route):
    return sum(W_BACK for i in range(len(route)-2)
               if route[i] == route[i+2])

def penal_pick_drop(route, pick, drop):
    if pick not in route or drop not in route:
        return BIG_PENALTY
    if route.index(pick) > route.index(drop):
        return BIG_PENALTY
    return 0.0

def penal_temporal(routes):
    """
    Penalización temporal GRADUAL:
    - Usa detect_conflicts para contar cuántos time-steps tienen conflictos.
    - Cada conflicto añade COLLISION_PENALTY al coste.
    Así el GA ve mejora al reducir conflictos, en vez de todo o nada.
    """
    conflicts, _ = detect_conflicts(routes)
    if not conflicts:
        return 0.0
    # Puedes cambiar a len(set(conflicts)) si quieres penalizar por timestep
    # en lugar de por par de agentes.
    return COLLISION_PENALTY * len(conflicts)

# ==============================================================
# FITNESS LIMPIO (SOLO DISTANCIA REAL)
# ==============================================================

def evaluate_clean_distance(ind):
    """
    Retorna únicamente la suma de distancias reales de todos los agentes,
    sin penalizaciones ni otros costes.
    """
    total = 0.0
    for r in ind:
        total += cost_distance(r)
    return total

# ==============================================================
# FITNESS CLÁSICO
# ==============================================================

def evaluate(ind, picks, drops, base_routes):
    if any(r is None for r in ind):
        return (BIG_PENALTY,)

    total = 0.0
    for k, r in enumerate(ind):
        total += cost_distance(r)
        total += WAIT_PENAL * cost_waits(r)
        total += cost_wait_blocks(r)
        total += cost_backtracking(r)
        total += penal_pick_drop(r, picks[k], drops[k])

    total += penal_temporal(ind)
    return (total,)

# ==============================================================
# FITNESS MULTI
# ==============================================================

def evaluate_multi(ind, picks, drops, base_routes):
    """
    Versión multiobjetivo para NSGA-II.
    Devuelve:
      - objetivo 1: fitness penalizado (como evaluate)
      - objetivo 2: distancia limpia total (evaluate_clean_distance)
    """
    penal_tuple = evaluate(ind, picks, drops, base_routes)
    penal = penal_tuple[0]
    clean = evaluate_clean_distance(ind)
    return (penal, clean)

# ==============================================================
# GA SETUP
# ==============================================================

# Evitar duplicar creación de clases DEAP
# --------------------------------------------------------------
# Fitness mono-objetivo (GA clásico)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Fitness multi-objetivo (penalizado, limpio)
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Individuos para cada caso
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

if not hasattr(creator, "IndividualMulti"):
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

def ga_setup(env, starts, picks, drops, multi=False):
    """
    Configura el entorno y las funciones registradas de DEAP para el GA.
    Si multi=True → usa evaluate_multi y la clase IndividualMulti (NSGA-II)
    Si multi=False → usa evaluate clásico y la clase Individual (GA simple)
    """
    global G
    G = create_graph(env)
    K = len(starts)

    # --- rutas base ---
    base_routes = []
    for k in range(K):
        r = build_route(G, starts[k], picks[k], drops[k])
        if r is None:
            raise RuntimeError("No se pudo construir la ruta base.")
        base_routes.append(r)

    tb = base.Toolbox()

    # --- seleccionar clase de individuo ---
    IndClass = creator.IndividualMulti if multi else creator.Individual

    # --- inicialización ---
    def init_ind():
        return IndClass([br[:] for br in base_routes])

    tb.register("individual", init_ind)
    tb.register("population", tools.initRepeat, list, tb.individual)

    # --- operador de mutación ---
    def mutate(ind):
        conflicts, _ = detect_conflicts(ind)
        for k in range(len(ind)):
            r = ind[k]
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
            if conflicts and random.random() < 0.95:
                r = mutate_conflict(r, G, conflicts)
            ind[k] = r
        return (ind,)

    # --- operador de cruce ---
    def mate(a, b):
        child1 = IndClass(a[:])
        child2 = IndClass(b[:])
        for i in range(len(a)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    tb.register("mutate", mutate)
    tb.register("mate", mate)
    tb.register("select", tools.selTournament, tournsize=3)

    # --- elegir función de evaluación ---
    if multi:
        tb.register("evaluate", evaluate_multi,
                    picks=picks, drops=drops, base_routes=base_routes)
    else:
        tb.register("evaluate", evaluate,
                    picks=picks, drops=drops, base_routes=base_routes)

    return tb, base_routes

# ==============================================================
# ENVIRONMENT PREPARATION
# ==============================================================

def load_env_from_bmp(path):
    """
    Carga el BMP del hospital y lo invierte para que:
        - 0 = suelo (libre)
        - 255 = pared (obstáculo)
    """
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    print("[INFO] Valores únicos del BMP original:", np.unique(img_gray))

    env = 255 - img_gray.astype(np.uint8)

    print("[INFO] Valores únicos tras invertir:", np.unique(env))
    return env

def show_env_grid_with_points(env, starts, picks, drops):
    H, W = env.shape
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(env, cmap="gray", vmin=0, vmax=255,
              origin="upper", interpolation="nearest")

    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="red", linewidth=0.25)

    colors = ["green", "blue", "orange", "purple", "cyan", "yellow"]
    markers = {"start": "o", "pick": "x", "drop": "s"}

    for k, (s, p, d) in enumerate(zip(starts, picks, drops)):
        color = colors[k % len(colors)]
        sy, sx = s
        py, px = p
        dy, dx = d

        ax.plot(sx, sy, markers["start"], color=color, markersize=10)
        ax.plot(px, py, markers["pick"], color=color,
                markersize=12, markeredgewidth=2)
        ax.plot(dx, dy, markers["drop"], color=color, markersize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def prepare_environment(show_grid=False):
    """
    Carga el environment desde el BMP completo, genera starts/picks/drops
    y opcionalmente los visualiza sobre el mapa con simbología por agente.
    """
    env = load_env_from_bmp("/home/dmore/code/Optimizacion/GA-hospital/GA-Based-Multi-Agent-Optimal-Path-Planning-for-Intra-Hospital-Transport/data/Mapa.bmp")

    H, W = env.shape
    print(f"[INFO] Environment cargado: {H} x {W}")

    starts_raw = [(5,10), (90,10), (50,90), (63,12)]
    picks_raw  = [(40,50), (60,40), (30,60), (12,38)]
    drops_raw  = [(80,80), (20,80), (70,20), (8,78)]

    starts = [nearest_free_black(env, y, x) for y, x in starts_raw]
    picks  = [nearest_free_black(env, y, x) for y, x in picks_raw]
    drops  = [nearest_free_black(env, y, x) for y, x in drops_raw]

    if show_grid:
        show_env_grid_with_points(env, starts, picks, drops)

    return env, starts, picks, drops
