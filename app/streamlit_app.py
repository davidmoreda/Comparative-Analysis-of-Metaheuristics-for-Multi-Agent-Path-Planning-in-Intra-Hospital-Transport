import sys
import pathlib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt
import time

# =============================================================
# PATHS
# =============================================================
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MAP_PATH = ROOT_DIR / "data" / "Mapa.bmp"

from algorithms.ga_core import nearest_free_black
from algorithms.ga_runner_multi import run_ga_multi_streamlit


# =============================================================
# LOAD MAP
# =============================================================
img_raw = Image.open(MAP_PATH).convert("L")
env = 255 - np.array(img_raw, dtype=np.uint8)  # free=0, wall=255
H, W = env.shape
img_display = img_raw.convert("RGB")


# =============================================================
# STREAMLIT SETTINGS
# =============================================================
st.set_page_config(page_title="Hospital MAOPP Planner", layout="wide")

N_AGENTS = 4
STAGES = ["start", "pickup", "drop"]


# =============================================================
# STATE INIT
# =============================================================
if "agent" not in st.session_state:
    st.session_state.agent = 0
if "stage" not in st.session_state:
    st.session_state.stage = 0
if "pending" not in st.session_state:
    st.session_state.pending = None
if "points" not in st.session_state:
    st.session_state.points = {a: {} for a in range(N_AGENTS)}
if "routes" not in st.session_state:
    st.session_state.routes = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False
if "auto_frame" not in st.session_state:
    st.session_state.auto_frame = 0


# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.title("📋 Estado del sistema")

st.sidebar.write(f"**Agente actual:** {st.session_state.agent+1}/{N_AGENTS}")
st.sidebar.write(f"**Etapa:** {STAGES[st.session_state.stage].upper()}")

if st.sidebar.button("🔄 Reset"):
    st.session_state.agent = 0
    st.session_state.stage = 0
    st.session_state.points = {a: {} for a in range(N_AGENTS)}
    st.session_state.pending = None
    st.session_state.routes = None
    st.session_state.auto_play = False
    st.session_state.auto_frame = 0
    st.rerun()


st.sidebar.markdown("---")
if st.sidebar.button("🚀 Optimizar rutas"):
    ok = all(len(st.session_state.points[a]) == 3 for a in range(N_AGENTS))

    if not ok:
        st.sidebar.error("🚫 Faltan puntos para algún agente.")
    else:
        starts = [st.session_state.points[a]["start"] for a in range(N_AGENTS)]
        picks  = [st.session_state.points[a]["pickup"] for a in range(N_AGENTS)]
        drops  = [st.session_state.points[a]["drop"] for a in range(N_AGENTS)]

        out = run_ga_multi_streamlit(
            env,
            starts=starts,
            picks=picks,
            drops=drops,
            pop_size=120,
            ngen=70,
            cxpb=0.6,
            mutpb=0.3,
            seed=0
        )
        st.session_state.routes = out["paths"]
        st.session_state.auto_frame = 0
        st.sidebar.success("✔ Rutas generadas.")
        st.rerun()


# =============================================================
# MAIN LAYOUT
# =============================================================
st.title("🏥 Hospital MAOPP – Planificador de rutas")
st.write("Selecciona puntos en el mapa: **start → pickup → drop**.")

col1, col2 = st.columns([1.2, 1])


# =============================================================
# MAPA — AGRANDADO PERO CON COORDENADAS REALES
# =============================================================
with col1:
    st.markdown("### 🗺️ Mapa hospitalario")

    coords = streamlit_image_coordinates(
        img_display,
        width=W*1.6,   # 🔥 mejor tamaño, no gigante
        height=H*1.6,
        key="map"
    )

    if coords is not None:
        y = int(coords["y"] / 1.6)
        x = int(coords["x"] / 1.6)
        if 0 <= y < H and 0 <= x < W:
            st.session_state.pending = (y, x)

    st.caption(f"Punto pendiente: {st.session_state.pending}")

    # CONFIRM
    if st.button("📍 Confirmar punto"):
        if st.session_state.pending is None:
            st.warning("Selecciona un punto primero.")
        else:
            y_raw, x_raw = st.session_state.pending

            pos = nearest_free_black(env, y_raw, x_raw)
            if pos is None:
                st.error("❌ Ese punto cae sobre un muro.")
            else:
                y, x = pos
                ag = st.session_state.agent
                stg = STAGES[st.session_state.stage]
                st.session_state.points[ag][stg] = (y, x)

                if st.session_state.stage < 2:
                    st.session_state.stage += 1
                else:
                    st.session_state.stage = 0
                    if st.session_state.agent < N_AGENTS - 1:
                        st.session_state.agent += 1

                st.session_state.pending = None
                st.rerun()


# =============================================================
# PUNTOS SELECCIONADOS
# =============================================================
with col2:
    st.markdown("### 🎯 Puntos seleccionados")
    fig_p, ax_p = plt.subplots(figsize=(4.8, 4.8))
    ax_p.imshow(img_display, origin="upper")

    colors = ["red", "cyan", "yellow", "lime"]
    markers = {"start": "o", "pickup": "s", "drop": "X"}

    for a in range(N_AGENTS):
        for k, (y, x) in st.session_state.points[a].items():
            ax_p.scatter(x, y, c=colors[a], marker=markers[k], s=80)
            ax_p.text(x+4, y+4, f"{a+1}-{k[0].upper()}", color=colors[a], fontsize=10)

    ax_p.set_xticks([])
    ax_p.set_yticks([])
    st.pyplot(fig_p, use_container_width=True)

    # =============================================================
# ANIMACIÓN CON PLOTLY — MAPA + RUTAS + PUNTOS + MÉTRICAS
# =============================================================
import plotly.graph_objects as go
import numpy as np

if st.session_state.routes is not None:

    st.markdown("## 🎬 Evolución de rutas (Plotly con mapa)")

    routes = st.session_state.routes
    n_agents = len(routes)
    max_len = max(len(r) for r in routes)
    colors = ["red", "cyan", "yellow", "lime"]

    # ==========================
    # MÉTRICAS
    # ==========================
    distances_total = []
    conflicts_list = []

    for t in range(max_len):
        # Distancia por agente
        dist_agents = []
        for a in range(n_agents):
            r = routes[a]
            if t == 0:
                dist_agents.append(0)
            else:
                d = 0
                for i in range(1, min(t+1, len(r))):
                    y1, x1 = r[i-1]
                    y2, x2 = r[i]
                    d += np.hypot(y2 - y1, x2 - x1)
                dist_agents.append(d)

        distances_total.append(dist_agents)

        # Conflictos
        conflicts = 0
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                y1, x1 = routes[i][min(t, len(routes[i])-1)]
                y2, x2 = routes[j][min(t, len(routes[j])-1)]
                if np.hypot(y2 - y1, x2 - x1) < 2:  # Umbral colisión
                    conflicts += 1

        conflicts_list.append(conflicts)


    # ==========================
    # FIGURA BASE
    # ==========================
    fig = go.Figure()

    # ----- Fondo del mapa -----
    fig.add_layout_image(
        dict(
            source=img_display,   # imagen PIL convertida a RGB
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=W,
            sizey=H,
            sizing="stretch",
            layer="below",
        )
    )

    # ==========================
    # RUTAS VACÍAS (inicio)
    # ==========================
    for i in range(n_agents):
        fig.add_trace(go.Scatter(
            x=[routes[i][0][1]],
            y=[H - routes[i][0][0]],   # invertimos eje Y para coincidir con mapa
            mode="lines+markers",
            line=dict(color=colors[i], width=4),
            marker=dict(size=10),
            name=f"Agente {i+1}"
        ))

    # ==========================
    # PUNTOS START, PICKUP, DROP
    # ==========================
    markers = {"start": "circle", "pickup": "square", "drop": "x"}

    for a in range(n_agents):
        for k, (yy, xx) in st.session_state.points[a].items():
            fig.add_trace(go.Scatter(
                x=[xx],
                y=[H - yy],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=colors[a],
                    symbol=markers[k]
                ),
                text=[f"{a+1}-{k[0].upper()}"],
                textposition="bottom right",
                showlegend=False
            ))

    # ==========================
    # FRAMES DE ANIMACIÓN
    # ==========================
    frames = []

    for t in range(max_len):
        frame_data = []

        for i in range(n_agents):
            r = routes[i]

            xs = [p[1] for p in r[:t+1]]
            ys = [H - p[0] for p in r[:t+1]]

            frame_data.append(go.Scatter(
                x=xs,
                y=ys
            ))

        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig.frames = frames

    # ==========================
    # CONTROLES DE ANIMACIÓN + MÉTRICAS
    # ==========================
    fig.update_layout(
        width=900,
        height=700,
        xaxis=dict(range=[0, W]),
        yaxis=dict(range=[0, H], scaleanchor="x", autorange=False),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "⏹ Stop",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    }
                ]
            }
        ],
        title="Animación de rutas con mapa hospitalario"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # MÉTRICAS — PANEL LATERAL
    # ==========================
    st.subheader("📊 Métricas en cada frame")

    st.write("Estas métricas se actualizan al mover el slider de tiempo.")
    st.write("*(Plotly proporciona el slider automáticamente)*")

    st.info("""
    - **Distancia por agente** (px recorridos hasta el frame actual)  
    - **Conflictos** (agentes más cerca de 2 px)  
    """)

