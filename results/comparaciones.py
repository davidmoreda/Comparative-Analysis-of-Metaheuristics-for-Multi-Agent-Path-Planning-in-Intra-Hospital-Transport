# comparador_total_svg.py
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from algorithms.ga_runner import run_ga
from algorithms.ga_runner_multi import run_ga_multi
from algorithms.mulambda_runner import run_mulambda
from algorithms.sa_runner import run_sa


# ============================================================
#       GENERADOR DE NOMBRE DESCRIPTIVO (SEGÚN PARÁMETROS)
# ============================================================
def make_tag(algoname, params):
    p = "_".join([f"{k}{v}" for k, v in params.items()
                  if isinstance(v, (int, float, str))])
    return f"{algoname}_{p}".replace(".", "-").replace("__", "_")


# ============================================================
#           Crea carpeta results si no existe
# ============================================================
os.makedirs("results", exist_ok=True)


# ============================================================
#       1) EJECUTAR TODOS LOS ALGORITMOS
# ============================================================
def ejecutar_todos(params_ga, params_sa, params_es, params_mo):
    print("\n===== Ejecutando GA Simple =====")
    res_ga = run_ga(**params_ga)

    print("\n===== Ejecutando SA =====")
    res_sa = run_sa(**params_sa)

    print("\n===== Ejecutando μ+λ =====")
    res_es = run_mulambda(**params_es)

    print("\n===== Ejecutando NSGA-II =====")
    res_mo = run_ga_multi(**params_mo)

    return {
        "GA": (res_ga, params_ga),
        "SA": (res_sa, params_sa),
        "MULAMBDA": (res_es, params_es),
        "NSGAII": (res_mo, params_mo)
    }


# ============================================================
#       2) GRÁFICAS DE CONVERGENCIA
# ============================================================
def plot_convergencia_best(results):
    plt.figure(figsize=(10,5))
    plt.yscale("log")

    # GA simple
    r, p = results["GA"]
    plt.plot(r["clean_best"], label="GA")

    # μ+λ
    r, p = results["MULAMBDA"]
    plt.plot(r["clean_best"], label="μ+λ")

    # SA
    r, p = results["SA"]
    plt.plot(r["clean_best"], label="SA")

    # NSGA-II
    r, p = results["NSGAII"]
    mins = r["logbook"].select("min")
    clean = [c[1] for c in mins]
    plt.plot(clean, label="NSGA-II")

    plt.xlabel("Gen / Iter")
    plt.ylabel("Clean distance (log)")
    plt.title("Convergencia — Mejor distancia real")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = "results/convergencia_best.svg"
    plt.savefig(filename, format="svg")
    print(f"Guardado: {filename}")


# ============================================================
#       3) GRÁFICA DE MEDIA / SOLUCIÓN ACTUAL
# ============================================================
def plot_convergencia_avg(results):
    plt.figure(figsize=(10,5))
    plt.yscale("log")

    # GA
    r, p = results["GA"]
    plt.plot(r["clean_avg"], label="GA avg", linestyle="--")

    # μ+λ
    r, p = results["MULAMBDA"]
    plt.plot(r["clean_avg"], label="μ+λ avg", linestyle="--")

    # SA
    r, p = results["SA"]
    plt.plot(r["clean_avg"], label="SA current", linestyle="--")

    # NSGA-II (no tiene avg limpio directo → usar clean)
    r, p = results["NSGAII"]
    mins = r["logbook"].select("avg")
    clean_avg = [m[1] for m in mins]
    plt.plot(clean_avg, label="NSGA-II avg", linestyle="--")

    plt.xlabel("Gen / Iter")
    plt.ylabel("Clean distance (log)")
    plt.title("Convergencia — Media / solución actual")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = "results/convergencia_avg.svg"
    plt.savefig(filename, format="svg")
    print(f"Guardado: {filename}")


# ============================================================
#       4) BARRAS: MEJOR DISTANCIA Y TIEMPO
# ============================================================
def plot_resumen_barras(results):
    labels = ["GA", "SA", "μ+λ", "NSGA-II"]

    best_dist = [
        results["GA"][0]["best_clean"],
        results["SA"][0]["best_distance"],
        results["MULAMBDA"][0]["best_distance"],
        results["NSGAII"][0]["best_tradeoff"].fitness.values[1],
    ]
    tiempos = [
        results["GA"][0]["time_sec"],
        results["SA"][0]["time_sec"],
        results["MULAMBDA"][0]["time_sec"],
        results["NSGAII"][0]["logbook"].select("gen")[-1],
    ]

    x = np.arange(len(labels))

    # Distancias
    plt.figure(figsize=(8,4))
    plt.bar(x, best_dist)
    plt.xticks(x, labels)
    plt.ylabel("Best clean distance")
    plt.title("Comparación de mejores soluciones")
    plt.tight_layout()
    plt.savefig("results/bar_best_distance.svg", format="svg")
    print("Guardado: results/bar_best_distance.svg")

    # Tiempo
    plt.figure(figsize=(8,4))
    plt.bar(x, tiempos)
    plt.xticks(x, labels)
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de tiempos")
    plt.tight_layout()
    plt.savefig("results/bar_tiempos.svg", format="svg")
    print("Guardado: results/bar_tiempos.svg")

# ============================================================
#       MAIN
# ============================================================
def main():
    params_ga = dict(
        pop_size=193, ngen=861, cxpb=0.86, mutpb=0.285,
        seed=42, show_plots=False, show_anim=False
    )

    params_sa = dict(
        n_iter=10000, start_temp=10, end_temp=0.01, seed=42,
        show_plots=False, show_anim=False
    )

    params_es = dict(
        pop_size=193, ngen=861, cxpb=0.86, mutpb=0.285,
        seed=42, show_plots=False, show_anim=False
    )

    params_mo = dict(
        pop_size=193, ngen=861, cxpb=0.86, mutpb=0.285,
        seed=42, show_plots=False, show_anim=False
    )

    results = ejecutar_todos(params_ga, params_sa, params_es, params_mo)

    plot_convergencia_best(results)
    plot_convergencia_avg(results)
    plot_resumen_barras(results)

    print("\n✔ Todas las figuras fueron guardadas en /results/*.svg")


if __name__ == "__main__":
    main()
