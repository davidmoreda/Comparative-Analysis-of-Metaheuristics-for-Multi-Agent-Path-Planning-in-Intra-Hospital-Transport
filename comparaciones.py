# comparaciones.py
import numpy as np
import matplotlib.pyplot as plt

from ga_runner import run_ga
from ga_runner_multi import run_ga
from sa_runner import run_sa
from mulambda_runner import run_mulambda


def ejecutar_todos():
    """
    Ejecuta GA, SA y μ+λ con unos parámetros razonables y devuelve
    sus resultados en un diccionario.
    """

    # ---------- Parámetros de cada algoritmo ----------
    # Ajusta estos si quieres que tengan presupuestos "similares"
    params_ga = dict(
        pop_size=100,
        ngen=200,
        cxpb=0.7,
        mutpb=0.3,
        seed=42,
        show_plots=False,
        show_anim=False,
        save_anim=False,
        debug_interval=50,
    )

    params_sa = dict(
        n_iter=5000,
        start_temp=10.0,
        end_temp=0.01,
        seed=42,
        show_plots=False,
        show_anim=False,
        save_anim=False,
        debug_interval=1000,
    )

    params_mulambda = dict(
        mu=100,
        lambda_=100,
        ngen=1333,
        cxpb=0.7,
        mutpb=0.3,
        seed=42,
        show_plots=False,
        show_anim=False,
        save_anim=False,
        debug_interval=50,
    )

    print("=== Ejecutando GA ===")
    res_ga = run_ga(**params_ga)

    print("\n=== Ejecutando SA ===")
    res_sa = run_sa(**params_sa)

    print("\n=== Ejecutando μ+λ ===")
    res_es = run_mulambda(**params_mulambda)

    return {
        "GA": res_ga,
        "SA": res_sa,
        "MULAMBDA": res_es,
    }


def plot_convergencia_best(results):
    """
    Compara la convergencia de la mejor distancia limpia (clean_best)
    de los tres algoritmos.
    """
    plt.figure(figsize=(10, 5))
    plt.yscale("log")

    # GA
    ga_best = np.array(results["GA"]["clean_best"], dtype=float)
    gens_ga = np.arange(1, len(ga_best) + 1)
    plt.plot(gens_ga, ga_best, label="GA (mejor)", linewidth=2)

    # μ+λ
    es_best = np.array(results["MULAMBDA"]["clean_best"], dtype=float)
    gens_es = np.arange(1, len(es_best) + 1)
    plt.plot(gens_es, es_best, label="μ+λ (mejor)", linewidth=2)

    # SA (usa iteraciones en vez de generaciones)
    sa_best = np.array(results["SA"]["clean_best"], dtype=float)
    it_sa = np.arange(1, len(sa_best) + 1)
    plt.plot(it_sa, sa_best, label="SA (mejor)", linewidth=2)

    plt.xlabel("Generación / Iteración")
    plt.ylabel("Distancia real (log)")
    plt.title("Convergencia de la mejor distancia real")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_convergencia_avg(results):
    """
    Compara la "media" de distancia limpia:
      - GA y μ+λ: clean_avg (media en la población)
      - SA: clean_avg se usa como distancia de la solución actual
    """
    plt.figure(figsize=(10, 5))
    plt.yscale("log")

    # GA
    ga_avg = np.array(results["GA"]["clean_avg"], dtype=float)
    gens_ga = np.arange(1, len(ga_avg) + 1)
    plt.plot(gens_ga, ga_avg, label="GA (media)", linestyle="--")

    # μ+λ
    es_avg = np.array(results["MULAMBDA"]["clean_avg"], dtype=float)
    gens_es = np.arange(1, len(es_avg) + 1)
    plt.plot(gens_es, es_avg, label="μ+λ (media)", linestyle="--")

    # SA: media = solución actual
    sa_avg = np.array(results["SA"]["clean_avg"], dtype=float)
    it_sa = np.arange(1, len(sa_avg) + 1)
    plt.plot(it_sa, sa_avg, label="SA (sol. actual)", linestyle="--")

    plt.xlabel("Generación / Iteración")
    plt.ylabel("Distancia real (log)")
    plt.title("Evolución de la distancia real media / actual")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_resumen_barras(results):
    """
    Pequeño resumen de:
      - mejor distancia alcanzada (best_distance)
      - tiempo de ejecución (time_sec)
    usando gráficas de barras.
    """
    nombres = ["GA", "SA", "μ+λ"]

    best_dist = [
        results["GA"]["best_distance"],
        results["SA"]["best_distance"],
        results["MULAMBDA"]["best_distance"],
    ]

    tiempos = [
        results["GA"]["time_sec"],
        results["SA"]["time_sec"],
        results["MULAMBDA"]["time_sec"],
    ]

    x = np.arange(len(nombres))

    # ---- Barra de mejores distancias ----
    plt.figure(figsize=(8, 4))
    plt.bar(x, best_dist)
    plt.xticks(x, nombres)
    plt.ylabel("Mejor distancia real")
    plt.title("Comparación de mejor distancia alcanzada")
    plt.tight_layout()
    plt.show()

    # ---- Barra de tiempos ----
    plt.figure(figsize=(8, 4))
    plt.bar(x, tiempos)
    plt.xticks(x, nombres)
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de tiempos de ejecución")
    plt.tight_layout()
    plt.show()


def main():
    # Ejecutar los tres algoritmos
    results = ejecutar_todos()

    # Mostrar un pequeño resumen numérico por consola
    print("\n===== RESUMEN NUMÉRICO =====")
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"  Mejor fitness penalizado: {res['best_penalized']:.3f}")
        print(f"  Mejor distancia real:     {res['best_distance']:.3f}")
        print(f"  Tiempo de ejecución:      {res['time_sec']:.2f} s")

    # Hacer las gráficas comparativas
    plot_convergencia_best(results)
    plot_convergencia_avg(results)
    plot_resumen_barras(results)


if __name__ == "__main__":
    main()
