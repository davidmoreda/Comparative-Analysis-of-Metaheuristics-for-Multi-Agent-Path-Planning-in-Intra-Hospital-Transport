# GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport

This repository contains the source code, data, and experimental results for the paper **"GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport"**.

The framework proposes a complete **Evolutionary Computation (EC)** pipeline to generate safe, efficient, and collision-free routes for multiple autonomous agents during Intra-Hospital Transport (IHT) missions.

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **Problem Formulation** | Modeling of the hospital environment as a directed graph, formalizing a **Multi-Agent Optimal Path Planning (MAOPP)** problem with operational and safety constraints. |
| **Multi-Objective Optimization** | Use of **NSGA-II** to optimize **path distance** *(F_clean)* and **safety compliance** *(F_pen)*, including penalties for collisions and insufficient separation. |
| **Algorithmic Benchmark** | Systematic comparison of **GA**, **Simulated Annealing (SA)**, **(μ+λ) ES**, and **NSGA-II** under a shared representation. |
| **Real Hospital Environment** | Experiments executed on a graph extracted from a real hospital floor plan. |

---

# 📁 Project Structure

```
.
├── algorithms/                 # Core evolutionary algorithms + MAOPP logic
│   ├── ga_core.py              # Encoding, evaluation, operators
│   ├── ga_runner.py            # Single-objective GA
│   ├── mulambda_runner.py      # (μ+λ) Evolution Strategy
│   ├── sa_runner.py            # Simulated Annealing
│   └── ga_runner_multi.py      # NSGA-II (multi-objective)
│
├── data/                       
│   └── Floorplan/              # Hospital floorplan images
│
├── hyperparametrization/
│   ├── grid_search.py          # Single-objective grid search
│   └── grid_search_algos.py    # Multi-objective grid search (NSGA-II)
│
├── results/                    # SQLite DB with all experiment results
├── figures/                    # Generated plots
└── README.md
```

---

# ⚙️ Methodology and Results

## **Single-Objective Performance**

A systematic benchmark revealed the superiority of the **(μ+λ) Evolution Strategy** for minimizing the penalized distance \( F_{\text{pen}} \).

| Algorithm | Min Penalized Distance | Mean Penalized Distance |
| :--- | :--- | :--- |
| **(μ+λ) ES** | **1104.66** | **1446.59** |
| GA | 1184.63 | 1453.53 |
| SA | 1989.32 | 2297.90 |

### Distribution of Penalized Distance

<br>

<img src="figures/fig_single_boxplots.png" width="600px">

<br>

---

## **Multi-Objective Optimization (NSGA-II)**

NSGA-II was used to identify the Pareto-optimal set for the bi-objective problem:

\[
\min(F_{\text{clean}},\; F_{\text{pen}})
\]

The best configurations achieved a hypervolume exceeding **\(1.54 \times 10^7\)**.

### Best Achieved Pareto Front

<br>

<img src="figures/fig_nsga2_pareto.png" width="650px">

<br>

### Optimized Multi-Agent Routes

<br>

<img src="figures/Routes_multiobjective_comparison_10_1000_06_04_s42.png" width="700px">

<br>

---

# 📄 Paper Reference

If you use this methodology, please cite:

````bibtex
@article{amezcua2025ga,
  title={GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport},
  author={Amezcua, David Moreda and Silva, Carmen Gutiérrez},
  journal={IEEE Access},
  year={2025},
  volume={XX},
  number={XX},
  pages={XXXX-XXXX},
  doi={10.1109/ACCESS.2017.DOI}
}
