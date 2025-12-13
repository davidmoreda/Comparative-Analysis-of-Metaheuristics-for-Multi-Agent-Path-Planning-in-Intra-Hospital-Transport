import sqlite3
import json
import os
# from tabulate import tabulate

base_path = '/home/dmore/code/Optimizacion/GA-hospital/GA-Based-Multi-Agent-Optimal-Path-Planning-for-Intra-Hospital-Transport/hyperparametrization/results'

def get_best_row_details(db_name, algo_name, param_criteria):
    db_path = os.path.join(base_path, db_name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM runs")
    rows = cursor.fetchall()
    conn.close()
    
    matched_rows = []
    
    for row in rows:
        # Check params
        if 'params' in row.keys():
            try:
                p = json.loads(row['params'])
                match = True
                for k, v in param_criteria.items():
                    if k not in p: match = False; break
                    if p[k] != v: match = False; break
                if match:
                    matched_rows.append((row, p))
            except:
                continue
        else:
            # NSGA-II case
            match = True
            for k, v in param_criteria.items():
                if k not in row.keys(): match = False; break
                if row[k] != v: match = False; break
            if match:
                # Create a dict for params to normalize interface
                p = {k: row[k] for k in param_criteria.keys()}
                matched_rows.append((row, p))
                
    if not matched_rows:
        return None, None, None
        
    # Find best
    if algo_name == 'NSGA-II':
        best_row, best_params = max(matched_rows, key=lambda x: x[0]['hv'])
        score = best_row['hv']
    else:
        best_row, best_params = min(matched_rows, key=lambda x: x[0]['score'])
        score = best_row['score']
        
    return best_row, score, best_params

# Criteria (Same as before)
ga_criteria = {'pop_size': 150, 'ngen': 800, 'cxpb': 0.8, 'mutpb': 0.3}
mulambda_criteria = {'mu': 120, 'lambda_': 50, 'ngen': 800, 'cxpb': 0.5, 'mutpb': 0.5}
sa_criteria = {'n_iter': 5000, 'start_temp': 20}
nsga2_criteria = {'pop_size': 120, 'ngen': 1000, 'cxpb': 0.6, 'mutpb': 0.2}

# Data Collection
data = []

# GA
r, s, p = get_best_row_details('grid_results_ga.db', 'GA', ga_criteria)
if r: data.append({'Algo': 'GA', 'Score': s, 'Seed': r['seed'], 'Params': p})

# ES (Mu+Lambda)
r, s, p = get_best_row_details('grid_results_mulambda.db', '(μ+λ) ES', mulambda_criteria)
if r: data.append({'Algo': '(μ+λ) ES', 'Score': s, 'Seed': r['seed'], 'Params': p})

# SA
r, s, p = get_best_row_details('grid_results_sa.db', 'SA', sa_criteria)
if r: data.append({'Algo': 'SA', 'Score': s, 'Seed': r['seed'], 'Params': p})

# NSGA-II
r, s, p = get_best_row_details('nsga2_grid.db', 'NSGA-II', nsga2_criteria)
if r: data.append({'Algo': 'NSGA-II', 'Score': s, 'Seed': r['seed'], 'Params': p})


# --- Output ---
print(f"| {'Algorithm':<15} | {'Best Result':<15} | {'Metric':<15} | {'Best Seed':<10} |")
print(f"|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*12}|")

for item in data:
    metric = "Max HV" if item['Algo'] == 'NSGA-II' else "Min Cost"
    val_str = f"{item['Score']:.2f}" if item['Algo'] != 'NSGA-II' else f"{item['Score']:.2e}"
    print(f"| {item['Algo']:<15} | {val_str:<15} | {metric:<15} | {item['Seed']:<10} |")

print("\n\n")

print(f"| {'Algorithm':<15} | {'Pop/Mu':<10} | {'Gen/Iter':<10} | {'CxProb':<8} | {'MutProb':<8} | {'Special Params':<30} |")
print(f"|{'-'*17}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*32}|")

for item in data:
    p = item['Params']
    # Defaults
    pop = "-"
    gen = "-"
    cx = "-"
    mut = "-"
    special = []
    
    if item['Algo'] == 'GA':
        pop = p.get('pop_size')
        gen = p.get('ngen')
        cx = p.get('cxpb')
        mut = p.get('mutpb')
        # Check for tourn_size in original row if not in criteria
        # (Assuming standard keys)
        special.append("TournSize=3") # Hardcoded based on knowledge of exp, or check p if we put it there
    
    elif item['Algo'] == '(μ+λ) ES':
        pop = p.get('mu') # Mu is population essentially
        gen = p.get('ngen')
        cx = p.get('cxpb')
        mut = p.get('mutpb')
        special.append(f"λ={p.get('lambda_')}")
        
    elif item['Algo'] == 'SA':
        gen = p.get('n_iter')
        special.append(f"T0={p.get('start_temp')}")
        special.append("Decay=Geom") # Known const
        
    elif item['Algo'] == 'NSGA-II':
        pop = p.get('pop_size')
        gen = p.get('ngen')
        cx = p.get('cxpb')
        mut = p.get('mutpb')
    
    special_str = ", ".join(special)
    print(f"| {item['Algo']:<15} | {str(pop):<10} | {str(gen):<10} | {str(cx):<8} | {str(mut):<8} | {special_str:<30} |")
