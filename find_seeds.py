import sqlite3
import json
import os

base_path = '/home/dmore/code/Optimizacion/GA-hospital/GA-Based-Multi-Agent-Optimal-Path-Planning-for-Intra-Hospital-Transport/hyperparametrization/results'

def get_best_row(db_name, algo_name, param_criteria):
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
                    # Handle potential key mismatch or type mismatch
                    if k not in p:
                        match = False
                        break
                    # Be tolerant with int vs float for generic numbers if needed, but usually equality is fine
                    if p[k] != v:
                        match = False
                        break
                if match:
                    matched_rows.append(row)
            except:
                continue
        else:
            # NSGA-II case: params are columns
            match = True
            for k, v in param_criteria.items():
                if k not in row.keys():
                    match = False
                    break
                if row[k] != v:
                    match = False
                    break
            if match:
                matched_rows.append(row)
                
    if not matched_rows:
        return None, None
        
    # Find best
    if algo_name == 'NSGA-II':
        # Max Hypervolume
        best_row = max(matched_rows, key=lambda r: r['hv'])
        score = best_row['hv']
    else:
        # Min Score (Time/Dist/Cost)
        best_row = min(matched_rows, key=lambda r: r['score'])
        score = best_row['score']
        
    return best_row['seed'], score

# Define Criteria
# Corrected keys based on DB sample
ga_criteria = {
    'pop_size': 150,
    'ngen': 800,
    'cxpb': 0.8,
    'mutpb': 0.3
    # 'tourn_size': 3 # Will verify if this key exists
}

mulambda_criteria = {
    'mu': 120,
    'lambda_': 50,
    'ngen': 800,
    'cxpb': 0.5,
    'mutpb': 0.5
}

sa_criteria = {
    'n_iter': 5000,
    'start_temp': 20
}

nsga2_criteria = {
    'pop_size': 120,
    'ngen': 1000,
    'cxpb': 0.6,
    'mutpb': 0.2
}

# --- Run ---

print("--- GA ---")
try:
    conn = sqlite3.connect(os.path.join(base_path, 'grid_results_ga.db'))
    cur = conn.cursor()
    # Fetch random row to check keys fully
    row = cur.execute("SELECT params FROM runs LIMIT 1").fetchone()
    if row: print(f"Sample Params: {row[0]}")
    conn.close()
    
    # Try with tourn_size first, if fails try without
    seed, score = get_best_row('grid_results_ga.db', 'GA', {**ga_criteria, 'tourn_size': 3})
    if seed is None:
        print("Match with tourn_size=3 failed, trying without...")
        seed, score = get_best_row('grid_results_ga.db', 'GA', ga_criteria)
        
    print(f"Best Seed: {seed}, Score: {score}")
except Exception as e:
    print(f"Error GA: {e}")

print("\n--- Mu+Lambda ---")
try:
    conn = sqlite3.connect(os.path.join(base_path, 'grid_results_mulambda.db'))
    cur = conn.cursor()
    row = cur.execute("SELECT params FROM runs LIMIT 1").fetchone()
    if row: print(f"Sample Params: {row[0]}")
    conn.close()
    
    seed, score = get_best_row('grid_results_mulambda.db', 'ES', mulambda_criteria)
    print(f"Best Seed: {seed}, Score: {score}")
except Exception as e:
    print(f"Error ES: {e}")

print("\n--- SA ---")
try:
    conn = sqlite3.connect(os.path.join(base_path, 'grid_results_sa.db'))
    cur = conn.cursor()
    row = cur.execute("SELECT params FROM runs LIMIT 1").fetchone()
    if row: print(f"Sample Params: {row[0]}")
    conn.close()
    
    seed, score = get_best_row('grid_results_sa.db', 'SA', sa_criteria)
    print(f"Best Seed: {seed}, Score: {score}")
except Exception as e:
    print(f"Error SA: {e}")

print("\n--- NSGA-II ---")
try:
    seed, score = get_best_row('nsga2_grid.db', 'NSGA-II', nsga2_criteria)
    print(f"Best Seed: {seed}, Max HV: {score}")
except Exception as e:
    print(f"Error NSGA-II: {e}")
