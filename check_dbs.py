import sqlite3
import os

db_dir = '/home/dmore/code/Optimizacion/GA-hospital/GA-Based-Multi-Agent-Optimal-Path-Planning-for-Intra-Hospital-Transport/hyperparametrization/results'
dbs = [
    'grid_results_ga.db', 
    'grid_results_mulambda.db', 
    'grid_results_sa.db',
    'nsga2_grid.db'
]

for db_name in dbs:
    db_path = os.path.join(db_dir, db_name)
    print(f"--- Database: {db_name} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            print(f"Columns: {col_names}")
            
            # Get one row to see sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            row = cursor.fetchone()
            print(f"Sample Row: {row}")
            print("-" * 20)
            
        conn.close()
    except Exception as e:
        print(f"Error reading {db_name}: {e}")
    print("\n")
