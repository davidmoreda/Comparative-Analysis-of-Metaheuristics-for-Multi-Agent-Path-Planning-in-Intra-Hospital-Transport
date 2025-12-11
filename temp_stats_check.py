
import csv
import math
import statistics

CSV_PATH = 'statistical analisis/results/combined/all_algorithms_combined.csv'

def calculate_stats():
    data = {}
    
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            # Header: algorithm, best_fitness, best_clean, time_sec
            
            for row in reader:
                if not row: continue
                algo = row[0]
                # best_fitness is index 1 (penalized), best_clean is index 2
                try:
                    val = float(row[1]) 
                except:
                    continue
                    
                if algo not in data:
                    data[algo] = []
                data[algo].append(val)
                
        print(f"{'Algorithm':<15} {'Mean':<10} {'Std':<10} {'CV(%)':<10} {'95% CI':<20} {'Best':<10}")
        print("-" * 75)
        
        for algo, values in data.items():
            n = len(values)
            if n < 2: continue
            
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            cv = (stdev / mean) * 100 if mean != 0 else 0
            best = min(values)
            
            # 95% CI for small sample (t-dist approx 2.045 for n=30)
            margin = 2.045 * (stdev / math.sqrt(n))
            
            print(f"{algo:<15} {mean:<10.2f} {stdev:<10.2f} {cv:<10.2f} +/- {margin:<10.2f}      {best:<10.2f}")

    except FileNotFoundError:
        print("CSV file not found.")

if __name__ == "__main__":
    calculate_stats()
