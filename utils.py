def print_results(results_dict):
    """Print formatted results"""
    for phase, phase_results in results_dict.items():
        print(f"\n{phase.upper()} Results:")
        for method, metrics in phase_results.items():
            print(f"{method}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

def find_best_method(results_dict, metric='accuracy'):
    """Find the best method based on a specific metric"""
    return max(results_dict.items(), key=lambda x: x[1][metric])[0]