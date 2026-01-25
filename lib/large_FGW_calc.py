#Imports
import pandas as pd
import numpy as np

import os,sys
sys.path.append(os.path.realpath('/home/uccaahw/src/FGW/lib'))
from ot_distances import Fused_Gromov_Wasserstein_distance
from joblib import Parallel, delayed, cpu_count, dump, load
import tempfile
import pickle
import ot
import argparse
import time

# Global variables for worker processes
_worker_graphs_1 = None
_worker_graphs_2 = None
_worker_fgw = None

def _init_worker(graphs_path_1, graphs_path_2, alpha, features_metric, method, force_recompute):
    """Initialize worker process with loaded graphs and FGW instance"""
    global _worker_graphs_1, _worker_graphs_2, _worker_fgw
    print(f"Initializing worker process {os.getpid()}")
    # Load without mmap to avoid file descriptor issues
    _worker_graphs_1 = load(graphs_path_1)
    _worker_graphs_2 = load(graphs_path_2)
    _worker_fgw = Fused_Gromov_Wasserstein_distance(
        alpha=alpha,
        features_metric=features_metric,
        method=method,
        force_recompute=force_recompute
    )
    print(f"Worker {os.getpid()} initialized with {len(_worker_graphs_1)} and {len(_worker_graphs_2)} graphs")

def compute_fgw_row(row_idx):
    """Compute FGW distances for a single row using pre-loaded graphs"""
    global _worker_graphs_1, _worker_graphs_2, _worker_fgw
    
    if _worker_graphs_2 is None:
        raise RuntimeError(f"Worker not initialized! _worker_graphs_2 is None in process {os.getpid()}")
    
    n_cols = len(_worker_graphs_2)
    row_results = []
    
    for col_idx in range(n_cols):
        try:
            dist = round(_worker_fgw.graph_d(_worker_graphs_1[row_idx], _worker_graphs_2[col_idx]), 6)
        except Exception as e:
            print(f"Error computing distance for row {row_idx}, col {col_idx}: {e}")
            dist = np.nan
        row_results.append((row_idx, col_idx, dist))
    
    return row_results

def run_FGW_two_files(graph_file_1, graph_file_2, alphas, distance, featurisation, method, 
                      n_cores=cpu_count()-1, force_recompute=False):
    """
    Compute FGW distance matrix between two sets of graphs.
    
    Args:
        graph_file_1: Path to first graph pickle file
        graph_file_2: Path to second graph pickle file
        alphas: List of alpha values for FGW
        distance: Distance metric for features
        featurisation: Feature type name
        method: Method for graph distance computation
        n_cores: Number of cores to use
        force_recompute: Whether to force recomputation
    """
    start_time = time.time()
    
    if method is not None:
        force_recompute = True
        if method not in ["shortest_path", "square_shortest_path", "weighted_shortest_path", 
                         "adjacency", "harmonic_distance", "true_distance",'distance_weighted_adjacency']:
            print("Please use one of the methods listed in the Graph.distance_matrix() function. "
                  "Defaulting to harmonic_distance")
            method = "harmonic_distance"
    
    print(f"Running with alphas {alphas}, feat {featurisation}, distance {distance}, "
          f"and method {method if method is not None else 'atomic distance'}")
    
    # Load both graph files
    with open(graph_file_1, 'rb') as f:
        graph_dict_1 = pickle.load(f)
    with open(graph_file_2, 'rb') as f:
        graph_dict_2 = pickle.load(f)
    
    n_rows = len(graph_dict_1)
    n_cols = len(graph_dict_2)
    
    print(f"Computing {n_rows} x {n_cols} distance matrix")
    
    # Create temporary directory and save graphs
    temp_dir = tempfile.mkdtemp()
    graphs_path_1 = os.path.join(temp_dir, 'graphs_1.pkl')
    graphs_path_2 = os.path.join(temp_dir, 'graphs_2.pkl')
    
    mpids_1, graphs_1 = zip(*graph_dict_1.items())
    mpids_2, graphs_2 = zip(*graph_dict_2.items())
    
    dump(graphs_1, graphs_path_1)
    dump(graphs_2, graphs_path_2)
    
    print(f'Running with {n_cores} jobs...')
    
    # Run in parallel for each alpha
    for alpha in alphas:
        alpha_time = time.time()
        
        # Use multiprocessing Pool with initializer to load graphs once per worker
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
        
        with multiprocessing.Pool(
            processes=n_cores,
            initializer=_init_worker,
            initargs=(graphs_path_1, graphs_path_2, alpha, distance, method, force_recompute)
        ) as pool:
            parallel_results = pool.map(compute_fgw_row, range(n_rows))
        
        print("Parallel computation complete")
        
        # Fill final matrix
        distance_matrix = np.zeros((n_rows, n_cols), dtype='f')
        for row_result in parallel_results:
            for row_idx, col_idx, dist in row_result:
                distance_matrix[row_idx, col_idx] = dist
        
        distance_matrix_df = pd.DataFrame(
            distance_matrix, 
            index=mpids_1, 
            columns=mpids_2
        ).astype(np.float32).round(6)
        
        # Save results
        if method is not None:
            filename = f'gen_vs_train_{featurisation}_{alpha:.2f}_{method}_{distance}.csv'
        else:
            filename = f'gen_vs_train_{featurisation}_{alpha:.2f}_atomic_distance_{distance}.csv'
        
        distance_matrix_df.to_csv(filename)
        print(f'Generated {filename} in {time.time() - alpha_time:.4f} seconds')
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    end_time = time.time()
    print(f"Total running time: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    # Example usage
    graph_file_1 = "train_graphs/graph_skip.pkl"
    graph_file_2 = "gen_graphs/generated_skip_graphs.pkl"
    featurisation = 'skip'
    alphas = [0,0.5,1.0]
    distance = 'cosine'
    method = 'distance_weighted_adjacency'
    
    run_FGW_two_files(
        graph_file_1=graph_file_1,
        graph_file_2=graph_file_2,
        alphas=alphas,
        distance=distance,
        featurisation=featurisation,
        method=method,
        n_cores=39
    )
