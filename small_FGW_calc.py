#Imports
import pandas as pd
import numpy as np

import os,sys
sys.path.append(os.path.realpath('/home/matt/src/FGW/lib'))
from ot_distances import Fused_Gromov_Wasserstein_distance
from joblib import Parallel, delayed, cpu_count, dump, load
import tempfile
import pickle
import ot
import argparse
import time

def compute_fgw_chunk(graph_path, pair_indices, alpha, features_metric,method,force_recompute=False):
    graph_data = load(graph_path, mmap_mode='r')
    fgw = Fused_Gromov_Wasserstein_distance(
        alpha=alpha,
        features_metric=features_metric,
        method=method,
        force_recompute=force_recompute
    )
    results = []
    for i, j in pair_indices:
        try:
            dist = round(fgw.graph_d(graph_data[i], graph_data[j]),6)
        except Exception:
            dist = 0.0
        results.append((i, j, dist))
    return results
def get_upper_triangle_indices(n):
    return [(i, j) for i in range(n) for j in range(i, n)]

def split_indices(indices, n_chunks):
    k, m = divmod(len(indices), n_chunks)
    return [indices[i * k + min(i, m):(i+1) * k + min(i+1, m)] for i in range(n_chunks)]

def run_FGW(alphas, distance, featurisation, method,n_cores=cpu_count()-1,graph_dir = 'graphs',force_recompute=False):
    start_time = time.time()
    if method is not None:
        force_recompute=True
        if method not in ["shortest_path","square_shortest_path", "weighted_shortest_path", "adjacency", "harmonic_distance","true_distance"]:
            print("Please use one of the methods listed in the Graph.distance_matrix() function in https://github.com/tvayer/FGW/blob/master/lib/graph.py. Defaulting to harmonic_distance")
            method = "harmonic_distance"
    print(f"Running with alphas {alphas}, feat {featurisation}, distance {distance}, and method {method if method is not None else 'atomic distance'}")
    graph_file = "generated_skip_graphs_temp_0.5.pkl"
    with open(graph_file, 'rb') as f:
        graph_dict = pickle.load(f)
    n=len(graph_dict)
    temp_dir = tempfile.mkdtemp()
    graphs_path = os.path.join(temp_dir, 'graphs.pkl')
    mpids, graphs = zip(*graph_dict.items())
    dump(graphs, graphs_path)
    upper_tri_indices = get_upper_triangle_indices(n)
    index_chunks = split_indices(upper_tri_indices, n_cores)
    print(f'Running with {n_cores} jobs...')
    # Run in parallel
    for alpha in alphas:
        alpha_time = time.time()
        parallel_results = Parallel(n_jobs=n_cores)(
        delayed(compute_fgw_chunk)(graphs_path, chunk, alpha,distance,method,force_recompute) for chunk in index_chunks)
        print("Parallel computation complete")
    # Fill final matrix
        distance_matrix = np.zeros((n, n),dtype='f')
        for chunk_result in parallel_results:
            for i, j, dist in chunk_result:
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # if symmetric
        #distance_matrix_Norm = distance_matrix/np.max(distance_matrix)
        distance_matrix_df = pd.DataFrame(distance_matrix, index=mpids,columns=mpids).astype(np.float32).round(6)
        #distance_matrix_Norm_df = pd.DataFrame(distance_matrix_Norm, index=mpids,columns=mpids)
        if method is not None:
            distance_matrix_df.to_csv(f'fabini_{featurisation}FGW_{alpha:.2f}_{method}_{distance}.csv')
            print(f'generated fabini_{featurisation}FGW_{alpha:.2f}_{method}_{distance}.csv in {time.time() - alpha_time:.4f} seconds')  
        else:
            distance_matrix_df.to_csv(f'fabini_{featurisation}FGW_{alpha:.2f}_atomic_distance_{distance}.csv')
            print(f'generated fabini_{featurisation}FGW_{alpha:.2f}_atomic_distance_{distance}.csv in {time.time() - alpha_time:.4f} seconds')  
            
    end_time = time.time()
    print(f"Total running time: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    featurisation = 'skip'
    alphas = [0.5] #np.linspace(0,1,21)
    distance = 'cosine'
    method = 'harmonic_distance'
    run_FGW(alphas, distance, featurisation, method,n_cores=40) 
