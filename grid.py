import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import numpy as np

from vector_model import *
from analysis import *
from model_runner import run_model


def get_base_params():
    return {
        'AGENT_TYPE': 'test',
        'N_DIMS': 2,
        'N_DYN_VECS': 3,
        'N_PERS_VECS': 0,
        'N_STATIC_TOPICS': 1,
        'N_DYNAMIC_TOPICS': 10,
        'TOPIC_DECAY_RATE': 0.95,
        'TOPIC_REPLACE_THRESHOLD': 0.1,
        'N_AGENTS': 50,
        'OPINION_ASSIMILATIVE_METHOD': 'closest',
        'VECTOR_ASSIMILATIVE_METHOD': 'closest',
        'OPINION_REPULSIVE_METHOD': 'furthest',
        'VECTOR_REPULSTIVE_METHOD': 'furthest',
        'EPSILON_T_OP': 0.1,
        'EPSILON_R_OP': 0.2,
        'EPSILON_T_VEC': 0.1,
        'EPSILON_R_VEC': 0.2,
        'LAMBDA_PARAM': 0.1,
        'MESSAGE_RATE': 1.0,
        'MAX_TARGETS': 4,
        'INCLUDE_OPINIONS': True,
        'INCLUDE_DYN_VECS': True,
        'INCLUDE_PERS_VECS': True,
        'N_STEPS': 100,
        'SIMILARITY_METHOD': 'tanh',
        'SELECTOR_METHODS': ['select_randomly'],
        'MAX_MESSAGES_SELECTED': 10,
        'OPINION_SIMILARITY_THRESHOLD': 0.4,
        'VECTOR_SIMILARITY_THRESHOLD': 0.4,
        'N_MAX_MESSAGES': 4,
        'PRODUCER_METHODS': ['opinionated'],
        'DEBUG_LEVEL': 'summary',
        'SEED': 1,
    }


def generate_parameter_grid():
    grid = {
        'N_DIMS': [2,4,10],
        'N_DYN_VECS': [1, 4],
        'N_PERS_VECS': [0, 1, 4],
        'N_STATIC_TOPICS': [1,5],
        'N_DYNAMIC_TOPICS': [0,1,5],
        'N_AGENTS': [50, 200],
        'LAMBDA_PARAM': [0.1, 0.5, 1.0],
        'MAX_TARGETS':[1, 5],
        'MAX_MESSAGES_SELECTED': [1, 10],
        'SELECTOR_METHODS': [    
            'select_all',
            'select_randomly',
            'select_directly',
            'select_from_similar_opinions',
            'select_from_similar_personalities_matched',
            'select_from_similar_personalities_unmatched',
            'select_from_similar_dyn_vecs_matched',
            'select_from_similar_dyn_vecs_unmatched',
            ],
        'PRODUCER_METHODS': [
            'opinionated', 
            'random_targets'
        ]
    }
    
    param_combinations = []
    keys = list(grid.keys())
    values = list(grid.values())
    
    for combo in product(*values):
        params = get_base_params()
        
        params['DATA_COLLECTOR'] = DataCollector('detailed', n_agents_to_track=1, n_messages_to_show=3)
        
        name_parts = []
        for key, value in zip(keys, combo):
            if key in ['SELECTOR_METHODS', 'PRODUCER_METHODS']:
                if isinstance(value, str):
                    params[key] = [value]
                    value_str = value
                elif isinstance(value, list):
                    params[key] = value
                    value_str = '_'.join(str(v) for v in value)
                else:
                    params[key] = [value]
                    value_str = str(value)
            elif isinstance(value, list):
                params[key] = value
                value_str = '_'.join(str(v) for v in value)
            else:
                params[key] = value
                value_str = str(value)
            
            key_short = ''.join([c for c in key if c.isupper() or c.isdigit()])
            name_parts.append(f"{key_short}_{value_str}")
        
        params['run_name'] = "_".join(name_parts)
        param_combinations.append(params)
    
    return param_combinations


def main():
    param_grid = generate_parameter_grid()
    
    print(f"Starting grid search with {len(param_grid)} parameter combinations")
    print(f"Using 16 threads")
    print("="*80)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_params = {executor.submit(run_model, params): params for params in param_grid}
        
        for i, future in enumerate(as_completed(future_to_params), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                print(f"[{i}/{len(param_grid)}] ✓ {result['run_name']}")
                print(f"    Bipolarization: {result['bipolarization']:.4f}, Entropy: {result['shannon_entropy']:.4f}")
            elif result['status'] == 'skipped':
                print(f"[{i}/{len(param_grid)}] ⊘ {result['run_name']} (skipped - already exists)")
            else:
                print(f"[{i}/{len(param_grid)}] ✗ {result['run_name']}")
                print(f"    Error: {result['error']}")
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Successful runs: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"Skipped runs: {sum(1 for r in results if r['status'] == 'skipped')}/{len(results)}")
    print(f"Failed runs: {sum(1 for r in results if r['status'] == 'error')}/{len(results)}")


if __name__ == '__main__':
    main()