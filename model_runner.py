import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pathlib import Path
import traceback

import numpy as np
from vector_model import *
from analysis import *


def run_model(params, output_dir='grid_experiments', show_plot=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    run_name = params['run_name']
    jpg_path = output_dir / f"{run_name}.jpg"
    txt_path = output_dir / f"{run_name}.txt"
    
    if jpg_path.exists():
        return {
            'run_name': run_name,
            'status': 'skipped',
            'message': 'Already completed'
        }
    
    try:
        rng = np.random.default_rng(params['SEED'])
        
        topic_space = TopicSpace(
            n_dims=params['N_DIMS'],
            n_static_topics=params['N_STATIC_TOPICS'],
            n_dynamic_topics=params['N_DYNAMIC_TOPICS'],
            decay_rate=params['TOPIC_DECAY_RATE'],
            replacement_threshold=params['TOPIC_REPLACE_THRESHOLD'],
            rng=rng
        )
        
        opinion_updater = OpinionUpdater(
            n_dims=params['N_DIMS'],
            n_vecs=params['N_DYN_VECS'],
            opinion_assimilative_method=params['OPINION_ASSIMILATIVE_METHOD'],
            vector_assimilative_method=params['VECTOR_ASSIMILATIVE_METHOD'],
            opinion_repulsive_method=params['OPINION_REPULSIVE_METHOD'],
            vector_repulsive_method=params['VECTOR_REPULSTIVE_METHOD'],
            similarity_method=params['SIMILARITY_METHOD'],
            epsilon_T_op=params['EPSILON_T_OP'],
            epsilon_R_op=params['EPSILON_R_OP'],
            epsilon_T_vec=params['EPSILON_T_VEC'],
            epsilon_R_vec=params['EPSILON_R_VEC'],
            lambda_param=params['LAMBDA_PARAM'],
        )
        
        model = VectorModel(
            n_agents=params['N_AGENTS'],
            topic_space=topic_space,
            opinion_updater=opinion_updater,
            n_dims=params['N_DIMS'],
            n_dyn_vecs=params['N_DYN_VECS'],
            n_pers_vecs=params['N_PERS_VECS'],
            similarity_method=params['SIMILARITY_METHOD'],
            data_collector=params['DATA_COLLECTOR'],
        )
        
        for i in range(params['N_AGENTS']):
            message_selector = MessageSelector(
                methods=params['SELECTOR_METHODS'],
                similarity_method=params['SIMILARITY_METHOD'],
                opinion_similarity_threshold=params['OPINION_SIMILARITY_THRESHOLD'],
                vector_similarity_threshold=params['VECTOR_SIMILARITY_THRESHOLD'],
                max_messages_selected=params['MAX_MESSAGES_SELECTED'],
                rng=rng,
            )
            
            message_producer = MessageProducer(
                methods=params['PRODUCER_METHODS'],
                similarity_method=params['SIMILARITY_METHOD'],
                message_rate=params['MESSAGE_RATE'],
                max_targets=params['MAX_TARGETS'],
                n_max_messages=params['N_MAX_MESSAGES'],
                rng=rng,
                include_opinions=params['INCLUDE_OPINIONS'],
                include_dyn_vecs=params['INCLUDE_DYN_VECS'],
                include_pers_vecs=params['INCLUDE_PERS_VECS'],
            )
            
            agent = VectorAgent(
                model=model,
                n_dims=params['N_DIMS'],
                n_dyn_vecs=params['N_DYN_VECS'],
                n_pers_vecs=params['N_PERS_VECS'],
                message_selector=message_selector,
                message_producer=message_producer,
                similarity_method=params['SIMILARITY_METHOD'],
                agent_type=params['AGENT_TYPE'],
                rng=rng
            )
            model.agents.add(agent)
        
        model.run(params['N_STEPS'])
        
        bipolarization = calculate_bipolarization(model, 0)
        shannon_entropy = calculate_opinion_entropy(model, 0)
        
        fig, ax = plot_opinion_trajectories(
            model.data_collector, 
            topic_id=0,
            save_path=jpg_path,
            show_plot=show_plot
        )

        if not show_plot:
            plt.close('all')
            
        
        with open(txt_path, 'w') as f:
            f.write(f"Run: {run_name}\n")
            f.write("="*80 + "\n\n")
            f.write("PARAMETERS:\n")
            for key, value in params.items():
                if key != 'DATA_COLLECTOR' and key != 'run_name':
                    f.write(f"  {key}: {value}\n")
            f.write("\nRESULTS:\n")
            f.write(f"  Final Bipolarization: {bipolarization}\n")
            f.write(f"  Final Shannon Entropy: {shannon_entropy}\n")
        
        del model

        return {
            'run_name': run_name,
            'status': 'success',
            'bipolarization': bipolarization,
            'shannon_entropy': shannon_entropy,
            #'model': model,
            #'fig': fig,
            #'ax': ax
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        with open(txt_path, 'w') as f:
            f.write(f"Run: {run_name}\n")
            f.write("="*80 + "\n\n")
            f.write("PARAMETERS:\n")
            for key, value in params.items():
                if key != 'DATA_COLLECTOR' and key != 'run_name':
                    f.write(f"  {key}: {value}\n")
            f.write("\nERROR:\n")
            f.write(error_trace)
        
        return {
            'run_name': run_name,
            'status': 'error',
            'error': str(e)
        }