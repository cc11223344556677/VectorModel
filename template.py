# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pathlib import Path

import numpy as np
from vector_model import *
from analysis import *

AGENT_TYPE = 'test'

N_DIMS = 3            
N_DYN_VECS = 3          
N_PERS_VECS = 0      

N_STATIC_TOPICS = 1
N_DYNAMIC_TOPICS = 10
TOPIC_DECAY_RATE = 0.95 
TOPIC_REPLACE_THRESHOLD = 0.1  

N_AGENTS = 50

OPINION_ASSIMILATIVE_METHOD = 'closest'
VECTOR_ASSIMILATIVE_METHOD = 'closest'
OPINION_REPULSIVE_METHOD = 'furthest'
VECTOR_REPULSTIVE_METHOD = 'furthest'

EPSILON_T_OP = 0.15
EPSILON_R_OP = 0.75
EPSILON_T_VEC = 0.15
EPSILON_R_VEC = 0.75
LAMBDA_PARAM = 0.1

MESSAGE_RATE = 1.5  
MAX_TARGETS = 4          

N_STEPS = 1000
SIMILARITY_METHOD: SIMILARITY_METHODS = 'tanh'

SELECTOR_METHODS: List[MessageSelector.ALLOWED_METHODS] = ['select_randomly']
MAX_MESSAGES_SELECTED = 10
OPINION_SIMILARITY_THRESHOLD = 0.4
VECTOR_SIMILARITY_THRESHOLD = 0.4

N_MAX_MESSAGES = 4
PRODUCER_METHODS: List[MessageProducer.PRODUCER_METHODS] = ['opinionated']

DEBUG_LEVEL: DEBUG_LEVELS = 'summary'

SEED = 1

DATA_COLLECTOR = DataCollector('detailed', n_agents_to_track=1, n_messages_to_show=3)

rng = np.random.default_rng(SEED)

topic_space = TopicSpace(
    n_dims=N_DIMS,
    n_static_topics=N_STATIC_TOPICS,
    n_dynamic_topics=N_DYNAMIC_TOPICS,
    decay_rate=TOPIC_DECAY_RATE,
    replacement_threshold=TOPIC_REPLACE_THRESHOLD,
    rng=rng
)

opinion_updater = OpinionUpdater(
    n_dims=N_DIMS,
    n_vecs=N_DYN_VECS,
    opinion_assimilative_method=OPINION_ASSIMILATIVE_METHOD,
    vector_assimilative_method=VECTOR_ASSIMILATIVE_METHOD,
    opinion_repulsive_method=OPINION_REPULSIVE_METHOD,
    vector_repulsive_method=VECTOR_REPULSTIVE_METHOD,
    similarity_method=SIMILARITY_METHOD,
    epsilon_T_op=EPSILON_T_OP,
    epsilon_R_op=EPSILON_R_OP,
    epsilon_T_vec=EPSILON_T_VEC,
    epsilon_R_vec=EPSILON_R_VEC,
    lambda_param=LAMBDA_PARAM,
)

model = VectorModel(
    n_agents=N_AGENTS,
    topic_space=topic_space,
    opinion_updater=opinion_updater,
    n_dims=N_DIMS,
    n_dyn_vecs=N_DYN_VECS,
    n_pers_vecs=N_PERS_VECS,
    similarity_method=SIMILARITY_METHOD,
    data_collector=DATA_COLLECTOR,
)

for i in range(N_AGENTS):
    message_selector = MessageSelector(
        methods=SELECTOR_METHODS,
        similarity_method=SIMILARITY_METHOD,
        opinion_similarity_threshold=OPINION_SIMILARITY_THRESHOLD,
        vector_similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        max_messages_selected=MAX_MESSAGES_SELECTED,
        rng=rng,
    )

    message_producer = MessageProducer(
        methods=PRODUCER_METHODS,
        similarity_method=SIMILARITY_METHOD,
        message_rate=MESSAGE_RATE,
        max_targets=MAX_TARGETS,
        n_max_messages=N_MAX_MESSAGES,
        rng=rng,
        include_opinions=True,
        include_dyn_vecs=True,
        include_pers_vecs=True,
    )
    
    agent = VectorAgent(
        model=model,
        n_dims=N_DIMS,
        n_dyn_vecs=N_DYN_VECS,
        n_pers_vecs=N_PERS_VECS,
        message_selector=message_selector,
        message_producer=message_producer,
        similarity_method=SIMILARITY_METHOD,
        agent_type=AGENT_TYPE,
        rng=rng
    )
    model.agents.add(agent)



print(f"Running vector opinion dynamics model")
print(f"  Agents: {N_AGENTS}")
print(f"  Topics: {N_STATIC_TOPICS} static + {N_DYNAMIC_TOPICS} dynamic")
print(f"  Dimensions: {N_DIMS}")
print(f"  Dynamic vectors per agent: {N_DYN_VECS}")
print(f"  Personality vectors per agent: {N_PERS_VECS}")
print(f"  Steps: {N_STEPS}")
print(f"  Debug level: {DEBUG_LEVEL}")
print(f"  Seed: {SEED}")

model.run(N_STEPS)

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)
                
bipolarization = calculate_bipolarization(model, 0)
print(f'FINAL BIPOLALIZATION: {bipolarization}')

shannon_entropy = calculate_opinion_entropy(model, 0)
print(f'FINAL OPINION CLUSTERING (Shannon Entropy): {shannon_entropy}')

script_path = Path(__file__)
output_path = script_path.with_suffix(".jpg")

plot_opinion_trajectories(
    model.data_collector, 
    topic_id=0,
    save_path=str(output_path),
)

plot_all_topics(model.data_collector,
                N_STATIC_TOPICS)

# %%
