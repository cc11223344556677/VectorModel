import numpy as np
from vector_model import *
from analysis import *

N_DIMS = 3             
N_ATT_VECS = 3           
N_PERS_VECS = 2          

N_STATIC_TOPICS = 1    
N_DYNAMIC_TOPICS = 3     
TOPIC_DECAY_RATE = 0.95 
TOPIC_REPLACE_THRESHOLD = 0.1  

N_AGENTS = 10

EPSILON_T_OP = 0.4       
EPSILON_R_OP = 0.2    
EPSILON_T_VEC = 0.4     
EPSILON_R_VEC = 0.2      
LAMBDA_PARAM = 100

MESSAGE_RATE = 1.5  
N_TARGETS = 2            
SIMILARITY_THRESHOLD = 0.6  # threshold for similarity-based selection

N_STEPS = 100
SIMILARITY_METHOD: SIMILARITY_METHODS = 'cosine'

DEBUG_LEVEL: DEBUG_LEVELS = 'detailed'

SEED = 0


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
    n_vecs=N_ATT_VECS,
    opinion_assimilative_method='closest',
    vector_assimilative_method='closest',
    opinion_repulsive_method='furthest',
    vector_repulsive_method='furthest',
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
    n_att_vecs=N_ATT_VECS,
    n_pers_vecs=N_PERS_VECS,
    similarity_method=SIMILARITY_METHOD,
    data_collector=DataCollector('detailed'),
)

for i in range(N_AGENTS):
    message_selector = MessageSelector(
        methods=['select_all'],
        similarity_method=SIMILARITY_METHOD,
        max_messages=None,
        rng=rng,
    )

    message_producer = MessageProducer(
        method='probabilistic',
        similarity_method=SIMILARITY_METHOD,
        message_rate=MESSAGE_RATE,
        n_targets=N_TARGETS,
        n_max_messages=1,
        rng=rng,
        include_opinions=True,
        include_att_vecs=True,
        include_pers_vecs=True,
    )
    
    agent = VectorAgent(
        model=model,
        n_dims=N_DIMS,
        n_att_vecs=N_ATT_VECS,
        n_pers_vecs=N_PERS_VECS,
        message_selector=message_selector,
        message_producer=message_producer,
        similarity_method=SIMILARITY_METHOD,
        agent_type='test',
        rng=rng
        #rng=np.random.default_rng(seed=rng.integers(0, 1000, 1))
    )
    model.agents.add(agent)



print(f"Running vector opinion dynamics model")
print(f"  Agents: {N_AGENTS}")
print(f"  Topics: {N_STATIC_TOPICS} static + {N_DYNAMIC_TOPICS} dynamic")
print(f"  Dimensions: {N_DIMS}")
print(f"  Attitude vectors per agent: {N_ATT_VECS}")
print(f"  Personality vectors per agent: {N_PERS_VECS}")
print(f"  Steps: {N_STEPS}")
print(f"  Debug level: {DEBUG_LEVEL}")
print(f"  Seed: {SEED}")

# Run the model
model.run(N_STEPS)

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)



if DEBUG_LEVEL != 'none':
    print(f"\nCollected debug data for {len(model.data_collector.step_data)} steps")
    
    agent_num = 2
    agent_opinions = []
    for step_data in model.data_collector.step_data:
        for agent_data in step_data:    
            if agent_data.agent_id == agent_num:
                agent_opinions.append(agent_data.opinions_after)
                break
    
    if agent_opinions:
        print(f"\nAgent {agent_num} opinion trajectory on topic 0:")
        for step, opinions in enumerate(agent_opinions):
            if opinions:
                print(f"  Step {step}: {opinions[0][1]:+.4f}")