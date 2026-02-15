from vector_model import *


def create_lurker(
    model,
    n_dims,
    n_dyn_vecs,
    n_pers_vecs,
    similarity_method,
    opinion_similarity_threshold,
    vector_similarity_threshold,
    max_messages_selected,
    message_rate,
    max_targets,
    n_max_messages,
    rng,
):
    message_selector = MessageSelector(
        methods=['select_directly', 'select_randomly'],
        similarity_method=similarity_method,
        opinion_similarity_threshold=opinion_similarity_threshold,
        vector_similarity_threshold=vector_similarity_threshold,
        max_messages_selected=max_messages_selected,
        rng=rng,
    )

    message_producer = MessageProducer(
        methods=['reciprocal'],
        similarity_method=similarity_method,
        message_rate=message_rate / 2,
        max_targets=max_targets,
        n_max_messages=n_max_messages,
        rng=rng,
        include_opinions=True,
        include_dyn_vecs=True,
        include_pers_vecs=False,
    )

    lurker = VectorAgent(
        model=model,
        n_dims=n_dims,
        n_dyn_vecs=n_dyn_vecs,
        n_pers_vecs=n_pers_vecs,
        message_selector=message_selector,
        message_producer=message_producer,
        similarity_method=similarity_method,
        agent_type='Lurker',
        rng=rng
    )
    return lurker

def create_strategic_agent(
    model,
    n_dims,
    n_dyn_vecs,
    n_pers_vecs,
    similarity_method,
    opinion_similarity_threshold,
    vector_similarity_threshold,
    max_messages_selected,
    message_rate,
    max_targets,
    n_max_messages,
    rng,
    only_create_extremists = False,
    normal_range = (0.1, 0.9),
    extremism_topic = 0
):
    message_selector = MessageSelector(
        methods=['select_none'],
        similarity_method=similarity_method,
        opinion_similarity_threshold=opinion_similarity_threshold,
        vector_similarity_threshold=vector_similarity_threshold,
        max_messages_selected=max_messages_selected,
        rng=rng,
    )

    message_producer = MessageProducer(
        methods=['opinionated','reciprocal','dissimilar','similar'],
        similarity_method=similarity_method,
        message_rate=message_rate,
        max_targets=max_targets,
        n_max_messages=n_max_messages * 3,
        rng=rng,
        include_opinions=True,
        include_dyn_vecs=True,
        include_pers_vecs=False,
    )
    strategic_agent = None
    while strategic_agent is None:
        strategic_agent = VectorAgent(
            model=model,
            n_dims=n_dims,
            n_dyn_vecs=n_dyn_vecs,
            n_pers_vecs=n_pers_vecs,
            message_selector=message_selector,
            message_producer=message_producer,
            similarity_method=similarity_method,
            agent_type='Strategic',
            rng=rng
        )
        if only_create_extremists:
            start, end = normal_range
            opinion =strategic_agent.calculate_opinion(model.topic_space.get_topic(extremism_topic))
            if start < opinion and opinion < end:
                strategic_agent.remove()
                strategic_agent = None
    
    return strategic_agent

def create_commenter(
    model,
    n_dims,
    n_dyn_vecs,
    n_pers_vecs,
    similarity_method,
    opinion_similarity_threshold,
    vector_similarity_threshold,
    max_messages_selected,
    message_rate,
    max_targets,
    n_max_messages,
    rng,
):
    message_selector = MessageSelector(
        methods=['select_directly'],
        similarity_method=similarity_method,
        opinion_similarity_threshold=opinion_similarity_threshold,
        vector_similarity_threshold=vector_similarity_threshold,
        max_messages_selected=max_messages_selected,
        rng=rng,
    )

    message_producer = MessageProducer(
        methods=['opinionated', 'similar'],
        similarity_method=similarity_method,
        message_rate=message_rate / 2,
        max_targets=max_targets,
        n_max_messages=n_max_messages,
        rng=rng,
        include_opinions=True,
        include_dyn_vecs=True,
        include_pers_vecs=True,
    )

    lurker = VectorAgent(
        model=model,
        n_dims=n_dims,
        n_dyn_vecs=n_dyn_vecs,
        n_pers_vecs=n_pers_vecs,
        message_selector=message_selector,
        message_producer=message_producer,
        similarity_method=similarity_method,
        agent_type='Commenter',
        rng=rng
    )
    return lurker
