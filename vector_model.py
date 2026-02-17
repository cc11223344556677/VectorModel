from __future__ import annotations
import sys
import mesa
from mesa.agent import AgentSet
import numpy as np
from typing import Callable, Dict, Tuple, List, Literal
from collections import deque
from typing import List, Tuple, get_args
from analysis import DataCollector

SIMILARITY_METHODS = Literal['cosine', 'tanh']

#scaled to 0 to 1
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    return (np.dot(vec1, vec2) + 1) / 2

#scaled to 0 to 1
def dimensionality_adjusted_similarity(vec1: np.ndarray, vec2: np.ndarray, ) -> float:
    raw = np.dot(vec1, vec2)
    expected_std = 1.0 / np.sqrt(vec1.shape[0])
    return (np.tanh(raw / expected_std) + 1) / 2

class Topic:
    def __init__(self, topic_id: int, vector: np.ndarray, is_static: bool = False, activation: float | np.ndarray = 1.0):
        self.topic_id = topic_id
        self.vector = vector
        self.is_static = is_static
        self.age = 0
        self.activation = activation
        
    def age_topic(self, decay_rate: float = 0.95):
        if not self.is_static:
            self.age += 1
            self.activation *= decay_rate

class TopicSpace:
    def __init__(self, 
                 n_dims: int,
                 n_static_topics: int,
                 n_dynamic_topics: int,
                 decay_rate: float = 0.95,
                 replacement_threshold: float = 0.1,
                 rng: np.random.Generator | None = None
                 ):
        self.n_dims = n_dims
        self.n_static_topics = n_static_topics
        self.n_dynamic_topics = n_dynamic_topics
        self.decay_rate = decay_rate
        self.replacement_threshold = replacement_threshold
        self.rng = rng or np.random.default_rng()
        
        self.topics: List[Topic] = []
        
        for i in range(n_static_topics):
            vec = self.rng.standard_normal(n_dims)
            vec = vec / np.linalg.norm(vec)
            self.topics.append(Topic(i, vec, is_static=True))
        
        for i in range(n_static_topics, n_static_topics + n_dynamic_topics):
            vec = self.rng.standard_normal(n_dims)
            vec = vec / np.linalg.norm(vec)
            initial_activation = self.rng.uniform(0, 1, 1)
            self.topics.append(Topic(i, vec, is_static=False, activation=initial_activation))
    
    def step(self):
        for topic in self.topics:
            if not topic.is_static:
                topic.age_topic(self.decay_rate)
                
                # replace topic if activation drops below threshold
                if topic.activation < self.replacement_threshold:
                    self.replace_topic(topic.topic_id)
    
    def replace_topic(self, topic_id: int):
        new_vector = self.rng.standard_normal(self.n_dims)
        new_vector = new_vector / np.linalg.norm(new_vector)
        initial_activation = self.rng.uniform(0, 1, 1)
        self.topics[topic_id] = Topic(topic_id, new_vector, is_static=False, activation=initial_activation)
    
    def get_topic(self, topic_id: int) -> Topic:
        return self.topics[topic_id]
    
    def get_all_topics(self) -> List[Topic]:
        return self.topics

class Message():
    """
    Class to represent an abstract message, which could be a direct message, social
    media post, news article, etc. Can be configured to contain any combination of the following:
    - a subset of scalar opinion values indexed by topic id
    - a subset of dynamic vectors indexed by dynamic vector id
    - a subset of personality vectors indexed by personality vector id
    - a list of target agents
    Messages are selected for an agent through a MessageSelector object
    """
    def __init__(self, 
                 sender_id: int, 
                 opinions: List[Tuple[int, float]] | None = None, 
                 dyn_vecs: List[Tuple[int, np.ndarray]] | None = None,
                 pers_vecs: List[Tuple[int, np.ndarray]] | None = None, 
                 target_ids: List[int] | None = None,
                 ):
        self.sender_id = sender_id
        self.opinions = opinions
        self.dyn_vecs = dyn_vecs
        self.pers_vecs = pers_vecs
        self.target_ids = target_ids

#changed to multiple methods for flexibility
#added select_none method for strategic agents
class MessageSelector:
    ALLOWED_METHODS = Literal[
    'select_none',
    'select_all',
    'select_randomly',
    'select_directly',
    'select_from_similar_opinions',
    'select_from_similar_personalities_matched',
    'select_from_similar_personalities_unmatched',
    'select_from_similar_dyn_vecs_matched',
    'select_from_similar_dyn_vecs_unmatched',
    ]
    
    def __init__(
        self,
        methods: List[ALLOWED_METHODS],
        similarity_method: SIMILARITY_METHODS, 
        max_messages_selected: int | None = None,
        rng: np.random.Generator | None = None,
        opinion_similarity_threshold: float | None =None,
        vector_similarity_threshold: float | None =None,
    ):
        if max_messages_selected is not None and rng is None:
            raise ValueError("max_messages was set but rng=None was passed")

        invalid = [m for m in methods if m not in get_args(self.ALLOWED_METHODS)]
        if invalid:
            raise ValueError(f"Invalid methods passed: {invalid}")

        self.selectors: List[Callable[[List[Message], VectorAgent], List[Message]]] = []
        
        match similarity_method:
            case 'cosine':
                self.calculate_similarity = cosine_similarity
            case 'tanh':
                self.calculate_similarity = dimensionality_adjusted_similarity

        for m in methods:
            match m:
                case 'select_none':
                    self.selectors.append(self.select_none)
                
                case 'select_all':
                    self.selectors.append(self.select_all)

                case 'select_randomly':
                    if max_messages_selected is None:
                        raise ValueError("method='select_randomly' requires max_messages")
                    self.selectors.append(self.select_randomly)

                case 'select_directly':
                    self.selectors.append(self.select_directly)

                case 'select_from_similar_opinions':
                    if opinion_similarity_threshold is None:
                        raise ValueError(
                            "method='select_from_similar_opinions' "
                            "requires opinion_similarity_threshold"
                        )
                    self.opinion_similarity_threshold: float = opinion_similarity_threshold
                    self.selectors.append(self.select_from_similar_opinions)

                case 'select_from_similar_personalities_matched':
                    if vector_similarity_threshold is None:
                        raise ValueError(
                            f"method='{m}' requires vector_similarity_threshold"
                        )
                    self.vector_similarity_threshold: float = vector_similarity_threshold
                    self.selectors.append(self.select_from_similar_personalities_matched)
                    
                case 'select_from_similar_personalities_unmatched':
                    if vector_similarity_threshold is None:
                        raise ValueError(
                            f"method='{m}' requires vector_similarity_threshold"
                        )
                    self.vector_similarity_threshold: float = vector_similarity_threshold
                    self.selectors.append(self.select_from_similar_personalities_unmatched)

                case 'select_from_similar_dyn_vecs_matched':
                    if vector_similarity_threshold is None:
                        raise ValueError(
                            f"method='{m}' requires vector_similarity_threshold"
                        )
                    self.vector_similarity_threshold: float = vector_similarity_threshold
                    self.selectors.append(self.select_from_similar_dyn_vecs_matched)
                    
                case 'select_from_similar_dyn_vecs_unmatched':
                    if vector_similarity_threshold is None:
                        raise ValueError(
                            f"method='{m}' requires vector_similarity_threshold"
                        )
                    self.vector_similarity_threshold: float = vector_similarity_threshold
                    self.selectors.append(self.select_from_similar_dyn_vecs_unmatched)

        self.max_messages: int | None = max_messages_selected
        self.rng: np.random.Generator = rng or np.random.default_rng()

    def select_messages(
        self,
        message_space: List[Message],
        agent: VectorAgent,
    ) -> List[Message]:
        messages: List[Message] = []

        for selector in self.selectors:
            messages.extend(selector(message_space, agent))

        if self.max_messages is not None and len(messages) > self.max_messages:
            return list(self.rng.choice(np.asarray(messages), self.max_messages, replace=False))

        return messages
    
    def select_none(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        return []
        
    def select_all(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        return message_space
    
    def select_randomly(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        assert self.max_messages #handled by init logic
        return list(self.rng.choice(np.asarray(message_space), min(self.max_messages, len(message_space)) , replace=False))
    
    def select_directly(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        for message in message_space:
            if message.target_ids and agent.unique_id in message.target_ids:
                messages.append(message)
                
        return messages
    
    def select_from_similar_opinions(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for message in message_space:
            if not message.opinions:
                continue
            for topic_id, opinion in message.opinions:
                if abs(agent.calculate_opinion(agent.model.topic_space.get_topic(topic_id)) - opinion) < self.opinion_similarity_threshold:
                    messages.append(message)
                    break
        
        return messages
                
    def select_from_similar_personalities_matched(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for message in message_space:
            if not message.pers_vecs: 
                continue 
            for vec_id, pers_vec in message.pers_vecs:
                if 1 - self.calculate_similarity(agent.pers_vecs[vec_id], pers_vec) < self.vector_similarity_threshold:
                    messages.append(message)
                    break
                
        return messages
        
        
    def select_from_similar_personalities_unmatched(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for message in message_space:
            if not message.pers_vecs:
                continue
            if any(
                1 - self.calculate_similarity(agent_pers_vec, pers_vec) < self.vector_similarity_threshold
                for _, pers_vec in message.pers_vecs
                for agent_pers_vec in agent.pers_vecs
            ):
                messages.append(message)
        
        return messages
    
    def select_from_similar_dyn_vecs_matched(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for message in message_space:
            if not message.dyn_vecs: 
                continue 
            for vec_id, dyn_vec in message.dyn_vecs:
                if 1 - self.calculate_similarity(agent.dyn_vecs[vec_id], dyn_vec) < self.vector_similarity_threshold:
                    messages.append(message)
                    break
                
        return messages
        
        
    def select_from_similar_dyn_vecs_unmatched(self, message_space: List[Message], agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for message in message_space:
            if not message.dyn_vecs:
                continue
            if any(
                1 - self.calculate_similarity(agent_dyn_vec, dyn_vec) < self.vector_similarity_threshold
                for _, dyn_vec in message.dyn_vecs
                for agent_dyn_vec in agent.dyn_vecs
            ):
                messages.append(message)
        
        return messages

#changed to always use both opinions and vectors depending on availability
#as such, changed set_method to set_methods with separate methods for vectors and opinions
#decided to normalize vectors at each step and rely only on angle, preventing explosions 
#or vanishing norms was unworkable and unnecessary, removed similarity type because in this case
#both are the same
#added repulsive influence
#separated methods into assimilative and repulsive. From a cognitive dissonance viewpoint it would
#make the most sense to use closest for assimilative and furthest for repulsive
#changed distance calculations from -1 to 1 to 0 to 1
#added dimensionality adjusted similarity metric using tanh
#moved out of class to share methods between opinion updater and message selector
class OpinionUpdater():
    OPINION_METHODS = Literal['closest', 'furthest']
    VECTOR_METHODS = Literal['closest', 'furthest', 'matched']

    def __init__(self,
                 n_dims: int,
                 n_vecs: int,
                 opinion_assimilative_method: OPINION_METHODS,
                 vector_assimilative_method: VECTOR_METHODS,
                 opinion_repulsive_method: OPINION_METHODS,
                 vector_repulsive_method: VECTOR_METHODS,
                 similarity_method: SIMILARITY_METHODS,
                 epsilon_T_op: float,
                 epsilon_R_op: float,
                 epsilon_T_vec: float,
                 epsilon_R_vec: float,
                 lambda_param: float,
                 ):
        self.n_dims = n_dims
        self.n_vecs = n_vecs
        self.opinion_assimilative_method: OpinionUpdater.OPINION_METHODS  = opinion_assimilative_method
        self.vector_assimilative_method: OpinionUpdater.VECTOR_METHODS = vector_assimilative_method
        self.opinion_repulsive_method: OpinionUpdater.OPINION_METHODS = opinion_repulsive_method
        self.vector_repulsive_method: OpinionUpdater.VECTOR_METHODS = vector_repulsive_method
        match similarity_method:
            case 'cosine':
                self.calculate_similarity = cosine_similarity
            case 'tanh':
                self.calculate_similarity = dimensionality_adjusted_similarity
        self.epsilon_T_op = epsilon_T_op
        self.epsilon_R_op = epsilon_R_op
        self.epsilon_T_vec = epsilon_T_vec
        self.epsilon_R_vec = epsilon_R_vec
        self.lambda_param = lambda_param

    def lambda_adjustment(self, messages: List[Message]):
        n_i = len(messages)
        return 0 if n_i == 0 else self.lambda_param / n_i

    def update(self, agent: VectorAgent, messages: List[Message]):
        opinion_updates = self.opinion_update(agent, messages)
        vector_updates  = self.vector_update(agent, messages)

        lam = self.lambda_adjustment(messages)

        for i in range(self.n_vecs):
            dyn_vec_update = opinion_updates[i] + vector_updates[i]

            new_unnormed = agent.dyn_vecs[i] + (lam * dyn_vec_update)
            norm = np.linalg.norm(new_unnormed)
            agent.next_dyn_vecs[i] = new_unnormed / norm if norm > 0 else agent.dyn_vecs[i]

    def _find_extreme_vec_index(self,
                                vecs: List[np.ndarray],
                                reference: np.ndarray,
                                method: OPINION_METHODS,
                                ) -> Tuple[int | None, float]:
        if method == 'closest':
            best_similarity = -sys.float_info.max
            is_better: Callable[[float, float], bool] = lambda similarity, best: similarity > best
        else:  # furthest
            best_similarity = sys.float_info.max
            is_better: Callable[[float, float], bool] = lambda similarity, best: similarity < best

        best_index = None
        for idx, vec in enumerate(vecs):
            similarity = self.calculate_similarity(vec, reference)
            if is_better(similarity, best_similarity):
                best_index = idx
                best_similarity = similarity

        return best_index, best_similarity

    def opinion_update(self,
                       agent: VectorAgent,
                       messages: List[Message],
                       ) -> List[np.ndarray]:
        update = [np.zeros(self.n_dims) for _ in range(self.n_vecs)]

        for message in messages:
            if not message.opinions:
                continue
            for topic_id, opinion in message.opinions:
                topic = agent.model.topic_space.get_topic(topic_id)
                opinion_diff = np.abs(opinion - agent.calculate_opinion(topic))
                # Due to changing from [-1,1] to [0,1] had to remap this part
                signed = opinion * 2 - 1
                target = topic.vector * signed

                if opinion_diff < self.epsilon_T_op:
                    best_index, _ = self._find_extreme_vec_index(agent.dyn_vecs, topic.vector, self.opinion_assimilative_method)
                    if best_index is not None:
                        update[best_index] += (target - agent.dyn_vecs[best_index]) * topic.activation

                elif opinion_diff > self.epsilon_R_op:
                    best_index, _ = self._find_extreme_vec_index(agent.dyn_vecs, topic.vector, self.opinion_repulsive_method)
                    if best_index is not None:
                        update[best_index] -= (target - agent.dyn_vecs[best_index]) * topic.activation

        return update

    def vector_update(self,
                      agent: VectorAgent,
                      messages: List[Message],
                      ) -> List[np.ndarray]:
        update = [np.zeros(self.n_dims) for _ in range(self.n_vecs)]

        for message in messages:
            if not message.dyn_vecs:
                continue

            #TODO: extract into separate function, this is gross
            for vec_id, dyn_vec in message.dyn_vecs:
                if self.vector_assimilative_method == 'matched':
                    similarity = self.calculate_similarity(dyn_vec, agent.dyn_vecs[vec_id])
                    if 1 - similarity < self.epsilon_T_vec:
                        update[vec_id] += dyn_vec - agent.dyn_vecs[vec_id]
                else:
                    best_index, best_similarity = self._find_extreme_vec_index(agent.dyn_vecs, dyn_vec, self.vector_assimilative_method)
                    if best_index is not None and 1 - best_similarity < self.epsilon_T_vec:
                        update[best_index] += dyn_vec - agent.dyn_vecs[best_index]
                        
                if self.vector_repulsive_method == 'matched':
                    similarity = self.calculate_similarity(dyn_vec, agent.dyn_vecs[vec_id])
                    if similarity > self.epsilon_R_vec:
                        update[vec_id] -= dyn_vec - agent.dyn_vecs[vec_id]
                else:
                    best_index, best_similarity = self._find_extreme_vec_index(agent.dyn_vecs, dyn_vec, self.vector_repulsive_method)
                    if best_index is not None and best_similarity > self.epsilon_R_vec:
                        update[best_index] -= dyn_vec - agent.dyn_vecs[best_index]
                        
                



        return update

#changed agent logic to that this is a component of an agent
#added multiple method flexibility like selector
#changed message history name and access patterns
#eliminated redundant methods
class MessageProducer:
    PRODUCER_METHODS = Literal['reciprocal', 'similar', 
                                'dissimilar', 'opinionated', 'random_targets']
    
    def __init__(self,
                 methods: List[PRODUCER_METHODS],
                 similarity_method: SIMILARITY_METHODS,
                 message_rate: float = 1.0,
                 reciprocal_boost: float = 2.0,
                 similarity_threshold: float = 0.5,
                 opinion_threshold: float = 0.7,
                 max_targets: int = 1,
                 n_max_messages: int = 5,
                 rng: np.random.Generator | None = None,
                 include_opinions: bool = True,
                 include_dyn_vecs: bool = True,
                 include_pers_vecs: bool = True,
                 n_topics_per_message: int = 100000, #default to max
                 n_dyn_vecs_per_message: int = 100000,
                 n_pers_vecs_per_message: int = 100000,
                 ):
        self.set_methods(methods)
        match similarity_method:
            case 'cosine':
                self.calculate_similarity = cosine_similarity
            case 'tanh':
                self.calculate_similarity = dimensionality_adjusted_similarity
        self.message_rate = message_rate
        self.reciprocal_boost = reciprocal_boost
        self.similarity_threshold = similarity_threshold
        self.opinion_threshold = opinion_threshold
        self.max_targets = max_targets
        self.n_max_messages = n_max_messages
        self.rng = rng or np.random.default_rng()
        self.include_opinions = include_opinions
        self.include_dyn_vecs = include_dyn_vecs
        self.include_pers_vecs = include_pers_vecs
        self.n_topics_per_message = n_topics_per_message
        self.n_dyn_vecs_per_message = n_dyn_vecs_per_message
        self.n_pers_vecs_per_message = n_pers_vecs_per_message
        
    def set_methods(self, methods: List[PRODUCER_METHODS]):
        self.methods: List[Callable[[VectorAgent], List[Message]]] = []
        
        for method in methods:
            match method:
                case 'reciprocal':
                    self.methods.append(self.reciprocal)
                case 'similar':
                    self.methods.append(self.similar)
                case 'dissimilar':
                    self.methods.append(self.dissimilar)
                case 'opinionated':
                    self.methods.append(self.opinionated)
                case 'random_targets':
                    self.methods.append(self.random_targets)
                
    def produce(self, agent: VectorAgent) -> List[Message]:
        messages: List[Message] = []
        
        for method in self.methods:
            messages.extend(method(agent))
        
        if len(messages) > self.n_max_messages:
            self.rng.shuffle(messages)
            messages = messages[:self.n_max_messages]
        
        return messages
    
    def _create_message(self, agent: VectorAgent, target_ids: List[int] | None = None) -> Message:
        opinions = None
        dyn_vecs = None
        pers_vecs = None
        
        if self.include_opinions:
            all_topics = list(range(len(agent.model.topic_space.topics)))
            if self.n_topics_per_message:
                topic_ids = self.rng.choice(all_topics, 
                                          min(self.n_topics_per_message, len(all_topics)), 
                                          replace=False)
            else:
                topic_ids = all_topics
                
            opinions = [(tid, agent.calculate_opinion(agent.model.topic_space.get_topic(tid))) 
                       for tid in topic_ids]
        
        if self.include_dyn_vecs:
            all_dyn_vecs = list(range(len(agent.dyn_vecs)))
            if self.n_dyn_vecs_per_message:
                vec_ids = self.rng.choice(all_dyn_vecs,
                                         min(self.n_dyn_vecs_per_message, len(all_dyn_vecs)),
                                         replace=False)
            else:
                vec_ids = all_dyn_vecs
            dyn_vecs = [(i, agent.dyn_vecs[i].copy()) for i in vec_ids]
            
        if self.include_pers_vecs:
            all_pers_vecs = list(range(len(agent.pers_vecs)))
            if self.n_pers_vecs_per_message:
                vec_ids = self.rng.choice(all_pers_vecs,
                                         min(self.n_pers_vecs_per_message, len(all_pers_vecs)),
                                         replace=False)
            else:
                vec_ids = all_pers_vecs
            pers_vecs = [(i, agent.pers_vecs[i].copy()) for i in vec_ids]
        
        return Message(
            sender_id=agent.unique_id,
            opinions=opinions,
            dyn_vecs=dyn_vecs,
            pers_vecs=pers_vecs,
            target_ids=target_ids
        )
    
    def reciprocal(self, agent: VectorAgent) -> List[Message]:
        recent_messages = agent.message_history[-1] if agent.message_history else []
        sender_counts: Dict[int, int] = {}
        for msg in recent_messages:
            sender_counts[msg.sender_id] = sender_counts.get(msg.sender_id, 0) + 1
        
        if not sender_counts:
            return []
        
        total = sum(sender_counts.values())
        boosted_rate = min(1.0, self.message_rate * (1 + self.reciprocal_boost * total / max(len(agent.message_history[-1]), 1)))
        
        if self.rng.random() < boosted_rate:
            weights = np.array([sender_counts.get(sid, 0) for sid in sender_counts.keys()])
            weights = weights / weights.sum()
            
            n_to_target = min(self.max_targets // len(self.methods), len(sender_counts))
            target_ids = self.rng.choice(list(sender_counts.keys()), 
                                        size=n_to_target, 
                                        replace=False, 
                                        p=weights)
            return [self._create_message(agent, list(target_ids))]
        
        return []
    
    def similar(self, agent: VectorAgent) -> List[Message]:
        recent_messages = agent.message_history[-1] if agent.message_history else []
        similar_agents: List[int] = []
        
        for msg in recent_messages:
            if msg.pers_vecs:
                for vec_id, pers_vec in msg.pers_vecs:
                    if vec_id < len(agent.pers_vecs):
                        similarity = self.calculate_similarity(agent.pers_vecs[vec_id], pers_vec)
                        if similarity > self.similarity_threshold:
                            similar_agents.append(msg.sender_id)
                            break
        
        if similar_agents:
            n_to_target = min(self.max_targets // len(self.methods), len(similar_agents))
            target_ids = self.rng.choice(similar_agents, size=n_to_target, replace=False)
            return [self._create_message(agent, list(target_ids))]
        
        return []
    
    def dissimilar(self, agent: VectorAgent) -> List[Message]:
        recent_messages = agent.message_history[-1] if agent.message_history else []
        dissimilar_agents: List[int] = []
        
        for msg in recent_messages:
            if msg.pers_vecs:
                for vec_id, pers_vec in msg.pers_vecs:
                    if vec_id < len(agent.pers_vecs):
                        similarity = self.calculate_similarity(agent.pers_vecs[vec_id], pers_vec)
                        if similarity < self.similarity_threshold:
                            dissimilar_agents.append(msg.sender_id)
                            break
        
        if dissimilar_agents:
            n_to_target = min(self.max_targets // len(self.methods), len(dissimilar_agents))
            target_ids = self.rng.choice(dissimilar_agents, size=n_to_target, replace=False)
            return [self._create_message(agent, list(target_ids))]
        
        return []
    
    def opinionated(self, agent: VectorAgent) -> List[Message]:
        strong_opinion_indices: List[int] = []
        for i, topic in enumerate(agent.model.topic_space.get_all_topics()):
            if abs(agent.calculate_opinion(topic)) > self.opinion_threshold:
                strong_opinion_indices.append(i)

        if not strong_opinion_indices or self.rng.random() >= self.message_rate:
            return []

        all_agents: List[VectorAgent] = list(agent.model.agents)
        all_agents.remove(agent)

        if not all_agents:
            return []

        n_to_target = min(self.max_targets // len(self.methods), len(all_agents))
        targets: List[VectorAgent] = list(self.rng.choice(np.asarray(all_agents), size=n_to_target, replace=False))
        target_ids: List[int] = [t.unique_id for t in targets]

        msg = self._create_message(agent, target_ids)
        if self.include_opinions and msg.opinions:
            msg.opinions = [(tid, val) for tid, val in msg.opinions if tid in strong_opinion_indices]

        return [msg]
    
    def random_targets(self, agent: VectorAgent) -> List[Message]:
        if self.rng.random() < self.message_rate:
            all_agents = list(agent.model.agents)
            all_agents.remove(agent)
            
            if all_agents:
                n_to_target = min(self.max_targets // len(self.methods), len(all_agents))
                targets: List[VectorAgent] = list(self.rng.choice(np.asarray(all_agents), size=n_to_target, replace=False))
                target_ids: List[int] = [t.unique_id for t in targets]
                return [self._create_message(agent, target_ids)]
        
        return []

#complete rewrite
class VectorAgent(mesa.Agent["VectorModel"]):
    def __init__(self,
                 model: VectorModel,
                 n_dims: int,
                 n_dyn_vecs: int,
                 n_pers_vecs: int,
                 message_selector: MessageSelector,
                 message_producer: MessageProducer,
                 similarity_method: SIMILARITY_METHODS= 'cosine',
                 agent_type: str = 'default',
                 rng: np.random.Generator | None = None):
        self.model: VectorModel
        self.unique_id: int
        super().__init__(model) # type: ignore , mesa's fault
        
        self.n_dims = n_dims
        self.n_dyn_vecs = n_dyn_vecs
        self.n_pers_vecs = n_pers_vecs
        self.agent_type = agent_type
        self._rng = rng or np.random.default_rng()
        
        match similarity_method:
            case 'cosine':
                self.calculate_similarity = cosine_similarity
            case 'tanh':
                self.calculate_similarity = dimensionality_adjusted_similarity
        
        self.dyn_vecs: List[np.ndarray] = []
        for _ in range(n_dyn_vecs):
            vec = self._rng.standard_normal(n_dims)
            vec = vec / np.linalg.norm(vec)
            self.dyn_vecs.append(vec)
        
        self.pers_vecs: List[np.ndarray] = []
        for _ in range(n_pers_vecs):
            vec = self._rng.standard_normal(n_dims)
            vec = vec / np.linalg.norm(vec)
            self.pers_vecs.append(vec)
        
        self.next_dyn_vecs = [v.copy() for v in self.dyn_vecs]
        
        self.message_selector: MessageSelector = message_selector
        self.message_producer: MessageProducer = message_producer
        
        self.message_history: deque[List[Message]] = deque(maxlen=5)
        
    def calculate_opinion(self, topic: Topic) -> float:
        avg_vec = np.zeros(self.n_dims)
        for pers_vec in self.pers_vecs:
            avg_vec += pers_vec
        for dyn_vec in self.dyn_vecs:
            avg_vec += dyn_vec
        avg_vec /= (self.n_dyn_vecs + self.n_pers_vecs)
        
        return self.calculate_similarity(avg_vec, topic.vector)
    
    def step(self):
        messages_received = self.message_selector.select_messages(
            self.model.message_space, 
            self
        )
        
        self.message_history.append(messages_received)
        
        opinions_before = []
        dyn_vecs_before = []
        if self.model.data_collector:
            opinions_before = [(i, self.calculate_opinion(topic)) 
                             for i, topic in enumerate(self.model.topic_space.get_all_topics())]
            dyn_vecs_before = [v.copy() for v in self.dyn_vecs]
        
        self.model.opinion_updater.update(self, messages_received)
        
        self.dyn_vecs = [v.copy() for v in self.next_dyn_vecs]
        
        messages_sent = self.message_producer.produce(self)
        
        for msg in messages_sent:
            self.model.message_space.append(msg)
        
        if self.model.data_collector:
            opinions_after = [(i, self.calculate_opinion(topic)) 
                            for i, topic in enumerate(self.model.topic_space.get_all_topics())]
            dyn_vecs_after = [v.copy() for v in self.dyn_vecs]
            
            self.model.data_collector.record_agent(
                agent_id=self.unique_id,
                agent_type=self.agent_type,
                messages_received=messages_received,
                messages_sent=messages_sent,
                opinions_before=opinions_before,
                opinions_after=opinions_after,
                dyn_vecs_before=dyn_vecs_before, 
                dyn_vecs_after=dyn_vecs_after
            )

#complete rewrite, no grid
class VectorModel(mesa.Model["VectorAgent"]):
    def __init__(self,
                 n_agents: int,
                 topic_space: TopicSpace,
                 opinion_updater: OpinionUpdater,
                 n_dims: int,
                 n_dyn_vecs: int,
                 n_pers_vecs: int,
                 similarity_method: str,
                 data_collector: "DataCollector | None" = None):
        self.agents: AgentSet[VectorAgent]
        super().__init__()
        
        self.n_agents = n_agents
        self.n_dims = n_dims
        self.n_dyn_vecs = n_dyn_vecs
        self.n_pers_vecs = n_pers_vecs
        self.similarity_method = similarity_method
        
        self.topic_space: TopicSpace = topic_space
        self.opinion_updater: OpinionUpdater = opinion_updater
        
        self.message_space: List[Message] = []
        
        self.data_collector = data_collector
        
        self.current_step = 0
    
    def step(self):
        if self.data_collector:
            self.data_collector.start_step(self.current_step)
        
        self.topic_space.step()
        self.agents.do("step") #type: ignore , mesa's fault
        
        if self.data_collector:
            self.data_collector.end_step(self)
        
        self.message_space.clear()
        
        self.current_step += 1
    
    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()