from dataclasses import dataclass, field
from typing import List, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, Optional
import numpy as np

from vector_model import Message, VectorModel

DEBUG_LEVELS = Literal['none', 'summary', 'agent_stats', 'detailed']

@dataclass
class AgentData:
    step: int
    agent_id: int
    agent_type: str
    messages_received: List[Message] = field(default_factory=list)
    messages_sent: List[Message] = field(default_factory=list)
    opinions_before: List[tuple[int, float]] = field(default_factory=list)
    opinions_after: List[tuple[int, float]] = field(default_factory=list)
    dyn_vecs_before: List[np.ndarray] = field(default_factory=list)
    dyn_vecs_after: List[np.ndarray] = field(default_factory=list)


class DataCollector:
    DEBUG_LEVEL_DICT = {
        'none': 0,
        'summary': 1,
        'agent_stats': 2,
        'detailed': 3,
    }
    
    def __init__(self, 
                 debug_level: DEBUG_LEVELS = 'none',
                 n_agents_to_track: int = 5,
                 n_messages_to_show: int = 5):
        self.debug_level = self.DEBUG_LEVEL_DICT[debug_level]
        self.n_agents_to_track = n_agents_to_track
        self.n_messages_to_show = n_messages_to_show
        self.step_data: List[List[AgentData]] = []
        
    def start_step(self, step: int):
        if self.debug_level >= self.DEBUG_LEVEL_DICT['summary']:
            print(f"\n{'='*80}")
            print(f"STEP {step}")
            print(f"{'='*80}")
        self.current_step = step
        self.current_step_data = []
        
    def record_agent(self, 
                    agent_id: int,
                    agent_type: str,
                    messages_received: List["Message"],
                    messages_sent: List["Message"],
                    opinions_before: List[tuple[int, float]],
                    opinions_after: List[tuple[int, float]],
                    dyn_vecs_before: List[np.ndarray],
                    dyn_vecs_after: List[np.ndarray]):
        data = AgentData(
            step=self.current_step,
            agent_id=agent_id,
            agent_type=agent_type,
            messages_received=messages_received[:],
            messages_sent=messages_sent[:],
            opinions_before=opinions_before[:],
            opinions_after=opinions_after[:],
            dyn_vecs_before=[v.copy() for v in dyn_vecs_before],
            dyn_vecs_after=[v.copy() for v in dyn_vecs_after]
        )
        self.current_step_data.append(data)
        
    def end_step(self, model: "VectorModel"):
        self.step_data.append(self.current_step_data)
        
        if self.debug_level >= self.DEBUG_LEVEL_DICT['summary']:
            self._print_step_summary(model)
            
        if self.debug_level >= self.DEBUG_LEVEL_DICT['agent_stats']:
            self._print_agent_stats()
            
        if self.debug_level >= self.DEBUG_LEVEL_DICT['detailed']:
            self._print_detailed_info()
    
    def _print_step_summary(self, model):
        """Print high-level summary of the step"""
        total_messages = len(model.message_space)
        avg_messages_received = np.mean([len(d.messages_received) for d in self.current_step_data])
        avg_messages_sent = np.mean([len(d.messages_sent) for d in self.current_step_data])
        
        print(f"\nStep Summary:")
        print(f"  Total messages in space: {total_messages}")
        print(f"  Avg messages received per agent: {avg_messages_received:.2f}")
        print(f"  Avg messages sent per agent: {avg_messages_sent:.2f}")
    
    def _print_agent_stats(self):
        """Print statistics for tracked agents"""
        print(f"\nAgent Statistics (showing first {self.n_agents_to_track} agents):")
        
        for data in self.current_step_data[:self.n_agents_to_track]:
            print(f"\n  Agent {data.agent_id} (type: {data.agent_type}):")
            print(f"    Messages received: {len(data.messages_received)}")
            print(f"    Messages sent: {len(data.messages_sent)}")
            
            if data.opinions_before and data.opinions_after:
                print(f"    Opinion changes:")
                for (tid_before, op_before), (tid_after, op_after) in zip(
                    data.opinions_before[:3], data.opinions_after[:3]
                ):
                    change = op_after - op_before
                    print(f"      Topic {tid_before}: {op_before:+.3f} → {op_after:+.3f} (Δ{change:+.3f})")
    
    def _print_detailed_info(self):
        print(f"\nDetailed Information (first {self.n_agents_to_track} agents):")
        
        for data in self.current_step_data[:self.n_agents_to_track]:
            print(f"\n  {'─'*76}")
            print(f"  Agent {data.agent_id} (type: {data.agent_type})")
            print(f"  {'─'*76}")
            
            print(f"\n  Messages Received ({len(data.messages_received)} total, showing first {self.n_messages_to_show}):")
            for i, msg in enumerate(data.messages_received[:self.n_messages_to_show]):
                print(f"    [{i}] From Agent {msg.sender_id}:")
                if msg.opinions:
                    print(f"        Opinions: {[(tid, f'{op:+.3f}') for tid, op in msg.opinions[:3]]}")
                if msg.dyn_vecs:
                    print(f"        Dyn vecs: {len(msg.dyn_vecs)} vectors")
                if msg.pers_vecs:
                    print(f"        Pers vecs: {len(msg.pers_vecs)} vectors")
                if msg.target_ids:
                    print(f"        Targets: {msg.target_ids}")
            
            print(f"\n  Messages Sent ({len(data.messages_sent)} total):")
            for i, msg in enumerate(data.messages_sent[:self.n_messages_to_show]):
                print(f"    [{i}]:")
                if msg.opinions:
                    print(f"        Opinions: {[(tid, f'{op:+.3f}') for tid, op in msg.opinions[:3]]}")
                if msg.target_ids:
                    print(f"        Targets: {msg.target_ids}")
            
            print(f"\n  Dynamic Vector Changes:")
            for i, (vec_before, vec_after) in enumerate(zip(data.dyn_vecs_before, data.dyn_vecs_after)):
                if not np.allclose(vec_before, vec_after):
                    change_magnitude = np.linalg.norm(vec_after - vec_before)
                    print(f"    Vec {i}: change magnitude = {change_magnitude:.4f}")
                    print(f"      Before: {vec_before[:3]}")
                    print(f"      After:  {vec_after[:3]}")           

def plot_opinion_trajectories(
    data_collector: DataCollector,
    topic_id: int = 0,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    alpha: float = 0.7,
    linewidth: float = 1.5,
):
    if not data_collector.step_data:
        print("No data collected!")
        return
    
    agent_trajectories: Dict[int, Dict] = {}
    
    for step_idx, step_data in enumerate(data_collector.step_data):
        for agent_data in step_data:
            agent_id = agent_data.agent_id
            
            if agent_id not in agent_trajectories:
                agent_trajectories[agent_id] = {
                    'type': agent_data.agent_type,
                    'opinions': []
                }
            
            opinion_value = None
            for tid, op in agent_data.opinions_after:
                if tid == topic_id:
                    opinion_value = op
                    break
            
            agent_trajectories[agent_id]['opinions'].append(opinion_value)
    
    agent_types = sorted(set(data['type'] for data in agent_trajectories.values()))
    color_map = plt.cm.get_cmap('Set1', len(agent_types))
    type_to_color = {agent_type: color_map(i) for i, agent_type in enumerate(agent_types)}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for agent_id, data in agent_trajectories.items():
        opinions = data['opinions']
        agent_type = data['type']
        color = type_to_color[agent_type]
        
        if any(op is not None for op in opinions):
            steps = list(range(len(opinions)))
            ax.plot(steps, opinions, 
                   color=color, 
                   alpha=alpha, 
                   linewidth=linewidth,
                   label=f'Agent {agent_id} ({agent_type})' if len(agent_trajectories) <= 20 else None)
    
    if len(agent_trajectories) > 20:
        legend_elements = [Line2D([0], [0], color=type_to_color[agent_type], 
                                 linewidth=linewidth*1.5, label=agent_type)
                          for agent_type in agent_types]
        ax.legend(handles=legend_elements, loc='best', title='Agent Type')
    else:
        ax.legend(loc='best', fontsize=8)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f'Opinion on Topic {topic_id}', fontsize=12)
    ax.set_title(f'Agent Opinion Trajectories - Topic {topic_id}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_all_topics(
    data_collector: DataCollector,
    n_topics: int,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    alpha: float = 0.6,
    linewidth: float = 1.0,
):
    if not data_collector.step_data:
        print("No data collected!")
        return
    
    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    agent_data_dict: Dict[int, Dict] = {}
    
    for step_idx, step_data in enumerate(data_collector.step_data):
        for agent_data in step_data:
            agent_id = agent_data.agent_id
            
            if agent_id not in agent_data_dict:
                agent_data_dict[agent_id] = {
                    'type': agent_data.agent_type,
                    'opinions_by_topic': {i: [] for i in range(n_topics)}
                }
            
            for tid, op in agent_data.opinions_after:
                if tid < n_topics:
                    agent_data_dict[agent_id]['opinions_by_topic'][tid].append(op)
    
    agent_types = sorted(set(data['type'] for data in agent_data_dict.values()))
    color_map = plt.cm.get_cmap('tab10', len(agent_types))
    type_to_color = {agent_type: color_map(i) for i, agent_type in enumerate(agent_types)}
    
    for topic_id in range(n_topics):
        row = topic_id // n_cols
        col = topic_id % n_cols
        ax = axes[row, col]
        
        for agent_id, data in agent_data_dict.items():
            opinions = data['opinions_by_topic'][topic_id]
            agent_type = data['type']
            color = type_to_color[agent_type]
            
            if opinions:
                steps = list(range(len(opinions)))
                ax.plot(steps, opinions, 
                       color=color, 
                       alpha=alpha, 
                       linewidth=linewidth)
        
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Opinion', fontsize=10)
        ax.set_title(f'Topic {topic_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for i in range(n_topics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    legend_elements = [Line2D([0], [0], color=type_to_color[agent_type], 
                             linewidth=linewidth*2, label=agent_type)
                      for agent_type in agent_types]
    fig.legend(handles=legend_elements, loc='upper right', title='Agent Type', fontsize=10)
    
    plt.suptitle('Agent Opinion Trajectories - All Topics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes

def calculate_bipolarization(model, topic_id=0):
    topic = model.topic_space.get_topic(topic_id)
    opinions = [agent.calculate_opinion(topic) for agent in model.agents]
    n = len(opinions)
    
    if n == 0:
        return 0.0
    
    d_bar = 0
    for i in range(n):
        for j in range(n):
            d_bar += abs(opinions[i] - opinions[j])
    d_bar /= (n * n)
    
    P_t = 0
    for i in range(n):
        for j in range(n):
            diff = abs(opinions[i] - opinions[j])
            P_t += (diff - d_bar) ** 2
    
    P_t = 4 * P_t / (n * n)
    return P_t

def calculate_opinion_entropy(model, topic_id, num_bins=20):
    topic = model.topic_space.get_topic(topic_id)
    
    opinions = []
    for agent in model.agents:
        opinion = agent.calculate_opinion(topic)
        opinions.append(opinion)
    
    opinions = np.array(opinions)
    
    counts, _ = np.histogram(opinions, bins=num_bins, range=(0, 1))
    
    probabilities = counts[counts > 0] / len(opinions)
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def plot_bipolarization_entropy_heatmaps(
    bipolarization_matrix,
    entropy_matrix,
    param1_range,
    param2_range,
    param1_name,
    param2_name
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_rows, n_cols = bipolarization_matrix.shape
    
    param1_labels = np.linspace(param1_range[0], param1_range[1], n_cols)
    param2_labels = np.linspace(param2_range[0], param2_range[1], n_rows)
    
    im1 = axes[0].imshow(
        bipolarization_matrix,
        cmap='viridis',
        aspect='auto',
        origin='lower',
        extent=[param1_range[0], param1_range[1], param2_range[0], param2_range[1]]
    )
    axes[0].set_xlabel(param1_name, fontsize=12)
    axes[0].set_ylabel(param2_name, fontsize=12)
    axes[0].set_title('Bipolarization', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Bipolarization', fontsize=11)
    
    im2 = axes[1].imshow(
        entropy_matrix,
        cmap='viridis',
        aspect='auto',
        origin='lower',
        extent=[param1_range[0], param1_range[1], param2_range[0], param2_range[1]]
    )
    axes[1].set_xlabel(param1_name, fontsize=12)
    axes[1].set_ylabel(param2_name, fontsize=12)
    axes[1].set_title('Shannon Entropy', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Shannon Entropy', fontsize=11)
    
    plt.tight_layout()
    
    return fig, axes