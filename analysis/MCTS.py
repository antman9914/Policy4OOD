import math
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import os
from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Any, Dict

from models.Policy4OOD import Policy4OOD
from utils.load_configs import get_link_prediction_args
from train import loadkg, load_data


class SchedulingState:
    """
    State representing a partial or complete policy schedule.
    
    Given:
    - P: set of policy candidates
    - t1~tn: time periods
    - c: cost metric to minimize
    
    This class tracks which policies are assigned to which time periods.
    """
    
    def __init__(self, policies: List[Any], time_periods: List[Any], model: Policy4OOD, time_series: torch.Tensor, graph: torch.Tensor, 
                 schedule: Optional[Dict[Any, Any]] = None, kg_idx_array: np.ndarray = None, future_array: np.ndarray = None):
        """
        Initialize scheduling state.
        """
        self.policies = policies
        self.effective_policy = set(self.policies)
        self.time_periods = time_periods
        self.model = model
        self.model.eval()
        self.time_series = time_series
        self.graph = graph
        self.kg_idx_array = kg_idx_array
        self.future_idx_arr = future_array
        self.padded_schedule = None
        self.schedule = schedule if schedule is not None else []
        self.update_pad_schedule()
        self.current_time_index = len(self.schedule)  # Next time period to assign
    
    def update_pad_schedule(self):
        if self.padded_schedule is None:
            self.padded_schedule = np.zeros(len(self.time_periods))
        else:
            if len(self.schedule) == 1 or self.schedule[-1] != -1:
                self.padded_schedule[len(self.schedule) - 1] = self.schedule[-1]
            else:
                self.padded_schedule[len(self.schedule) - 1] = self.schedule[len(self.schedule)-2]
        
        if len(self.schedule) != 0:
            self.padded_schedule[len(self.schedule):] = self.padded_schedule[len(self.schedule)-1]


    def get_legal_moves(self) -> List[Any]:
        """
        Return list of legal policy assignments for the next time period.
        Each move is a policy that can be assigned to the current time period.
        """
        if self.is_terminal():
            return []
        return list(self.effective_policy) 
    
    def make_move(self, policy: Any) -> 'SchedulingState':
        """
        Return a new state after assigning a policy to the next time period.
        
        Args:
            policy: The policy to assign to the current time period
        """
        new_state = deepcopy(self)
        new_state.schedule.append(policy)
        if policy != -1:
            new_state.effective_policy.remove(policy)
        new_state.update_pad_schedule()
        
        new_state.current_time_index += 1
        return new_state
    
    def is_terminal(self) -> bool:
        """Check if all time periods have been assigned policies."""
        return self.current_time_index >= len(self.time_periods)
    
    def compute_cost(self) -> float:
        with torch.no_grad():
            self.future_idx_arr[state_idx, :] = torch.from_numpy(self.padded_schedule)
            embeds, _ = model[0](self.time_series, graph, kg_list, self.kg_idx_array, self.future_idx_arr)  
            predicts = self.model[1](embeds)
        return predicts[state_idx, :]
    
    def get_reward(self) -> float:
        return -self.compute_cost().sum()
    
    def __str__(self) -> str:
        """String representation of the schedule."""
        if not self.schedule:
            return "Empty schedule"
        
        lines = ["Summary:"]
        if self.is_terminal():
            cost = self.compute_cost().sum() * ts_std[0, 0, 0] + ts_mean[0, 0, 0]
            lines.append(f"\nTotal Cost: {cost:.2f}")
        
        return '\n'.join(lines)


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree for scheduling optimization.
    """
    
    def __init__(self, state: SchedulingState, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.move = move  # Policy assigned that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_legal_moves()
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been created."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.state.is_terminal()
    
    def best_child(self, c_param: float = 1.41) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        For minimization, we use negative values.
        """
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self) -> 'MCTSNode':
        """Expand tree by creating a new child node."""
        move = self.untried_moves.pop()
        next_state = self.state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def update(self, reward: float):
        """Update node statistics."""
        self.visits += 1
        self.value += reward


class MCTS:
    """
    Monte Carlo Tree Search algorithm for scheduling optimization.
    Finds policy assignments that minimize cost metric.
    """
    
    def __init__(self, exploration_param: float = 1.41):
        self.exploration_param = exploration_param
    
    def get_best_schedule_state(self, initial_state: SchedulingState, num_simulations: int = 1000) -> SchedulingState:
        """
        Run MCTS and return the complete best state (with cost computed).
        
        Args:
            initial_state: Starting state (empty schedule)
            num_simulations: Number of simulations to run
            
        Returns:
            Complete SchedulingState with best schedule
        """
        root = MCTSNode(initial_state)
        
        for _ in tqdm(range(num_simulations)):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        # Follow most visited path to get complete state
        current = root
        while current.children:
            best_child = max(current.children, key=lambda c: c.visits)
            current = best_child
        
        return current.state
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child(self.exploration_param)
        return node
    
    def _simulate(self, state: SchedulingState) -> float:
        """
        Simulation phase: randomly complete the schedule.
        Returns reward (negative cost).
        """
        current_state = deepcopy(state)
        
        while not current_state.is_terminal():
            moves = current_state.get_legal_moves()
            move = random.choice(moves)
            current_state = current_state.make_move(move)
        
        return current_state.get_reward()
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update all ancestors."""
        while node is not None:
            node.update(reward)
            node = node.parent


def optimize_schedule(model, input_ts, graph, kg_idx_array, future_array, policies, time_periods):
    
    print("=" * 60)
    print("Policy Scheduling Optimization with MCTS")
    print("=" * 60)
    print(f"\nPolicies available: {policies}")
    print(f"Time periods: {time_periods}")
    print(f"\nObjective: Minimize cost metric\n")
    
    # Create initial empty state
    initial_state = SchedulingState(policies, time_periods, model, input_ts, graph, schedule=None, kg_idx_array=kg_idx_array, future_array=future_array)
    
    # Run MCTS optimization
    mcts = MCTS(exploration_param=1.41)
    
    print("Running MCTS optimization...")
    print("(This may take a few seconds)\n")
    
    # Get the best schedule
    best_state = mcts.get_best_schedule_state(initial_state, num_simulations=1000)
    
    print("OPTIMAL SCHEDULE FOUND:")
    print("=" * 60)
    print(best_state.padded_schedule)
    print(best_state.current_time_index)
    print("=" * 60)
    print(best_state.compute_cost())
    
    # Compare with random schedules
    print("\n\nComparison with random schedules:")
    print("-" * 60)
    
    # Ground truth policy trajectory. Modify the ground truth based on your demand.
    policy_seq = [-1, -1, -1, 0, 0, 0]
    random_state = deepcopy(initial_state)
    for policy in policy_seq:
        random_state.schedule.append(policy)
    random_state.current_time_index = len(time_periods)
    cost = random_state.compute_cost()
    cost_val = cost.sum()
    avg_random_cost = cost_val * ts_std[0, 0, 0] + ts_mean[0, 0, 0]
    
    print(f"\nGround truth cost: {avg_random_cost:.2f}")
    print(f"MCTS optimal cost: {best_state.compute_cost().sum() * ts_std[0, 0, 0] + ts_mean[0, 0, 0]:.2f}")
    
    return best_state.schedule


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=False)
    args.seed = 1
    args.save_model_name = f'{args.model_name}_seed{args.seed}'

    state_dict, state_map = {}, {}
    state_list = pd.read_csv('processed_data/state_ref.csv')
    state_name, state_abbr = state_list['state_name'], state_list['state_abbr']
    idx = 0
    for i in range(len(state_name)):
        if state_abbr[i] != 'HI' and state_abbr[i] != 'AK':
            state_dict[state_name[i]] = state_abbr[i]
            state_map[state_abbr[i]] = idx
            idx += 1
    state_name, state_abbr = list(state_dict.keys()), list(state_map.keys())

    # Data loading
    graph, time_series, ts_mean, ts_std = load_data(return_std=True)
    kg_list, kg_idx_array = loadkg()

    encoder = Policy4OOD(time_series.size(-1), args.hidden_dim, args.output_step)
    pred_head = nn.Sequential(nn.Linear(args.hidden_dim + 16, time_series.size(-1)), nn.ReLU(), nn.Linear(time_series.size(-1), args.output_step))
    model = nn.Sequential(encoder, pred_head)
    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    save_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pkl")
    model.load_state_dict(torch.load(save_model_path, map_location=torch.device('cpu')))


    # Specify the state for policy optimization
    state_idx = state_map['TN']
    # Specify the time span for policy optimization, including historical time window and the target future time span.
    # In our case, we conduct policy optimization in Tennessee from Mar. 2021 to Sep. 2021.
    time_series = time_series[:, 20:26, :]
    kg_idx_array, future_array = torch.from_numpy(kg_idx_array[:, 20:26]), torch.from_numpy(kg_idx_array[:, 26:32])
    # Specify candidate policy documents. The candidates are specified via np.arange(0, n_document)
    # An empty action -1 is needed to represent the case where no policy is enacted.
    n_doc = 6
    candidate_policy = np.arange(n_doc + 1) - 1
    optimize_schedule(model, time_series, graph, kg_idx_array, future_array, candidate_policy, np.arange(6).tolist())  