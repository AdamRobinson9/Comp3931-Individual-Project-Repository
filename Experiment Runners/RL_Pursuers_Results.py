import random
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Strategies import MoveController
from Placement import Placement
from collections import deque
plt.ion()

class RLController:
    """Reinforcement learning controller for evader strategy selection"""
    def __init__(self, actions):
        self.actions = actions # Available Strategies (actions)
        self.epsilon = 0.2 # Exploration Rate

        self.Q = {} # Reward table (context, action) -> expected reward
        self.N = {} # Usage frequency table of each (context, action)

    def select_action(self, context, available_actions=None):
        """Select a strategy using the epsilon greedy policy"""
        if available_actions is None: 
            available_actions = self.actions

        # Exploration: if random number < epsilon, choose random action
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: if random number >= epsilon, use highest Q-value
        best_action = None
        best_value = float('-inf')

        for action in available_actions: # Check all available actions
            key = (context, action) # Key for Q-table: context-action pair
            # Get expected reward from Q-table
            if key in self.Q:
                value = self.Q[key]
            else:
                value = 0.0
            # Choose the best action
            if value > best_value:
                best_value = value
                best_action = action
            # Randomly choose if there is no best
            elif value == best_value and best_action is not None:
                best_action = random.choice([action, best_action])

        return best_action
    
    def update(self, context, action, reward):
        """Update Q-table and count table using observed reward"""
        key = (context,action) # Q-table key
        # If context-action not in table, add it
        if key not in self.Q:
            self.N[key] = 0
            self.Q[key] = 0.0

        self.N[key] += 1 # Increment count table
        self.Q[key] += (reward - self.Q[key]) / self.N[key] # Update expected reward

    def output_q_table(self):
        """Extract the best action from each context from the q table, for csv output"""
        best_actions = {}

        # Get the best action for each context
        for (context, action), q_value in self.Q.items():
            if context not in best_actions or q_value > best_actions[context][1]:
                best_actions[context] = (action, q_value)

        rows = []
        for context, (best_action, best_q) in best_actions.items():
            rows.append({
                "dist_to_flag": context[0],
                "dist_to_teammate": context[1],
                "flag_pressure": context[2],
                "graph_position": context[3],
                "action": best_action,
                "q_value": round(best_q, 4),
                "visit_count": self.N[(context, best_action)],
            })

        return sorted(rows, key=lambda row: (row["dist_to_flag"]))

class Flag:
    """Flag object assigned to one evader, tracking ownership & location."""
    def __init__(self, carrier):
        self.carrier = carrier # The current carrier of the flag

    def position(self):
        """Return the vertex that the flag is currently at"""
        return self.carrier.position

    def transfer(self, new_carrier):
        """Pass the flag from current evader to the new carrier."""
        self.carrier.has_flag = False
        new_carrier.has_flag = True
        self.carrier = new_carrier

class Agent:
    """
    A single acting player in the game
    Pursuers & Evaders share the same structure.
    """
    def __init__(self, start_node, strategy, is_pursuer=False):
        self.position = start_node
        self.strategy = strategy
        self.is_pursuer = is_pursuer
        self.has_flag = False   # Only true for one evader at a time
        self.last_context = None
        self.last_action = None
        self.d_before = None

    def move(self, state):
        """Move to any neighbour."""
        neighbours = list(state.graph.neighbors(self.position))
        if neighbours:
            self.position = random.choice(neighbours)

class GameState:
    """
    Store the current game situation
    Also provides methods to support game management
    """
    def __init__(self, graph, pursuers, evaders, flag_carrier, max_turns):
        self.graph = graph
        self.pursuers = pursuers
        self.evaders = evaders

        # Initialise flag object & owner
        self.flag = Flag(flag_carrier)
        flag_carrier.has_flag = True

        self.max_turns = max_turns
        self.current_turn = 0
        self.current_team = "P"  # P = pursuers, E = evaders

        self.game_over = False
        self.winner = None        
        # Precompute vertex centralities for context & reward calculations
        self.centrality = nx.closeness_centrality(graph)
        self.centrality_median = np.median(list(self.centrality.values()))

    def get_context(self, agent):
        """Build state representation for RL"""
        graph = self.graph

        # Distance to flag
        if (agent.has_flag):
            dist_to_flag = "carrier"
        else:
            flag_distance = nx.shortest_path_length(graph, agent.position, self.flag.position())
            if (flag_distance <= 1):
                dist_to_flag = "near"
            elif (2 <= flag_distance <=3):
                dist_to_flag = "mid"
            else:
                dist_to_flag = "far"

        # Distance to nearest teammate
        if agent.is_pursuer:
            teammates = [teammate for teammate in self.pursuers if teammate != agent]
        else:
            teammates = [teammate for teammate in self.evaders if teammate != agent]

        if teammates:
            teammate_dist = min(
                nx.shortest_path_length(graph, agent.position, teammate.position)
                for teammate in teammates
            )
            if teammate_dist <= 2:
                dist_to_nearest_teammate = "near"  
            else:
                dist_to_nearest_teammate = "far"
        else:
            dist_to_nearest_teammate = "none"

        # Flag Pressure
        flag_neighbourhood = nx.single_source_shortest_path_length(graph, self.flag.position(), cutoff=2)
        pressers = 0
        for pursuer in self.pursuers:
            if pursuer.position in flag_neighbourhood:
                pressers += 1
        if pressers >= 2:
            flag_pressure = "high"
        elif pressers == 1:
            flag_pressure = "med"
        else:
            flag_pressure = "low"
        
        # Graph Position
        if self.centrality[agent.position] >= self.centrality_median:
            graph_position = "central"
        else:
            graph_position = "peripheral"

        # Return context tuple
        context = (dist_to_flag, dist_to_nearest_teammate, flag_pressure, graph_position)
        return context

    def compute_reward(self, agent):
        """Compute reward signal for RL"""
        graph = self.graph

        # Pursuer Reward
        if agent.is_pursuer:
            # Terminal Reward
            if self.game_over:
                if self.winner == "Pursuers":
                    return 10
                else:
                    return -10
            # Intermediate Rewards: 
            reward = 0
            # Distance (to flag) closing reward
            d_after = nx.shortest_path_length(graph, agent.position, self.flag.position())
            reward += agent.d_before - d_after
            # Survival Penalty
            if not self.game_over:
                reward -= 1
            # Clustering penalies
            if agent.position in [pursuer.position for pursuer in self.pursuers if pursuer != agent]:
                reward -=0.5
            for pursuer in self.pursuers:
                if pursuer != agent:
                    if nx.shortest_path_length(graph, agent.position, pursuer.position) == 1:
                        reward -= 0.3
                
            # Flag pressure Bonus
            reward += self.flag_pressure()

        # EVADER REWARD
        else:
            # Terminal Reward
            if self.game_over:
                if self.winner == "Evaders":
                    return 10
                else:
                    return -10
                
            # Intermediate rewards:    
            reward = 0
            # Distance (to target) increase reward
            d_after = min(nx.shortest_path_length(graph, self.flag.position(), pursuer.position) 
                          for pursuer in self.pursuers)
            reward += (d_after - self.flag.d_before) / (d_after+1)
            # Survival Reward
            if not self.game_over:
                reward +=1
            # Graph position rewards    
            reward += graph.degree(self.flag.position()) / max(dict(graph.degree()).values())
            reward += self.centrality[self.flag.position()]
            # Flag Pressure Penalty
            reward -= self.flag_pressure() * (1 + (1 - self.centrality[self.flag.position()]))
            # Team coordination reward
            reachable = self.reachable_teammates()
            chain_size = max(nx.shortest_path_length(self.graph, self.flag.position(), teammate.position) for teammate in reachable)+1
            reward += chain_size/len(self.evaders)

        return reward

    def flag_pressure(self):
        """Compute pressure indicator on the flag"""
        graph = self.graph
        flag_pos = self.flag.position()

        # Nodes within distance 2 of the flag
        neighbourhood = nx.single_source_shortest_path_length(graph, flag_pos, cutoff=2)

        pursuers_near = 0
        evaders_near = 0
        # Count pursuers within distance 2
        for pursuer in self.pursuers:
            if pursuer.position in neighbourhood:
                pursuers_near += 1
        # Count evaders within distance 2
        for evader in self.evaders:
            if evader.position in neighbourhood:
                evaders_near += 1

        # Pressure formula
        pressure = (pursuers_near - evaders_near)/ (pursuers_near + evaders_near + 1)

        return pressure

    def agents_to_act(self):
        """Return list of agents who act in the current phase."""
        if self.current_team == "P":
            return self.pursuers
        else:
            agents_to_act = [self.flag.carrier] # Ensure carrier moves first
            for player in self.evaders: # Remaining evaders move after
                if player != self.flag.carrier:
                    agents_to_act.append(player)
            return agents_to_act

    def reachable_teammates(self):
        """Allow flag to be passed through chains of adjacent evaders"""
        carrier = self.flag.carrier
        graph = self.graph

        queue = deque([carrier]) # BFS queue, starting with flag carrier
        visited = set([carrier]) # Set of visited evaders

        reachable = [carrier] # List of evaders that can recieve flag

        while queue: # While there are evaders to explore
            current = queue.popleft()
            # Check every evader to see if they neighbour the current one
            for evader in self.evaders:
                if evader not in visited:
                    # If the evader is adjacent to current, add them to the queue
                    if evader.position in graph.neighbors(current.position):
                        visited.add(evader)
                        reachable.append(evader)
                        queue.append(evader)

        # Return all teammates that can recieve the flag
        return reachable
    
    def check_flag_pass(self):
        """Allow carrier to pass to a reachable teammate."""
        carrier = self.flag.carrier
        pursuer_positions = [pursuer.position for pursuer in self.pursuers]

        # evaders sharing an edge are eligible
        candidates = self.reachable_teammates()
        
        best_candidate = carrier
        best_score = -1

        # Check through all reachable teammates
        for candidate in candidates:
            distances = []
            # Find distance to the nearest pursuer 
            for pursuer in pursuer_positions:
                distances.append(nx.shortest_path_length(self.graph, candidate.position, pursuer))
            min_dist = min(distances)
            # Find the candidat that is furthest from the nearest pursuer
            if min_dist > best_score:
                best_score = min_dist
                best_candidate = candidate
        
        # If a reachable teammate is further from all pursuers, pass the flag
        if best_candidate != carrier:
            # Swap carrier and non carrier strategies 
            carrier_strategy = carrier.strategy
            candidate_strategy = best_candidate.strategy
            best_candidate.strategy = carrier_strategy
            carrier.strategy = candidate_strategy
            # Transfer to new carrier
            self.flag.transfer(best_candidate)
            self.evaders.sort(key=lambda evader: 
                              (0 if evader == self.flag.carrier 
                               else 1,
                               nx.shortest_path_length(self.graph, evader.position, self.flag.carrier.position)))
            
    def check_capture(self):
        """Check if any pursuer shares a vertex with the flag carrier."""
        carrier_pos = self.flag.position()
        for pursuer in self.pursuers:
            if pursuer.position == carrier_pos:
                self.game_over = True
                self.winner = "Pursuers"
                return

    def check_win_conditions(self):
        """Check if the turn limit has been reached."""
        if self.current_turn >= self.max_turns:
            self.game_over = True
            self.winner = "Evaders"

    def change_turn(self):
        """Switch the acting team, increment counter if the entire turn has passed"""
        if self.current_team == "E":
            self.current_team = "P"
            self.current_turn += 1
        else:
            self.current_team = "E"

class CaptureTheFlag:
    """
    Main game controller
    Used to draw the graph, run training episodes, and execute the game loop
    """
    def __init__(self, graph, pursuers, evaders, flag_carrier, max_turns, pursuer_controller):
        self.state = GameState(graph, pursuers, evaders, flag_carrier, max_turns)
        self.layout = nx.kamada_kawai_layout(self.state.graph) # Precompute graph layout
        self.controller = MoveController() # Controller to execute strategies
        self.pursuer_controller = pursuer_controller  # Controller for RL

    def train(self, episode=0, evader_baseline=1, log=None):
        """Run one full training episode, from beggining to end"""
        episode_reward = 0.0  # Track the reward for the episode
        flag_transfers = 0    # Track the flag passes for the episode

        while not self.state.game_over:
            # All agents of the active team perform moves this phase
            agents = self.state.agents_to_act()
            for agent in agents:
                # Build RL state representation
                context = self.state.get_context(agent)
                
                # Only pursuers use RL strategies
                if agent.is_pursuer:
                    # Select action, using epsilon greedy policy
                    action = self.pursuer_controller.select_action(context)
                    # Store context & chosen action for learning update
                    agent.last_action = action
                    agent.last_context = context
                    # Store flag distance before move, for reward calculation
                    agent.d_before = nx.shortest_path_length(self.state.graph, agent.position, self.state.flag.position())
                    # Assign the selected strategy
                    agent.strategy = action
                else:
                    # Evader flag distance reward calculations (not used here)
                    self.state.flag.d_before = min(
                        nx.shortest_path_length(self.state.graph, self.state.flag.position(), pursuer.position)
                        for pursuer in self.state.pursuers
                    )

                # Use move controller to compute the new position, and move agent
                new_position = self.controller.choose_move(agent, self.state)
                agent.position = new_position

            # Evaders may attempt a flag pass
            if self.state.current_team == "E":
                prev_carrier = self.state.flag.carrier
                self.state.check_flag_pass()
                # Increment flag_transfers if there is a new carrier
                if self.state.flag.carrier != prev_carrier:
                    flag_transfers += 1

            # Check capture / evader win
            self.state.check_capture()
            self.state.check_win_conditions()

            # Learning update
            for agent in self.state.agents_to_act():
                if agent.is_pursuer:
                    reward = self.state.compute_reward(agent)
                    episode_reward += reward
                    self.pursuer_controller.update(agent.last_context, agent.last_action, reward)

            # If game not over, switch turn
            if not self.state.game_over:
                self.state.change_turn()
            else:
                # If game over, apply terminal rewards
                for agent in self.state.pursuers:
                    reward = self.state.compute_reward(agent)
                    episode_reward += reward
                    self.pursuer_controller.update(agent.last_context, agent.last_action, reward)

        # Write the outcomes of the training episode to the log
        if log is not None:
            log.append({
                "episode": episode,
                "phase": "train",
                "evader_baseline": evader_baseline,
                "n_pursuers": len(self.state.pursuers),
                "n_evaders": len(self.state.evaders),
                "winner": self.state.winner,
                "evader_win": 1 if self.state.winner == "Evaders" else 0,
                "turns": self.state.current_turn,
                "flag_transfers": flag_transfers,
                "total_reward": round(episode_reward, 3),
                "epsilon": round(self.pursuer_controller.epsilon, 4)
            })

    def evaluate(self, episode=0, evader_baseline=1, log=None):
        """Run a single visualised game using trained policy"""
        flag_transfers = 0 # Store number of flag passes

        while not self.state.game_over:
            # All agents of the active team perform moves
            agents = self.state.agents_to_act()
            for agent in agents:
                context = self.state.get_context(agent)

                # Run RL process for pursuers
                if agent.is_pursuer:
                    # Choose the best learned strategy
                    action = self.pursuer_controller.select_action(context)
                    agent.last_action = action
                    agent.last_context = context
                    agent.d_before = nx.shortest_path_length(self.state.graph, agent.position, self.state.flag.position())
                    # Assign the chosen strategy
                    agent.strategy = action
                # Execute the move with Move Controller
                new_position = self.controller.choose_move(agent, self.state)
                agent.position = new_position

            # Evaders may attempt a flag pass
            if self.state.current_team == "E":
                prev_carrier = self.state.flag.carrier
                self.state.check_flag_pass()
                # Increment flag_transfers if there is a new carrier
                if self.state.flag.carrier != prev_carrier:
                    flag_transfers += 1

            # Check capture / evader win
            self.state.check_capture()
            self.state.check_win_conditions()

            # No Q updates during evaluation
            if not self.state.game_over:
                self.state.change_turn()

        if log is not None:
            log.append({
                "episode": episode,
                "phase": "eval",
                "evader_baseline": evader_baseline,
                "n_pursuers": len(self.state.pursuers),
                "n_evaders": len(self.state.evaders),
                "winner": self.state.winner,
                "evader_win": 1 if self.state.winner == "Evaders" else 0,
                "turns": self.state.current_turn,
                "flag_transfers": flag_transfers,
                "total_reward": None,
                "epsilon": round(self.pursuer_controller.epsilon, 4),
            })

# Baseline role generators

def get_pursuer_roles(n_pursuers):
    """Starting roles for RL pursuers — RL overrides these."""
    baseline = ["chase", "chase", "flank", "flank", "intercept", "intercept", "lurk", "lurk"]
    return baseline[:n_pursuers]

def get_evader_roles(baseline, num_evaders):
    """Return the evader roles for a given baseline and team size."""
    if baseline == 1:
        # Baseline 1 — carrier escapes, rest support
        return ["escape"] + ["support"] * (num_evaders - 1)

    elif baseline == 2:
        # Baseline 2 — carrier patrols, half support, half expand
        non_carriers    = num_evaders - 1
        num_support = non_carriers // 2
        num_expand    = non_carriers - num_support
        return ["patrol"] + ["support"] * num_support + ["expand"] * num_expand

    elif baseline == 3:
        # Baseline 3 — escape, support, expand chain
        if num_evaders == 4:
            return ["escape", "support", "expand", "expand"]
        elif num_evaders == 6:
            return ["escape", "support", "support", "expand", "expand", "expand"]
        elif num_evaders == 8:
            return ["escape", "support", "support", "expand", "expand", "expand", "expand", "expand"]
        else:
            num_support = max(1, (num_evaders - 1) // 3)
            num_expand = num_evaders - 1 - num_support
            return ["escape"] + ["support"] * num_support + ["expand"] * num_expand

    raise ValueError("Error: evader baseline " + str(baseline))


# Experiment global variables
CONFIGURATIONS = [(4,4), (4,6), (4,8), (6,4), (6,6), (6,8), (8,4), (8,6), (8,8)]
EVADER_BASELINES = [1, 2, 3]
TRAIN_EPISODES   = 200
EVAL_EPISODES    = 20
MAX_TURNS        = 200

def build_graph():
    """Create an 8x8 grid graph and label its vertices"""
    graph = nx.grid_2d_graph(8,8) # create graph
    
    # Generate labels (0-63)
    labels = {}
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        labels[nodes[i]] = i
    graph = nx.relabel_nodes(graph, labels)

    return graph

def run_rl_experiments(output_csv="rl_pursuers_results.csv", qtable_csv="rl_pursuers_q_table.csv"):
    """Run the simulations for each experimental condition and store results"""
    # Build the graph for all experiments
    graph = build_graph()
    # List to store training/evaluation logs
    log = []
    # Store the controller for q table output
    last_controller = None

    # Total number of experiments to run
    total = len(CONFIGURATIONS) * len(EVADER_BASELINES)
    done = 0

    for evader_baseline in EVADER_BASELINES:
        for (num_pursuers, num_evaders) in CONFIGURATIONS:
            done += 1
            print("[" + str(done) + "/" + str(total) + "]  Evader-BL" + str(evader_baseline) + "  " 
                + str(num_pursuers) + "v" + str(num_evaders) 
                + " - training: " + str(TRAIN_EPISODES) + " episodes")
            
            # Initialise controller for RL, for pursuers
            pursuer_controller = RLController(actions=["chase", "flank", "intercept", "lurk"])
            # Assign initial roles to pursuer and evader agents
            pursuer_roles = get_pursuer_roles(num_pursuers)
            evader_roles = get_evader_roles(evader_baseline, num_evaders)

            # Training
            for episode in range(TRAIN_EPISODES):
                pursuers = []
                evaders = []
                # Initialise pursuer agents
                for role in pursuer_roles:
                    start = Placement.choose_start_position(role, graph, pursuers, evaders)
                    pursuers.append(Agent(start, role, is_pursuer=True))
                for role in evader_roles:
                    start = Placement.choose_start_position(role, graph, pursuers, evaders)
                    evaders.append(Agent(start, role, is_pursuer=False))

                game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier=evaders[0], max_turns=MAX_TURNS, pursuer_controller=pursuer_controller)
                game.train(episode=episode, evader_baseline=evader_baseline, log=log)

            # Evaluation (epsilon=0, no Q updates)
            print("Evaluating: " + str(EVAL_EPISODES) + " episodes")
            pursuer_controller.epsilon = 0.0 # Set exploration to 0

            for episode in range(EVAL_EPISODES):
                pursuers = []
                evaders = []
                # Initialise pursuer agents
                for role in pursuer_roles:
                    start = Placement.choose_start_position(role, graph, pursuers, evaders)
                    pursuers.append(Agent(start, role, is_pursuer=True))
                for role in evader_roles:
                    start = Placement.choose_start_position(role, graph, pursuers, evaders)
                    evaders.append(Agent(start, role, is_pursuer=False))

                # Create game and run evaluation episode
                game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier=evaders[0], max_turns=MAX_TURNS, pursuer_controller=pursuer_controller)
                game.evaluate(episode=episode, evader_baseline=evader_baseline, log=log)

            # Store controller
            last_controller = pursuer_controller

    # Save episode log
    fieldnames = ["episode", "phase", "evader_baseline", "n_pursuers", "n_evaders",
                  "winner", "evader_win", "turns", "flag_transfers", "total_reward", "epsilon"]
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)
    print("\nEpisode log saved to " + output_csv)

    # Save Q-table from final trained condition
    q_rows = last_controller.output_q_table()
    q_fields = ["dist_to_flag", "dist_to_teammate", "flag_pressure",
                "graph_position", "action", "q_value", "visit_count"]
    with open(qtable_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=q_fields)
        writer.writeheader()
        writer.writerows(q_rows)
    print("Q-table saved to " + qtable_csv)

    return log

if __name__ == "__main__":
    run_rl_experiments()