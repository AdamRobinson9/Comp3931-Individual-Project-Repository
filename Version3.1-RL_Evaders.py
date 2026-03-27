import random
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

        # PURSUER REWARD
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
            if agent.position in [pursuer.position for pursuer in self.pursuers if pursuer != agent]:
                reward -=0.5

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
    def __init__(self, graph, pursuers, evaders, flag_carrier, max_turns, evader_controller):
        self.state = GameState(graph, pursuers, evaders, flag_carrier, max_turns)
        self.layout = nx.kamada_kawai_layout(self.state.graph) # Precompute graph layout
        self.controller = MoveController() # Controller to execute strategies
        self.evader_controller = evader_controller # Controller for RL

    def draw_graph(self):
        """Draw the graph and current positions of all agents & flag"""
        plt.clf()
        graph = self.state.graph
        layout = self.layout

        # Draw the graph structure
        nx.draw(
            graph,
            layout,
            with_labels=True,
            node_color="white",
            edgecolors="black",
            font_size=8
        )

        # Draw the flag carrier (highlighted yellow)
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=[self.state.flag.position()],
            node_color="none",
            edgecolors="yellow",
            linewidths=4,
            node_size=500)

        # Draw the pursuers (red)
        pursuer_positions = [pursuer.position for pursuer in self.state.pursuers]
        nx.draw_networkx_nodes(graph, layout, nodelist=pursuer_positions, node_color="red", alpha=0.5)

        # Draw the evaders (blue)
        evader_positions = [evader.position for evader in self.state.evaders]
        nx.draw_networkx_nodes(graph, layout, nodelist=evader_positions, node_color="blue", alpha=0.5)

        plt.pause(0.1)

    def train(self):
        """Run one full training episode, from beggining to end"""
        while not self.state.game_over:
            # All agents of the active team perform moves this phase
            agents = self.state.agents_to_act()
            for agent in agents:
                # Build RL state representation
                context = self.state.get_context(agent)
                
                # Only evaders use RL strategies
                if not agent.is_pursuer:
                    # Make sure agents only use actions for their role
                    if agent.has_flag:
                         allowed_actions = ["escape", "handoff", "patrol", "decoy"]
                    else:
                        allowed_actions = ["support", "expand"]
                    # Select action, using epsilon greedy policy
                    action = self.evader_controller.select_action(context, allowed_actions)
                    # Store context & chosen action for learning update
                    agent.last_action = action
                    agent.last_context = context
                    # Store flag distance before move, for reward calculation
                    self.state.flag.d_before = min(nx.shortest_path_length(self.state.graph, self.state.flag.position(), pursuer.position) 
                                                   for pursuer in self.state.pursuers)
                    # Assign the selected strategy
                    agent.strategy = action

                # Use move controller to compute the new position, and move agent
                new_position = self.controller.choose_move(agent, self.state)
                agent.position = new_position

            # Evaders may attempt a flag pass
            if self.state.current_team == "E":
                self.state.check_flag_pass()            

            # Check capture / evader win
            self.state.check_capture()
            self.state.check_win_conditions()

            # Learning update
            for agent in self.state.agents_to_act():
                if not agent.is_pursuer:
                    reward = self.state.compute_reward(agent)
                    self.evader_controller.update(agent.last_context, agent.last_action, reward)

            # If game not over, switch turn
            if not self.state.game_over:
                self.state.change_turn()
            else:
                # If game over, apply terminal rewards
                for agent in self.state.evaders:
                    reward = self.state.compute_reward(agent)
                    self.evader_controller.update(agent.last_context, agent.last_action, reward)
                print("Episode complete")

    def play(self):
        """Run a single visualised game using trained policy"""
        while not self.state.game_over:
            self.draw_graph()

            # All agents of the active team perform moves
            agents = self.state.agents_to_act()
            for agent in agents:
                context = self.state.get_context(agent)
                
                # Run RL process for evaders
                if not agent.is_pursuer:
                    if agent.has_flag:
                         allowed_actions = ["escape", "handoff", "patrol", "decoy"]
                    else:
                        allowed_actions = ["support", "expand"]
                    # Choose the best learned strategy
                    action = self.evader_controller.select_action(context, allowed_actions)
                    agent.last_action = action
                    agent.last_context = context
                    self.state.flag.d_before = min(nx.shortest_path_length(self.state.graph, self.state.flag.position(), pursuer.position) for pursuer in self.state.pursuers)
                    # Assign the chosen strategy
                    agent.strategy = action
                    print(agent.position, action) # Print the chosen strategy

                # Execute the move with Move Controller
                new_position = self.controller.choose_move(agent, self.state)
                agent.position = new_position

            # Evaders may attempt a flag pass
            if self.state.current_team == "E":
                self.state.check_flag_pass()

            # Check capture / evader win
            self.state.check_capture()
            self.state.check_win_conditions()

            if not self.state.game_over:
                self.state.change_turn()

            input("[ENTER] next turn " + str(self.state.current_team))

        print("\nGAME OVER — Winner:", self.state.winner, "- Turns Taken:", self.state.current_turn)

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

def main():
    graph = build_graph()
    # Build pursuer & evader teams and choose initial strategies
    num_pursuers = 4
    pursuer_roles = ["chase", "flank", "intercept", "lurk", "flank", "intercept", "chase", "flank"]
    num_evaders = 6
    evader_roles = ["escape", "support", "expand","expand", "expand", "expand", "expand", "expand","expand","expand"]
    # Initialise shared RL controller for evader team
    evader_controller = RLController(actions = ["escape", "handoff", "support", "patrol", "expand", "decoy"])

    # Create and run 200 training games
    for episode in range(200):
        pursuers = []
        evaders = []

        for i in range(num_pursuers):
            start = Placement.choose_start_position(pursuer_roles[i], graph, pursuers, evaders)
            pursuers.append(Agent(start, pursuer_roles[i], True))

        for i in range(num_evaders):
            start = Placement.choose_start_position(evader_roles[i], graph, pursuers, evaders)
            evaders.append(Agent(start, evader_roles[i], False))

        flag_carrier = evaders[0]

        game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier, max_turns=200, evader_controller=evader_controller)
        game.train()

    # Create the agents for the evaluation game
    pursuers=[]
    evaders = []
    for i in range(num_pursuers):
        start = Placement.choose_start_position(pursuer_roles[i], graph, pursuers, evaders)
        pursuers.append(Agent(start_node=start, strategy=pursuer_roles[i], is_pursuer=True))
    
    for i in range(num_evaders):
        start = Placement.choose_start_position(evader_roles[i], graph, pursuers, evaders)
        evaders.append(Agent(start_node=start, strategy=evader_roles[i], is_pursuer=False))

    flag_carrier = evaders[0] # Assign flag

    evader_controller.epsilon = 0 # Exploitation only during play
    # Create demonstration game
    game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier, max_turns=200, evader_controller=evader_controller)
    # Run demonstration game
    game.play()

if __name__ == "__main__":
    main()
