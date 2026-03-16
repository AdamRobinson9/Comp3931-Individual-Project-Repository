import random
import time
import networkx as nx
import matplotlib.pyplot as plt
from Strategies import MoveController
from Placement import Placement
from collections import deque
plt.ion()

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
    Used to draw the graph and execute the game loop
    """
    def __init__(self, graph, pursuers, evaders, flag_carrier, max_turns):
        self.state = GameState(graph, pursuers, evaders, flag_carrier, max_turns)
        self.layout = nx.kamada_kawai_layout(self.state.graph) # Precompute graph layout
        self.controller = MoveController() # Controller to execute strategies

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

    def play(self):
        """Main game loop, runs until a winning condition is met"""
        while not self.state.game_over:
            self.draw_graph()

            # All agents of the active team perform moves 
            agents = self.state.agents_to_act()
            for agent in agents:
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
    """Game setup, create game instance and run the game"""
    graph = build_graph()
    # Build pursuer & evader teams and choose fixed strategies
    num_pursuers = 4
    pursuer_roles = ["chase", "flank", "chase", "flank", "chase", "flank", "chase", "flank"]
    num_evaders = 8
    evader_roles = ["escape", "support", "support","support", "support", "support", "support", "support"]

    # Create the agents and add them to the graph
    pursuers=[]
    evaders = []
    for i in range(num_pursuers):
        start = Placement.choose_start_position(pursuer_roles[i], graph, pursuers, evaders)
        pursuers.append(Agent(start_node=start, strategy=pursuer_roles[i], is_pursuer=True))
    
    for i in range(num_evaders):
        start = Placement.choose_start_position(evader_roles[i], graph, pursuers, evaders)
        evaders.append(Agent(start_node=start, strategy=evader_roles[i], is_pursuer=False))

    # Assign flag
    flag_carrier = evaders[0]

    # Create game instance
    game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier, max_turns=200)
    
    # Start game
    game.play()


if __name__ == "__main__":
    main()