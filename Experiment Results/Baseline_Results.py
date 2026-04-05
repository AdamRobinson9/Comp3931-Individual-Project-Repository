import random
import csv
import networkx as nx
from collections import deque
from Strategies import MoveController
from Placement import Placement

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
    Used to execute the game loop and log the outcomes
    """
    def __init__(self, graph, pursuers, evaders, flag_carrier, max_turns):
        self.state = GameState(graph, pursuers, evaders, flag_carrier, max_turns)
        self.controller = MoveController() # Controller to execute strategies

    def simulate(self):
        """
        Run a full game until winning condition met. 
        Return the winner, no. turns, no. flag passes.
        """
        # Store the number of flag passes
        flag_transfers = 0

        while not self.state.game_over:
            # All agents of the active team perform moves
            agents = self.state.agents_to_act()
            for agent in agents:
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

            if not self.state.game_over:
                self.state.change_turn()

        # Return game outcome metrics for logging
        return self.state.winner, self.state.current_turn, flag_transfers

# Baseline role generators

def get_pursuer_roles(baseline, num_pursuers):
    """Return the pursuer roles for a given baseline and team size."""
    if baseline == 1:
        # Baseline 1 — all chase
        return ["chase"] * num_pursuers

    elif baseline == 2:
        # Baseline 2 — half chase, half lurk
        return ["chase", "flank"] * (num_pursuers//2)

    elif baseline == 3:
        # Baseline 3 — chase, intercept, flank, lurk spread
        if num_pursuers == 4:
            return ["chase", "intercept", "flank", "lurk"]
        elif num_pursuers == 6:
            return ["chase", "intercept", "intercept", "flank", "flank", "lurk"]
        elif num_pursuers == 8:
            return ["chase", "chase", "intercept", "intercept", "flank", "flank", "lurk", "lurk"]
        else:
            quarter = max(1, num_pursuers // 4)
            return (["chase"] * quarter +
                    ["intercept"] * quarter +
                    ["flank"] * quarter +
                    ["lurk"] * (num_pursuers - 3 * quarter))

    raise ValueError("Error: pursuer baseline " + str(baseline))

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
PURSUER_BASELINES = [1, 2, 3]
EVADER_BASELINES = [1, 2, 3]
EPISODES = 20
MAX_TURNS = 200

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

def run_episode(graph, pursuer_roles, evader_roles, max_turns):
    """Build agents, run one game & return outcome metrics."""
    pursuers = []
    evaders = []

    # Initialise pursuers
    for role in pursuer_roles:
        start = Placement.choose_start_position(role, graph, pursuers, evaders)
        pursuers.append(Agent(start, role, is_pursuer=True))

    # Initialise evaders
    for role in evader_roles:
        start = Placement.choose_start_position(role, graph, pursuers, evaders)
        evaders.append(Agent(start, role, is_pursuer=False))
    
    # Create game instance
    game = CaptureTheFlag(graph, pursuers, evaders, flag_carrier=evaders[0], max_turns=max_turns)
    # Run the game and return its outcomes
    return game.simulate()

def run_baseline_experiments(episodes=EPISODES, output_csv="Baseline_Results.csv"):
    """Simulate the games for each experimental condition and store the results"""
    # Build the graph for all experiments
    graph = build_graph()
    # List to store all results
    results = []

    # Total number of experiments to run
    total = len(CONFIGURATIONS) * len(PURSUER_BASELINES) * len(EVADER_BASELINES) * episodes
    done = 0 # Track completed experiments

    # Loop over all combinations of pursuer and evader baselines
    for pursuer_baseline in PURSUER_BASELINES:
        for evader_baseline in EVADER_BASELINES:
            # Loop over different team sizes
            for (num_pursuers, num_evaders) in CONFIGURATIONS:
                # Assign roles to pursuer and evader agents
                pursuer_roles = get_pursuer_roles(pursuer_baseline, num_pursuers)
                evader_roles = get_evader_roles(evader_baseline, num_evaders)

                # Lists to track the results of each episode
                winners = []
                turn_counts = []
                transfer_counts = []

                # Run all episodes for the configuration and log its result
                for episode in range(episodes):
                    winner, turns, transfers = run_episode(graph, pursuer_roles, evader_roles, MAX_TURNS)
                    winners.append(winner)
                    turn_counts.append(turns)
                    transfer_counts.append(transfers)

                    done += 1
                    # Print simulation result to terminal
                    print("[" + str(done) + "/" + str(total) + "]  Pursuer-BL" + str(pursuer_baseline) +
                          " vs Evader-BL" + str(evader_baseline) + "  " +
                          str(num_pursuers) + "v" + str(num_evaders) +
                          "  Episode " + str(episode + 1) + ": " +
                          str(winner) + " in " + str(turns) +" turns")

                # Compute overall results from the configuration
                evader_wins = sum(1 for winner in winners if winner == "Evaders")
                win_rate = evader_wins / episodes
                avg_turns = sum(turn_counts) / episodes
                avg_transfers = sum(transfer_counts) / episodes

                # Store the results from the configuration
                results.append({
                    "pursuer_baseline": pursuer_baseline,
                    "evader_baseline": evader_baseline,
                    "n_pursuers": num_pursuers,
                    "n_evaders": num_evaders,
                    "pursuer_roles": ",".join(pursuer_roles),
                    "evader_roles": ",".join(evader_roles),
                    "episodes": episodes,
                    "evader_wins": evader_wins,
                    "evader_win_rate": round(win_rate, 3),
                    "avg_turns": round(avg_turns, 1),
                    "avg_flag_transfers": round(avg_transfers, 2),
                })

    # Write CSV file
    fields = list(results[0].keys())
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to " + output_csv)
    return results

if __name__ == "__main__":
    results = run_baseline_experiments(episodes=EPISODES)