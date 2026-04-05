import random
import networkx as nx
from collections import deque


class Placement:
    @staticmethod
    def choose_start_position(role, graph, pursuers, evaders):
        """Dispatcher method, call the method for the corresponding role"""

        strategies = {
            # Pursuers
            "chase": Placement.chase_start,
            "intercept": Placement.chase_start,
            "flank": Placement.degree_start,
            "lurk": Placement.degree_start,

            # Evaders
            "escape": Placement.escape_start,
            "handoff": Placement.escape_start,
            "decoy": Placement.decoy_start,
            "patrol": Placement.patrol_start,
            "expand": Placement.expand_start,
            "support": Placement.support_start
        }

        if role in strategies:
            return strategies[role](graph, pursuers, evaders)

        return random.choice(list(graph.nodes()))

    # Utility Functions
    @staticmethod
    def occupied_nodes(pursuers, evaders):
        return {pursuer.position for pursuer in pursuers} | {evader.position for evader in evaders}

    @staticmethod
    def available_nodes(graph, pursuers, evaders):
        occupied = Placement.occupied_nodes(pursuers, evaders)
        return [node for node in graph.nodes() if node not in occupied]

    @staticmethod
    def min_dist_to_pursuers(graph, node, pursuers):
        if not pursuers:
            return 999
        return min(nx.shortest_path_length(graph, node, p.position) for p in pursuers)

    # Pursuer positions:

    # Chase / Intercept - start at most central vertex
    @staticmethod
    def chase_start(graph, pursuers, evaders):

        # Get available vertices - not already occupied
        available = Placement.available_nodes(graph, pursuers, evaders)
        if not available:
            return random.choice(list(graph.nodes())) # Choose randomly if none free

        # Choose the most central available vertex
        centrality = nx.closeness_centrality(graph)
        return max(available, key=lambda vertex: centrality[vertex])

    # Flank / Lurk - start at high degree nodes
    @staticmethod
    def degree_start(graph, pursuers, evaders):

        available = Placement.available_nodes(graph, pursuers, evaders)
        if not available:
            return random.choice(list(graph.nodes()))

        return max(available, key=lambda n: graph.degree[n])

    # EVADER START STRATEGIES

    # Escape / Handoff - Start far from pursuers, avoid articulation points
    @staticmethod
    def escape_start(graph, pursuers, evaders):

        nodes = list(graph.nodes())
        articulation = set(nx.articulation_points(graph))

        # Only consider vertices that are not articulation points
        candidates = [node for node in nodes if node not in articulation]

        # If there are no suitable options, choose from any vertex
        if not candidates:
            candidates = nodes

        # Choose furthest option from pursuers
        return max(
            candidates,
            key=lambda node: Placement.min_dist_to_pursuers(graph, node, pursuers)
        )

    # Decoy - start at high degree, safe vertices
    @staticmethod
    def decoy_start(graph, pursuers, evaders):
        nodes = list(graph.nodes())
        # Pick furthest from pursuers, with highest degree
        return max(
            nodes,
            key=lambda node: (
                Placement.min_dist_to_pursuers(graph, node, pursuers),
                graph.degree[node]))

    # Patrol - start in central areas away from pursuers
    @staticmethod
    def patrol_start(graph, pursuers, evaders):
        # Collect evader possitions
        evader_occupied = {evader.position for evader in evaders}

        # Make sure the agent can't be immediately caught
        excluded = set()
        for pursuer in pursuers:
            excluded.add(pursuer.position)
            excluded.update(graph.neighbors(pursuer.position))

        # Pick a position that is not already occupied by a teammate
        options = [
            node for node in graph.nodes()
            if node not in evader_occupied and node not in excluded
        ]

        # If all vertices are already occupied, consider all vertices on the graph
        if not options:
            options = [node for node in graph.nodes() if node not in evader_occupied]
        if not options:
            options = list(graph.nodes())

        # Choose the most central option
        centrality = nx.closeness_centrality(graph)
        return max(options, key=lambda node: centrality[node])

    # expand - start at a neighbour of teammates that is safe
    @staticmethod
    def expand_start(graph, pursuers, evaders):
        # Locate positions of the other evaders
        evader_occupied = {evader.position for evader in evaders}

        # Locate neighbours of all evaders that are not currently occupied - candidates to place
        candidates = set()
        for evader in evaders:
            for neighbour in graph.neighbors(evader.position):
                if neighbour not in evader_occupied:
                    candidates.add(neighbour)

        # If there are no candidates, choose a random vertex
        if not candidates:
            return random.choice(list(graph.nodes()))

        # Choose the option that is furthest from the nearest pursuer
        return max(
            candidates,
            key=lambda node: Placement.min_dist_to_pursuers(graph, node, pursuers)
        )

    # SUPPORT - start at closest safe vertex to the flag carrier
    @staticmethod
    def support_start(graph, pursuers, evaders):

        # If there are no other evaders pick a random vertex (fallback)
        if not evaders:
            return random.choice(list(graph.nodes()))

        # Locate the flag carrier and other evader positions
        carrier = evaders[0].position
        evader_occupied = {evader.position for evader in evaders}

        # BFS starting from flag carrier
        visited = {carrier}
        queue = deque([(carrier, 0)])
        # Store best visited option
        best_node = carrier
        best_dist = None
        best_safety = -1

        while queue:

            node, dist = queue.popleft()

            # Only consider vertices not already occupied by ther evaders
            if node not in evader_occupied or node == carrier:

                safety = Placement.min_dist_to_pursuers(graph, node, pursuers)

                # Update best option if is closer to the carrier, break ties with safety
                if node != carrier and (
                    best_dist is None
                    or dist < best_dist
                    or (dist == best_dist and safety > best_safety)):
                    best_node = node
                    best_dist = dist
                    best_safety = safety

            # Stop searching when best distance has lready been found
            if best_dist is not None and dist > best_dist:
                break

            # Add neighbours to BFS queue
            for neighbour in graph.neighbors(node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, dist + 1))

        return best_node