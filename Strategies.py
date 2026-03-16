import networkx as nx
import random

class MoveController:
    
    @staticmethod
    def shortest_distance(graph, u, v):
        """Return shortest path distance between vertices u and v"""
        try:
            return nx.shortest_path_length(graph, u, v)
        except nx.NetworkXNoPath: # Fallback
            return float("inf")

    @staticmethod
    def nearest_pursuer_distance(graph, node, pursuers):
        """Retuen the distance of the nearest pursuer to the passed node"""
        if not pursuers: # Fallback
            return float("inf")
        return min(MoveController.shortest_distance(graph, node, pursuer.position) for pursuer in pursuers)

    @staticmethod
    def best_by_centrality(nodes, graph):
        """Return the most central node in the given list of nodes"""
        centrality = nx.closeness_centrality(graph)
        if not nodes: # Fallback
            return None
        return max(nodes, key=lambda node: centrality[node])

    @staticmethod
    def avoid_teammates_pursuer(agent, state, candidates):
        """Filter out nodes already occupied by other pursuers."""
        occupied = {pursuer.position for pursuer in state.pursuers if pursuer != agent}
        # if all candidates are occupied, return them all
        if occupied:
            return [node for node in candidates if node not in occupied]
        else:
            return candidates
    
    # Evader Strategies - 
    
    @staticmethod
    def escape(agent, state):
        """Move to position furthes from nearest pursuer"""
        graph = state.graph
        # Can only move to a neighbour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        best_distance = 0
        best_choice = agent.position
        for choice in choices:
            distance = MoveController.nearest_pursuer_distance(graph, choice, state.pursuers)
            # Choose moves that increase distance from pursuers
            if distance > best_distance:
                best_distance = distance
                best_choice = choice
            # Break ties with centrality
            if distance == best_distance:
                best_choice = MoveController.best_by_centrality([best_choice, choice], graph)

        return best_choice

    @staticmethod
    def handoff(agent, state):
        """Move towards a teammate that can safely recieve the flag"""
        graph = state.graph
        evaders = state.evaders
        pursuers = state.pursuers

        best_score = float("-inf")
        best_distance = float("inf")
        best_target = None

        for teammate in evaders:
            if teammate == agent:
                continue

            safety = MoveController.nearest_pursuer_distance(graph, teammate.position, pursuers)
            # pick neighbour of teammate closest to carrier
            target = None
            min_distance = float("inf")
            for neighbour in graph.neighbors(teammate.position):
                distance = MoveController.shortest_distance(graph, agent.position, neighbour)
                if distance < min_distance:
                    min_distance = distance
                    target = neighbour

            # Choose best tradeoff between safety and distance
            score = safety - min_distance
            if score > best_score:
                best_score = score
                best_target = target
                best_distance = min_distance
            elif score == best_score and min_distance < best_distance:
                best_target = target
                best_distance = min_distance

        # fallback if no suitable teammate found
        if best_target is None:
            return MoveController.escape(agent, state)

        # move one step along shortest path toward best_target
        try:
            path = nx.shortest_path(graph, agent.position, best_target)
            if len(path) > 1:
                return path[1]
            else: 
                return agent.position
        except nx.NetworkXNoPath:
            return agent.position

    @staticmethod
    def support(agent, state):
        """Stay near flag carrier and maximise distance from pursuers"""
        graph = state.graph
        flag_carrier = state.flag.position()
        pursuers = state.pursuers
        neighbours = graph.neighbors(flag_carrier)
        
        # Can only move to a neighour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        
        # Find distance to nearest pursuer for each possible move
        safety=[]
        for choice in choices:
            safety.append(MoveController.nearest_pursuer_distance(graph,choice,pursuers))

        # Avoid positions occupied by teammates
        occupied = {
            evader.position for evader in state.evaders
            if evader != agent
        }
        choices = [choice for choice in choices if choice not in occupied]
        
        # Move directly to a neighbour of the flag carrier if possible
        neighbour_choices =[]
        neighbour_safety=[]
        for i in range(len(choices)):
            if choices[i] in neighbours:
                neighbour_choices.append(choices[i])
                neighbour_safety.append(safety[i])
        
        for i in range(len(neighbour_choices)):
            if neighbour_safety[i] == max(neighbour_safety):
                return neighbour_choices[i]
            
        # Otherwise, move one step toward flag
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = float("inf")
        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, flag_carrier) 
            degree = graph.degree(choice)    
            # Prioritise moving closer
            if distance < best_distance:
                best_choice = choice
                best_distance = distance
                best_degree = degree
            # Break ties by moving to higher degree
            elif distance == best_distance and degree < best_degree:
                best_choice = choice
                best_degree = degree
        return best_choice

    @staticmethod
    def patrol(agent, state):
        """Move to the most central neighbour"""
        graph = state.graph
        # go to the neighbour with highest degree (local centrality)
        neighbours = list(graph.neighbors(agent.position))
        if not neighbours:
            return agent.position
        return MoveController.best_by_centrality(neighbours, graph)

    @staticmethod
    def expand(agent, state):
        """Move towards nodes adjacent to any teammate"""
        graph = state.graph
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        teammates = [evader for evader in state.evaders if evader != agent]
        
        # Vertices adjacent to all teammates
        teammate_neighbours=[] 
        for teammate in teammates:
            teammate_neighbours.extend(list(graph.neighbors(teammate.position)))

        # If you can, go for a teammate's neighbour immediately
        reachable_targets = [target for target in teammate_neighbours if target in choices]

        # If not immediately reachable, move toward nearest teammate
        if not reachable_targets:
            target = None
            min_distance = float("inf")
            for option in teammate_neighbours:
                distance = MoveController.shortest_distance(graph, agent.position, option)
                if distance < min_distance:
                    min_distance = distance
                    target = option
            
            # Move one step toward target with tie-breaking by degree
            best_distance = float("inf")
            best_choice = agent.position
            best_degree = float("inf")

            for choice in choices:
                if choice not in [teammate.position for teammate in teammates]:
                    distance = MoveController.shortest_distance(graph, choice, target)
                    degree = graph.degree(choice)
                    if distance < best_distance:
                        best_choice = choice
                        best_distance = distance
                        best_degree = degree
                    elif distance == best_distance and degree < best_degree:
                        best_choice = choice
                        best_degree = degree

            return best_choice

        # If already near teammate, pick safest neighbour
        best_choice = agent.position
        best_safety = -1
        for target in reachable_targets:
            if target not in [teammate.position for teammate in teammates]:
                safety = MoveController.nearest_pursuer_distance(graph, target, state.pursuers)
                if safety > best_safety:
                    best_safety = safety
                    best_choice = target

        # Fallback
        if best_choice is None:
            return random.choice(graph.neighbors(agent.position))

        return best_choice

    @staticmethod
    def decoy(agent, state):
        """Move to a suboptimal escape route that remains safe"""
        graph = state.graph
        pursuers = state.pursuers

        # Can only move to a neighbour or stay put
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # Calculate distance from nearest pursuer for each possible move       
        safety_scores = []
        for choice in choices:
            safety_scores.append(MoveController.nearest_pursuer_distance(graph, choice, pursuers))

        # Generate list of moves that are safe but not optimal
        max_safety = max(safety_scores)
        good_moves = []
        for choice, safety in zip(choices, safety_scores):
            if safety > 1 and safety <= max_safety - 1:
                good_moves.append(choice)

        # Randomly choose a move from the list
        if good_moves:
            return random.choice(good_moves)
        else:
            return random.choice(choices) # Fallback

    # Pursuer Strategies - 

    @staticmethod
    def chase(agent, state):
        """Choose the move that gets closest to flag carrier"""
        graph = state.graph
        flag_carrier = state.flag.position()
        
        # Possible moves (including staying put)
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can directly capture the flag, do it
        if flag_carrier in choices:
            return flag_carrier
        
        # Find moves that minimize distance to flag carrier
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1  # prefer high-degree vertices

        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, flag_carrier)
            degree = graph.degree(choice)

            # Pick choice that minimises distance
            if distance < best_distance:
                best_distance = distance
                best_choice = choice
                best_degree = degree
            # break ties using highest degree
            elif distance == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice

    @staticmethod
    def flank(agent, state):
        """Approach the flag from nearby vertices"""
        graph = state.graph
        flag_carrier = state.flag.position()

        # Move to any neighour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        if flag_carrier in choices:
            return flag_carrier

        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        # Compute candidate "flank targets" = nodes 1 or 2 steps away from carrier
        targets = set()
        neighbours_1 = set(graph.neighbors(flag_carrier))
        neighbours_2 = set()
        for neighbour in neighbours_1:
            neighbours_2.update(graph.neighbors(neighbour))
        targets = neighbours_1.union(neighbours_2)
        targets.discard(flag_carrier)  # skip carrier's current node

        # Choose the move that gets closest to any potential target
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1

        for choice in choices:
            # distance to closest flank target
            distances = [MoveController.shortest_distance(graph, choice, target) for target in targets]
            if not distances:
                continue
            min_dist = min(distances)
            degree = graph.degree[choice]

            # Prioritise moving closest to target
            if min_dist < best_distance:
                best_distance = min_dist
                best_choice = choice
            # Break ties by choosing highest degree
            elif min_dist == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice


    @staticmethod
    def intercept(agent, state):
        """Predict where the flag carrier will move and target that"""
        graph = state.graph    
        # Predict the flag carriers next position
        flag_carrier = state.flag.carrier
        prediction = MoveController.escape(flag_carrier, state)

        # Can only move to a neighbour or stay put
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can capture the flag, then do it
        if flag_carrier.position in choices:
            return flag_carrier.position
        
        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        # Find moves that minimize distance to prediction
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1  # prefer high-degree nodes

        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, prediction)
            degree = graph.degree(choice)

            # Prefer choices that reduce distance to target
            if distance < best_distance:
                best_distance = distance
                best_choice = choice
                best_degree = degree
            # Break ties using degree
            elif distance == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice

    @staticmethod
    def lurk(agent, state):
        """Shadow flag carrier by aiming to keep distance 3"""
        graph = state.graph
        flag_carrier = state.flag.position()

        # Possible moves (including staying put)
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can capture the flag then do it
        if flag_carrier in choices:
            return flag_carrier

        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        best_choice = agent.position
        best_diff = float("inf")  # difference from desired distance

        for move in choices:
            distance = MoveController.shortest_distance(graph, move, flag_carrier)
            diff = abs(distance - 2)  # target distance = 2

            # Pick choice that is closest to desired distance (2)
            if diff < best_diff:
                best_diff = diff
                best_choice = move
            # Break ties using highest degree
            elif diff == best_diff and graph.degree(move) > graph.degree(best_choice):
                best_choice = move

        return best_choice

    # Map the role to the method computing the move
    Strategies = {
        # Evaders
        "escape": escape,
        "handoff": handoff,
        "support": support,
        "patrol": patrol,
        "expand": expand,
        "decoy": decoy,

        # Pursuers
        "chase": chase,
        "flank": flank,
        "intercept": intercept,
        "lurk": lurk,
    }

    def choose_move(self, agent, state):
        """Execute the strategy assigned to the agent returning a valid move"""
        strategy = agent.strategy

        # random fallback
        graph = state.graph
        neighbours = list(graph.neighbors(agent.position))
        if not neighbours:
            return agent.position

        # run the strategy
        if strategy in self.Strategies:
            new_pos = self.Strategies[strategy](agent, state)

            # ensure it's a legal move
            legal_moves = neighbours + [agent.position]
            if new_pos in legal_moves:
                return new_pos

        # fallback
        return random.choice(neighbours)            return candidates
    
    # Evader Strategies - 
    
    @staticmethod
    def escape(agent, state):
        """Move to position furthes from nearest pursuer"""
        graph = state.graph
        # Can only move to a neighbour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        best_distance = 0
        best_choice = agent.position
        for choice in choices:
            distance = MoveController.nearest_pursuer_distance(graph, choice, state.pursuers)
            # Choose moves that increase distance from pursuers
            if distance > best_distance:
                best_distance = distance
                best_choice = choice
            # Break ties with centrality
            if distance == best_distance:
                best_choice = MoveController.best_by_centrality([best_choice, choice], graph)

        return best_choice

    @staticmethod
    def handoff(agent, state):
        """Move towards a teammate that can safely recieve the flag"""
        graph = state.graph
        evaders = state.evaders
        pursuers = state.pursuers

        best_score = float("-inf")
        best_distance = float("inf")
        best_target = None

        for teammate in evaders:
            if teammate == agent:
                continue

            safety = MoveController.nearest_pursuer_distance(graph, teammate.position, pursuers)
            # pick neighbour of teammate closest to carrier
            target = None
            min_distance = float("inf")
            for neighbour in graph.neighbors(teammate.position):
                distance = MoveController.shortest_distance(graph, agent.position, neighbour)
                if distance < min_distance:
                    min_distance = distance
                    target = neighbour

            # Choose best tradeoff between safety and distance
            score = safety - min_distance
            if score > best_score:
                best_score = score
                best_target = target
                best_distance = min_distance
            elif score == best_score and min_distance < best_distance:
                best_target = target
                best_distance = min_distance

        # fallback if no suitable teammate found
        if best_target is None:
            return MoveController.escape(agent, state)

        # move one step along shortest path toward best_target
        try:
            path = nx.shortest_path(graph, agent.position, best_target)
            if len(path) > 1:
                return path[1]
            else: 
                return agent.position
        except nx.NetworkXNoPath:
            return agent.position

    @staticmethod
    def support(agent, state):
        """Stay near flag carrier and maximise distance from pursuers"""
        graph = state.graph
        flag_carrier = state.flag.position()
        pursuers = state.pursuers
        neighbours = graph.neighbors(flag_carrier)
        
        # Can only move to a neighour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        
        # Find distance to nearest pursuer for each possible move
        safety=[]
        for choice in choices:
            safety.append(MoveController.nearest_pursuer_distance(graph,choice,pursuers))

        # Avoid positions occupied by teammates
        occupied = {
            evader.position for evader in state.evaders
            if evader != agent
        }
        choices = [choice for choice in choices if choice not in occupied]
        
        # Move directly to a neighbour of the flag carrier if possible
        neighbour_choices =[]
        neighbour_safety=[]
        for i in range(len(choices)):
            if choices[i] in neighbours:
                neighbour_choices.append(choices[i])
                neighbour_safety.append(safety[i])
        
        for i in range(len(neighbour_choices)):
            if neighbour_safety[i] == max(neighbour_safety):
                return neighbour_choices[i]
            
        # Otherwise, move one step toward flag
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = float("inf")
        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, flag_carrier) 
            degree = graph.degree(choice)    
            # Prioritise moving closer
            if distance < best_distance:
                best_choice = choice
                best_distance = distance
                best_degree = degree
            # Break ties by moving to higher degree
            elif distance == best_distance and degree < best_degree:
                best_choice = choice
                best_degree = degree
        return best_choice

    @staticmethod
    def patrol(agent, state):
        """Move to the most central neighbour"""
        graph = state.graph
        # go to the neighbour with highest degree (local centrality)
        neighbours = list(graph.neighbors(agent.position))
        if not neighbours:
            return agent.position
        return MoveController.best_by_centrality(neighbours, graph)

    @staticmethod
    def link(agent, state):
        """Move towards nodes adjacent to any teammate"""
        graph = state.graph
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)
        teammates = [evader for evader in state.evaders if evader != agent]
        
        # Vertices adjacent to all teammates
        teammate_neighbours=[] 
        for teammate in teammates:
            teammate_neighbours.extend(list(graph.neighbors(teammate.position)))

        # If you can, go for a teammate's neighbour immediately
        reachable_targets = [target for target in teammate_neighbours if target in choices]

        # If not immediately reachable, move toward nearest teammate
        if not reachable_targets:
            target = None
            min_distance = float("inf")
            for option in teammate_neighbours:
                distance = MoveController.shortest_distance(graph, agent.position, option)
                if distance < min_distance:
                    min_distance = distance
                    target = option
            
            # Move one step toward target with tie-breaking by degree
            best_distance = float("inf")
            best_choice = agent.position
            best_degree = float("inf")

            for choice in choices:
                if choice not in [teammate.position for teammate in teammates]:
                    distance = MoveController.shortest_distance(graph, choice, target)
                    degree = graph.degree(choice)
                    if distance < best_distance:
                        best_choice = choice
                        best_distance = distance
                        best_degree = degree
                    elif distance == best_distance and degree < best_degree:
                        best_choice = choice
                        best_degree = degree

            return best_choice

        # If already near teammate, pick safest neighbour
        best_choice = agent.position
        best_safety = -1
        for target in reachable_targets:
            if target not in [teammate.position for teammate in teammates]:
                safety = MoveController.nearest_pursuer_distance(graph, target, state.pursuers)
                if safety > best_safety:
                    best_safety = safety
                    best_choice = target

        # Fallback
        if best_choice is None:
            return random.choice(graph.neighbors(agent.position))

        return best_choice

    @staticmethod
    def decoy(agent, state):
        """Move to a suboptimal escape route that remains safe"""
        graph = state.graph
        pursuers = state.pursuers

        # Can only move to a neighbour or stay put
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # Calculate distance from nearest pursuer for each possible move       
        safety_scores = []
        for choice in choices:
            safety_scores.append(MoveController.nearest_pursuer_distance(graph, choice, pursuers))

        # Generate list of moves that are safe but not optimal
        max_safety = max(safety_scores)
        good_moves = []
        for choice, safety in zip(choices, safety_scores):
            if safety > 1 and safety <= max_safety - 1:
                good_moves.append(choice)

        # Randomly choose a move from the list
        if good_moves:
            return random.choice(good_moves)
        else:
            return random.choice(choices) # Fallback

    # Pursuer Strategies - 

    @staticmethod
    def chase(agent, state):
        """Choose the move that gets closest to flag carrier"""
        graph = state.graph
        flag_carrier = state.flag.position()
        
        # Possible moves (including staying put)
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can directly capture the flag, do it
        if flag_carrier in choices:
            return flag_carrier
        
        # Find moves that minimize distance to flag carrier
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1  # prefer high-degree vertices

        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, flag_carrier)
            degree = graph.degree(choice)

            # Pick choice that minimises distance
            if distance < best_distance:
                best_distance = distance
                best_choice = choice
                best_degree = degree
            # break ties using highest degree
            elif distance == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice

    @staticmethod
    def flank(agent, state):
        """Approach the flag from nearby vertices"""
        graph = state.graph
        flag_carrier = state.flag.position()

        # Move to any neighour or stay still
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        if flag_carrier in choices:
            return flag_carrier

        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        # Compute candidate "flank targets" = nodes 1 or 2 steps away from carrier
        targets = set()
        neighbours_1 = set(graph.neighbors(flag_carrier))
        neighbours_2 = set()
        for neighbour in neighbours_1:
            neighbours_2.update(graph.neighbors(neighbour))
        targets = neighbours_1.union(neighbours_2)
        targets.discard(flag_carrier)  # skip carrier's current node

        # Choose the move that gets closest to any potential target
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1

        for choice in choices:
            # distance to closest flank target
            distances = [MoveController.shortest_distance(graph, choice, target) for target in targets]
            if not distances:
                continue
            min_dist = min(distances)
            degree = graph.degree[choice]

            # Prioritise moving closest to target
            if min_dist < best_distance:
                best_distance = min_dist
                best_choice = choice
            # Break ties by choosing highest degree
            elif min_dist == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice


    @staticmethod
    def intercept(agent, state):
        """Predict where the flag carrier will move and target that"""
        graph = state.graph    
        # Predict the flag carriers next position
        flag_carrier = state.flag.carrier
        prediction = MoveController.escape(flag_carrier, state)

        # Can only move to a neighbour or stay put
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can capture the flag, then do it
        if flag_carrier.position in choices:
            return flag_carrier.position
        
        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        # Find moves that minimize distance to prediction
        best_distance = float("inf")
        best_choice = agent.position
        best_degree = -1  # prefer high-degree nodes

        for choice in choices:
            distance = MoveController.shortest_distance(graph, choice, prediction)
            degree = graph.degree(choice)

            # Prefer choices that reduce distance to target
            if distance < best_distance:
                best_distance = distance
                best_choice = choice
                best_degree = degree
            # Break ties using degree
            elif distance == best_distance and degree > best_degree:
                best_choice = choice
                best_degree = degree

        return best_choice

    @staticmethod
    def lurk(agent, state):
        """Shadow flag carrier by aiming to keep distance 3"""
        graph = state.graph
        flag_carrier = state.flag.position()

        # Possible moves (including staying put)
        choices = list(graph.neighbors(agent.position))
        choices.append(agent.position)

        # If you can capture the flag then do it
        if flag_carrier in choices:
            return flag_carrier

        choices = MoveController.avoid_teammates_pursuer(agent, state, choices)

        best_choice = agent.position
        best_diff = float("inf")  # difference from desired distance

        for move in choices:
            distance = MoveController.shortest_distance(graph, move, flag_carrier)
            diff = abs(distance - 2)  # target distance = 2

            # Pick choice that is closest to desired distance (2)
            if diff < best_diff:
                best_diff = diff
                best_choice = move
            # Break ties using highest degree
            elif diff == best_diff and graph.degree(move) > graph.degree(best_choice):
                best_choice = move

        return best_choice

    # Map the role to the method computing the move
    Strategies = {
        # Evaders
        "escape": escape,
        "handoff": handoff,
        "support": support,
        "patrol": patrol,
        "link": link,
        "decoy": decoy,

        # Pursuers
        "chase": chase,
        "flank": flank,
        "intercept": intercept,
        "lurk": lurk,
    }

    def choose_move(self, agent, state):
        """Execute the strategy assigned to the agent returning a valid move"""
        strategy = agent.strategy

        # random fallback
        graph = state.graph
        neighbours = list(graph.neighbors(agent.position))
        if not neighbours:
            return agent.position

        # run the strategy
        if strategy in self.Strategies:
            new_pos = self.Strategies[strategy](agent, state)

            # ensure it's a legal move
            legal_moves = neighbours + [agent.position]
            if new_pos in legal_moves:
                return new_pos

        # fallback
        return random.choice(neighbours)
