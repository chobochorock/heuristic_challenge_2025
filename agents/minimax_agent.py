from pathlib import Path
from random import choice
from typing import List, Literal, Union

from action import *
from board import GameBoard
from time import time, sleep


class Agent:  # Do not change the name of this class!
    """
    An agent class
    """

    # Do not modify this.
    name = Path(__file__).stem

    # Do not change the constructor argument!
    def __init__(self, player: Literal['white', 'black']):
        """
        Initialize the agent

        :param player: Player label for this agent. White or Black
        """
        self.player = player

    def heuristic_search(self, board: GameBoard) -> List[Action]:
        """
        * Complete this function to answer the challenge PART I.

        This function uses heuristic search for finding the best route to the goal line.
        You have to return a list of action, which denotes the shortest actions toward the goal.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Breadth-first search
        - Depth-first search
        - Uniform-cost search
        - Greedy Best-first search
        - A* search
        - IDA*
        - RBFS
        - SMA*

        :param board: The game board with initial game setup.
        :return: A list of actions.
        """
        return []

    def local_search(self, board: GameBoard, time_limit: float) -> Union[MOVE, List[BLOCK]]:
        """
        * Complete this function to answer the challenge PART II.

        This function uses local search for finding the three best place of the fence.
        The system calls your algorithm multiple times, repeatedly.
        Each time, it provides new position of your pawn and asks your next decision
         until time limit is reached, or until you return a BLOCK action.

        Each time you have to decide one of the action.
        - If you want to look around neighborhood places, return MOVE action
        - If you decide to answer the best place, return BLOCK action
        * Note: you cannot move to the position that your opponent already occupied.

        You can use your heuristic search function, which is previously implemented, to compute the fitness score of each place.
        * Note that we will not provide any official fitness function here. The quality of your answer depends on the first part.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Hill-climbing search and its variants
        - Simulated annealing and its variants
        - Tabu search and its variants
        - Greedy Best-first search
        - Local/stochastic beam search (note: parallel execution should be called as sequentially)
        - Evolutionary algorithms
        - Empirical/Stochastic gradient methods
        - Newton-Raphson method

        :param board: The game board with current state.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The next MOVE or list of three BLOCKs.
            That is, you should either return MOVE() action or [BLOCK(), BLOCK(), BLOCK()].
        """
        return [
            BLOCK(player=self.player, edge=(1, 1), orientation='vertical'),
            BLOCK(player=self.player, edge=(2, 2), orientation='vertical'),
            BLOCK(player=self.player, edge=(3, 3), orientation='vertical')
        ]

    def belief_state_search(self, board: GameBoard, time_limit: float) -> List[Action]:
        """
        * Complete this function to answer the challenge PART III.

        This function uses belief state search for finding the best move for certain amount of time.
        The system calls your algorithm only once. Your algorithm should consider the time limit.

        You can use your heuristic search or local search function, which is previously implemented, to compute required information.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - AND-OR search and its variants
        - Heuristic/Uninformed search algorithms whose state is actually a belief state.
        - Online DFS algorithm, or other online variant of heuristic/uninformed search.
        - LRTA*

        :param board: The game board with initial game setup.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The next move.
        """
        return [BLOCK(self.player, edge=(1,1), orientation='horizontal'),
                BLOCK(self.player, edge=(7,7), orientation='horizontal'),
                BLOCK(self.player, edge=(3,3), orientation='horizontal'),
                BLOCK(self.player, edge=(6,6), orientation='horizontal'),]

    def adversarial_search(self, board: GameBoard, time_limit: float) -> Action:
        """
        * Complete this function to answer the challenge PART IV.

        This function uses adversarial search to win the game.
        The system calls your algorithm whenever your turn arrives.
        Each time, it provides new position of your pawn and asks your next decision until time limit is reached.

        You can use your search function, which is previously implemented, to compute relevant information.

        RESTRICTIONS: USE one of the following algorithms or its variant.
        - Minimax algorithm, H-minimax algorithm, and Expectminimax algorithm
        - RBFS search
        - Alpha-beta search and heuristic version of it.
        - Pure Monte-Carlo search
        - Monte-Carlo Tree Search and its variants
        - Minimax search with belief states
        - Alpha-beta search with belief states

        :param board: The game board with current state.
        :param time_limit: The time limit for the search. Datetime.now() should have lower timestamp value than this.
        :return: The next move.
        """
        # heuristic layer
        #efficient moving    
        def path_finding(player) : 
            def heur(path : list) :
                move = path[-1]
                if   board.get_player_id() == 'white' : heur = 8 - move[0]
                elif board.get_player_id() == 'black' : heur = move[0]
                return heur
            
            def eval(path) : return len(path) + heur(path)
            
            def find_min_frontier() :
                min = None
                for i in frontier:
                    if min == None or i['priority'] < min['priority'] : min = i

                if min == None : return 0
                return frontier.index(min)

            init_state = board.get_initial_state()
            init_coordinate = tuple(init_state['player'][player]['pawn'])
            visited = {init_coordinate : 0}
            frontier = []
            for move in board.get_applicable_moves(player) :
                frontier.append({'priority' : eval([move]), 'moves' : [move]})
                visited[move] = len([move])
            
            while frontier : 
                current_frontier = frontier.pop(find_min_frontier())
                if not current_frontier['moves'] == [] :
                    board.simulate_action(
                        init_state,
                        *[MOVE(player, move) for move in current_frontier['moves']],
                    )

                if board.is_game_end() : 
                    return [MOVE(player, move) for move in current_frontier['moves']]
                
                current_applicable_moves = board.get_applicable_moves(player)
                for move in current_applicable_moves :
                    c = len(current_frontier['moves'])
                    if move in visited and visited[move] < c : continue
                    visited[move] = c
                    new_moves = current_frontier['moves'][:] + [move]
                    frontier.append({'priority' : eval(new_moves), 'moves' : new_moves})

        def fence_finding(player, opponent_path=None) :
            # heuristic : fence is good to be placed at behind of player
            # consider, when moving is good? or installing fences? if so many fences are used before we get in middle game or end game, we may suffer unilaterally from opponent's strategy
            # horizantal : Even if opponent is close or far, it is effective when it is placed
            # vertical : only efficient but powerful when the opponent is close to its placement
            '''fences that blocks the opponents path and its position is behind of player
            or ...'''
            
            fences = board.get_applicable_fences(player=player)
            opponent = 'white' if player == 'black' else 'black'
            if opponent_path is None : opponent_path = path_finding(opponent)
            player_pos = board.get_position(self.player)
            opponent_pos = board.get_position(self.opponent)

            for fence in fences :
                coord, orient = fence
                if (player == 'black' and coord[0] < opponent_pos[0]-1 and coord[0] < player_pos[0]  ) \
                or (player == 'white' and opponent_pos[0] > coord[0]   and player_pos[0]-1 > coord[0]) : 
                    # 장벽이 플레이어의 뒤 상대한테는 앞에 있게 두는 경우 가산점 (또는 그 경우만 고려)
                    pass
                else : continue # 이 코드도 고려해보는 것이 좋음 (위 조건에 좋은 상황만 고려)

                # 이득 계산(경로를 실행하면서, 상대의 거리증감을 계산)
            
            return fences
        
        #minimax algorithm
        INF = 100000
        self.opponent = 'white' if self.player == 'black' else 'black'
        init_player_path   = path_finding(self.player)
        init_opponent_path = path_finding(self.opponent)
        init_player_pos = board.get_position(self.player)
        init_opponent_pos = board.get_position(self.opponent)

        def utility() : # 형세판단
            if board.is_game_end() : 
                if board.current_player == self.player : return INF
                else :                                   return -INF
            else : # depth limit -> heuristic required
                player_util   = len(path_finding(self.player))
                opponent_util = len(path_finding(self.opponent))
                return player_util - opponent_util

        def get_actions(player) :
            # fences = [BLOCK(player, edge=coord, orientation=orient) 
            #           for coord, orient in board.get_applicable_fences(player=player)]
            # moves  = [MOVE(player, position) 
            #           for position in board.get_applicable_moves(player=player)]
            
            # need to add heuristic priority
            moves  =  path_finding(player)[0]
            fences = fence_finding(player)
            return moves + fences

        def MaxValue(state, depth=1) :
            board.set_to_state(state)
            if board.is_game_end() or depth > 6 : return utility(), None
            v, move = (-INF, None)
            for act in get_actions(self.player) : 
                v2, a2 = MinValue(board.simulate_action(state, act))
                if v2 > v : v, move = v2, act
            return v, move

        def MinValue(state, depth=1) :
            board.set_to_state(state)
            if board.is_game_end() or depth > 6 : return utility(), None
            v, move = (INF, None)
            for act in get_actions(self.opponent) : 
                v2, a2 = MaxValue(board.simulate_action(state, act))
                if v2 < v : v, move = v2, act
            return v, move

        value, action = MaxValue(board.get_state())

        return action

