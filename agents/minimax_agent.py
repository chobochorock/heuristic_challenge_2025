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
        # import logging
        # from argparse import ArgumentParser
        # argparser = ArgumentParser()
        # argparser.set_defaults(debug=False)
        # argparser.add_argument('-p', '-players', '--players', type=str, nargs='+',
        #                     help='Players to compete (Program will evaluate your players as a league match)')
        # argparser.add_argument('--debug', action='store_true', dest='debug',
        #                     help='Enable debug mode')
        # args = argparser.parse_args()
        # logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
        #                 format='%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s',
        #                 filename='execution.log',
        #                 # Also, the output will be logged in 'execution.log' file.
        #                 filemode='w+',
        #                 force=True, encoding='UTF-8')
        # _logger = logging.getLogger('heurisitc_agent_check')
        # _logger.debug(f'player utility : ')



        # heuristic layer
        # efficient moving    
        def path_finding(state, player) : 
            '''A* 알고리즘으로 작성된 최적 경로를를 찾는 알고리즘이다. '''
            def heur(path : list) :
                move = path[-1]
                if   player == 'white' : heur = 8 - move[0]
                elif player == 'black' : heur = move[0]
                return heur
            
            def eval(path) : return len(path) + heur(path)
            
            def find_min_frontier() :
                min = None
                for i in frontier:
                    if min == None or i['priority'] < min['priority'] : min = i

                if min == None : return 0
                return frontier.index(min)

            init_state = state
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

        def fence_finding(state, player, opponent_path=None) -> List[BLOCK] :
            # heuristic : fence is good to be placed at behind of player
            # consider, when moving is good? or installing fences? if so many fences are used before we get in middle game or end game, we may suffer unilaterally from opponent's strategy
            # horizantal : Even if opponent is close or far, it is effective when it is placed
            # vertical : only efficient but powerful when the opponent is close to its placement
            # fences that blocks the opponents path and its position is behind of player or ...
            '''휴리스틱을 기반으로 설치할 만한 장벽을 계산한다. 예를 들어 현재 휴리스틱은 다음과 같다.

            :heuristic 1: 장벽의 위치가 자신보다 뒤에이면서 상대보다 앞에 있게 하면 채택 (이렇게 하면, 상대는 반드시 해당 장벽에 영향을 받고 목표지점에 도달해야 한다)
            :heuristic 2: 해당 장벽이 상대의 최적 경로를 막는다면 채택 (이렇게 하면, 상대는 반드시 이 장벽을 피해야 하기에 이동거리에 손해 가능성이 크다다)
            :else: 이후에 추가하시거나 구현할 수 있다. 
            '''
            board.set_to_state(state)
            opponent = 'white' if player == 'black' else 'black'
            if opponent_path is None : opponent_path = path_finding(state, opponent)
            player_pos = board.get_position(self.player)
            opponent_pos = board.get_position(self.opponent)
            blocking_fences = []
            for fence in board.get_applicable_fences(player=player) :
                coord, orient = fence
                if (player == 'black' and coord[0] > opponent_pos[0]   and coord[0] > player_pos[0]-1) \
                or (player == 'white' and opponent_pos[0]-1 > coord[0] and player_pos[0] > coord[0]) : 
                    # 장벽이 플레이어의 뒤 상대한테는 앞에 있게 두는 경우 가산점 (또는 그 경우만 고려)
                    pass
                else : continue # 이 코드도 고려해보는 것이 좋음 (위 조건에 좋은 상황만 고려)

                # 이 장벽이 상대의 길을 가로막는지 확인 (상대의 최적 경로를 막을 수 있다면 좋응)
                try : # 잘 안돌아 가는 것 같음
                    # 장벽이 설치된 이후에 상대가 이전 최적 경로를 지날 수 있는지 확인한다. 
                    coord, orient = fence
                    board.simulate_action(state, *([BLOCK(player, coord, orient)] + opponent_path))
                except : 
                    # 지나지 못한다면(예외가 발생한다면면), 경로를 막는 장벽이므로 합격!
                    blocking_fences.append(fence)
            
            board.set_to_state(state)
            return blocking_fences
        
        #minimax algorithm
        INF = 100000
        self.opponent = 'white' if self.player == 'black' else 'black'
        init_state = board.get_state()

        def utility(state) : 
            '''
            상대의 비용과 자신의 비용을 휴리스틱을 포함하여 계산

            :heuristic: 자신의 최적경로의 거리와 상대의 장벽 갯수가 자신의 비용으로 책정된다. 둘 사이에 가중치(w)는 임의로 조정해야 한다. 
            '''
            # depth limit -> heuristic required
            player_fence_num   = board.number_of_fences_left(self.player)
            opponent_fence_num = board.number_of_fences_left(self.opponent)
            player_path_len    = len(path_finding(state, self.player))
            opponent_path_len  = len(path_finding(state, self.opponent))

            # 장벽이 줄수 있는 피해는 w 턴정도라고 예측
            w = 1
            player_cost   = player_path_len   + w * opponent_fence_num
            opponent_cost = opponent_path_len + w * player_fence_num
            return opponent_cost - player_cost

        def get_actions(state, player) -> List[Action] :
            board.set_to_state(state)
            # fences = [BLOCK(player, edge=coord, orientation=orient) 
            #           for coord, orient in board.get_applicable_fences(player=player)]
            # moves  = [MOVE(player, position) 
            #           for position in board.get_applicable_moves(player=player)]
            
            # need to add heuristic priority
            # 각각이 휴리스틱을 적용한 장벽과 이동의 리스트이다. 
            moves  = [path_finding(state, player)[0]]
            fences = fence_finding(state, player)
            # assert False, f'{moves + fences}'
            if moves + fences is None : return choice(board.get_applicable_fences() + board.get_applicable_moves())
            return moves + fences

        def MaxValue(state, depth=1) -> tuple[int, Action] : # player turn
            board.set_to_state(state)
            if board.is_game_end() : return -INF,      None
            if depth > 10 :          return utility(state), None

            v, move = (-INF, None)
            applicable_actions = get_actions(state, self.player)
            for act in applicable_actions :
                v2, a2 = MinValue(board.simulate_action(state, act), depth + 1)
                if v2 > v : v, move = v2, act

            if move is None : return v, choice(get_actions(state, self.player))
            return v, move

        def MinValue(state, depth=1) -> tuple[int, Action] : # opponent turn
            board.set_to_state(state)
            if board.is_game_end() : return INF,       None
            if depth > 10 :          return utility(state), None

            v, move = (INF, None)
            applicable_actions = get_actions(state, self.opponent)
            for act in applicable_actions : 
                v2, a2 = MaxValue(board.simulate_action(state, act), depth + 1)
                if v2 < v : v, move = v2, act
            
            if move is None : return v, choice(get_actions(state, self.opponent))
            return v, move

        result = MaxValue(board.get_state())
        action = result[1]
        # assert False, f'{action}'
        try :
            action = MOVE(action.player, action.position)
        except :
            action = BLOCK(action.player, action.edge, action.orientation)
        
        board.set_to_state(init_state)

        return action

