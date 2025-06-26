from pathlib import Path
from random import choice, sample
from typing import List, Literal, Union

from action import *
from board import GameBoard

from time import time
import math


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
        class Node : 
            def __init__(self, state : dict, player, parent=None, depth=0):
                self.children = []
                self.winning  = 0 # player가 이긴 횟수
                self.rollout  = 0 # 시도된 횟수 
                self.parent   = parent
                self.state    = state
                self.player   = player # 현 상태의 플레이어 : rollout시 사용
                self.actions  = considerable_action(self.state, self.player)
                self.depth    = depth # 디버깅용

        def opponent_of(player) : return 'white' if player == 'black' else 'black'

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

            init_state      = state
            init_coordinate = tuple(init_state['player'][player]['pawn'])
            visited         = {init_coordinate : 0}
            frontier        = []
            for move in board.get_applicable_moves(player) :
                frontier.append({'priority' : eval([move]), 'moves' : [move]})
                visited[move] = len([move])
            
            while frontier : 
                current_frontier = frontier.pop(find_min_frontier())
                if not current_frontier['moves'] == [] :
                    try : 
                        board.simulate_action(
                            init_state,
                            *[MOVE(player, move) for move in current_frontier['moves']],
                        )
                    except : continue

                if board.is_game_end() : 
                    return [MOVE(player, move) for move in current_frontier['moves']]
                
                current_applicable_moves = board.get_applicable_moves(player)
                for move in current_applicable_moves :
                    c = len(current_frontier['moves'])
                    if move in visited and visited[move] < c : continue
                    visited[move] = c
                    new_moves = current_frontier['moves'][:] + [move]
                    frontier.append({'priority' : eval(new_moves), 'moves' : new_moves})
        
        def fence_finding(state, player) :
            '''설치하기 적당한 위치를 탐색함'''
            def fence_block_path(player, path) : 
                # 상대의 진로를 막는 장벽을 찾음
                fences = []
                for i in range(len(path)-1) : # path가 none인 경우가 존재
                    init_pos, end_pos = path[i].position, path[i+1].position
                    if   init_pos[0] - end_pos[0] == -1 : 
                        fences.append(BLOCK(player, (init_pos[0]-1, init_pos[1]-1), 'horizontal'))
                        fences.append(BLOCK(player, (init_pos[0]-1, init_pos[1]  ), 'horizontal'))
                    elif init_pos[0] - end_pos[0] ==  1 : 
                        fences.append(BLOCK(player, (init_pos[0]  , init_pos[1]-1), 'horizontal'))
                        fences.append(BLOCK(player, (init_pos[0]  , init_pos[1]  ), 'horizontal'))
                    elif init_pos[1] - end_pos[1] == -1 : 
                        fences.append(BLOCK(player, (init_pos[0]-1, init_pos[1]-1), 'vertical'))
                        fences.append(BLOCK(player, (init_pos[0]  , init_pos[1]-1), 'vertical'))
                    elif init_pos[1] - end_pos[1] ==  1 : 
                        fences.append(BLOCK(player, (init_pos[0]-1, init_pos[1]  ), 'vertical'))
                        fences.append(BLOCK(player, (init_pos[0]  , init_pos[1]  ), 'vertical'))
                return fences

            board.set_to_state(state)
            fences = []
            opponent = opponent_of(player)
            opponent_path = path_finding(state, opponent)
            
            board.set_to_state(state)
            play_pos = board.get_position(player)
            oppo_pos = board.get_position(opponent)
            if board.number_of_fences_left(player) == 0 : return fences

            for fence in board.get_applicable_fences(player) :
                fence = BLOCK(player, fence[0], fence[1])
                # 이 부분에서 고려할 만한 장벽의 조건을 제시해 줘야 함
                # if fence in fence_block_path(player, opponent_path) :
                target_edges = {(f.edge, f.orientation) for f in fence_block_path(player, opponent_path)}
                if (fence.edge, fence.orientation) in target_edges:
                    fence_pos = fence.edge
                    if fence.orientation == 'horizontal' : 
                        # 나 뒤에 상대 앞에 : 
                        if   player == 'black' \
                            and play_pos[0]-1 < fence_pos[0] \
                            and oppo_pos[0]   < fence_pos[0] : 
                            fences.append(fence)
                        elif player == 'white' \
                            and play_pos[0]   > fence_pos[0] \
                            and oppo_pos[0]-1 > fence_pos[0] : 
                            fences.append(fence)
                    else : # fence.orientation == 'vertical'
                        # 상대 바로 옆에 : 
                        if      0 <= (oppo_pos[0] - fence_pos[0]) <= 1 \
                            and 0 <= (oppo_pos[1] - fence_pos[1]) <= 1 :
                            fences.append(fence)
                else : # 마지막에 자신의 길을 트기 위한 장벽을 탐색하는 것을 고려
                    # 내 앞에 : 
                    continue
                
            return fences
        
        def considerable_action(state, player) :
            '''선택할 만한 수를 선별함'''
            move   =  path_finding(state, player)[0]
            fences = fence_finding(state, player)
            if fences == [] : 
                app_fences = board.get_applicable_fences(player)
                fences = sample(app_fences, min(4, len(app_fences)))
                fences = [BLOCK(player, fence[0], fence[1]) for fence in fences]
            return [move] + fences

        def MCTS() :
            def Select(tree : Node) : 
                '''
                트리에서 노드를 선택하는 과정
                '''
                def UCB1(node : Node) :
                    if node.rollout == 0 or node.parent.rollout == 0 : return float('inf') # 이게 inf여야 하는지 확인 필요
                    c = 2 # param : 이상적인 수는 2
                    return node.winning / node.rollout + math.sqrt(c * math.log(node.parent.rollout) / node.rollout) # math domain error
                
                print('get in select process')
                present = tree 
                while present.children : 
                    # print(f'present actions : {present.actions}')
                    # if present.actions : break
                    selected = max(present.children, key=lambda n : UCB1(n))
                    # if UCB1(selected) < UCB1(present) : return present
                    present  = selected
                print('selected the leaf')
                return present

            def Expand(selected : Node) :
                '''
                노드의 상황을 시뮬레이션 한 뒤 다음 (적절한) 행동을 무작위로 선택하는 과정
                '''
                def create_from(node : Node) : 
                    # if len(node.children) >= 5 : return node # 속도가 느리다 싶을 때 5개나 그 이하로 child 제약
                    opponent  = opponent_of(selected.player)
                    for action in node.actions : 
                        new_state = board.simulate_action(node.state, action) # simulate with node.state
                        new_node  = Node(new_state, opponent, node, node.depth + 1)
                        node.children.append(new_node)
                    return choice(node.children)
                
                print('get in expand process')
                if selected.actions : return create_from(selected)
                else                : return selected

            def RolOut(expanded : Node) :
                '''
                무작위 action을 시행하는 과정 : action은 게임 끝까지 하는 것이 원칙이나, 
                heuristic을 사용하면 확률적으로 계산 가능 
                예) 시그모이드((상대의 거리 + 나의 장벽 수) - (나의 거리 + 상대의 장벽 수)) => 승률
                '''
                def utility(player) : 
                    player_distance    = len(path_finding(state, player))
                    opponent_distance  = len(path_finding(state, opponent_of(player)))
                    player_num_fence   = board.number_of_fences_left(player)
                    opponent_num_fence = board.number_of_fences_left(opponent_of(player))
                    w = 2 # 장벽이 거리에 영향을 끼치는 것에 대한 가중치
                    weight = opponent_distance - player_distance + w * (player_num_fence - opponent_num_fence)
                    print(f'rollout utility : {1 / (1 + 1 / math.exp(weight))}')
                    return 1 / (1 + 1 / math.exp(weight)) # player 기준 승률 평가

                print('get in rollout')
                state = expanded.state
                board.set_to_state(state)
                depth_limit = 10
                turn_player = expanded.player
                for i in range(depth_limit) : 
                    if board.is_game_end() : return 0 if expanded.player == turn_player else 1 # expanded.player를 사용하는 것을 고려
                    action      = choice(considerable_action(state, turn_player))
                    # print(f'turn player actions : {considerable_action(state, turn_player)}')
                    state       = board.simulate_action(state, action)
                    turn_player = opponent_of(turn_player)

                return utility(expanded.player) # 이거를 어떻게 제대로 적용해야 할지 고민해봐야 함

            def BackPropagate(result, expanded : Node) :
                '''
                시행의 결과를 역전파하는 과정
                '''
                present = expanded
                while present.parent is not None : 
                    print(f'backpropagate depth {present.depth} : {present.winning} / {present.rollout}')
                    present.rollout += 1
                    if present.player == expanded.player : present.winning += result
                    else                                 : present.winning += 1 - result
                    present = present.parent

            init_time = time()
            treeroot = Node(board.get_state(), self.player) # 이것이 트리의 헤드
            while time() - init_time < 55 : 
                selected = Select(treeroot)
                expanded = Expand(selected)
                result   = RolOut(expanded)
                BackPropagate(result, expanded)
            
            return max(treeroot.children, key=lambda x : x.winning / x.rollout if x.rollout != 0 else -1) # children중에서 최고를 찾아야 함 max(treeroot.children, key=lambda x : x.winning / x.rollout)


        # action = path_finding(board.get_state(), self.player)[0]
        action = MCTS()

        return action

