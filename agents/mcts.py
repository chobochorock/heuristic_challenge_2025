from pathlib import Path
from random import choice
from typing import List, Literal, Union

from action import *
from board import GameBoard
from collections import defaultdict
import numpy as np
import time

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
        주어진 게임 상태에서 Monte Carlo Tree Search(MCTS)를 사용하여 최적의 행동을 찾는 함수.
        """
        # MCTS에서 사용되는 각 노드를 나타내는 클래스
        class MCTSNode:
            def __init__(self, state, parent=None):
                """
                노드 초기화
                :param state: 이 노드에서의 게임 상태
                :param parent: 부모 노드
                """
                self.state = state  # 현재 게임 상태를 저장
                self.parent = parent  # 부모 노드에 대한 참조
                self.children = []  # 자식 노드를 저장하는 리스트
                self._number_of_visits = 0  # 이 노드가 방문된 횟수
                self._results = defaultdict(int)  # 결과를 저장 (승리/패배 기록)
                self._untried_actions = None  # 아직 시도하지 않은 행동들

            @property
            def untried_actions(self):
                """
                아직 시도하지 않은 행동을 반환.
                처음 호출될 때, 가능한 행동 목록을 초기화.
                """
                if self._untried_actions is None:
                    self._untried_actions = self.state.get_legal_actions()
                return self._untried_actions

            @property
            def q(self):
                """
                이 노드에서 얻은 점수(승리 - 패배).
                부모 노드의 플레이어를 기준으로 계산.
                """
                wins = self._results[self.parent.state.current_player]  # 부모 플레이어의 승리 수
                losses = self._results[-1 * self.parent.state.current_player]  # 부모 플레이어의 패배 수
                return wins - losses

            @property
            def n(self):
                """
                이 노드가 방문된 횟수.
                """
                return self._number_of_visits

            def expand(self):
                """
                아직 시도하지 않은 행동을 기반으로 자식 노드를 생성.
                새로운 상태로 이동한 자식 노드를 반환.
                """
                action = self.untried_actions.pop()  # 아직 시도하지 않은 행동 중 하나를 선택
                next_state = self.state.move(action)  # 선택한 행동을 적용한 새로운 상태 생성
                child_node = MCTSNode(next_state, parent=self)  # 새로운 자식 노드 생성
                self.children.append(child_node)  # 자식 리스트에 추가
                return child_node

            def is_terminal_node(self):
                """
                현재 노드가 게임 종료 상태인지 확인.
                """
                return self.state.is_game_over()

            def rollout(self):
                """
                현재 상태에서 무작위로 게임을 진행하여 종료 상태까지 도달.
                결과를 반환.
                """
                current_rollout_state = self.state
                while not current_rollout_state.is_game_over():  # 게임 종료 상태가 될 때까지
                    possible_moves = current_rollout_state.get_legal_actions()  # 가능한 행동 목록
                    action = np.random.choice(possible_moves)  # 무작위로 행동 선택,, 고민해봐야할 문제
                    current_rollout_state = current_rollout_state.move(action)  # 선택한 행동 적용
                return current_rollout_state.game_result  # 게임 결과 반환 (승리/패배/무승부)

            def backpropagate(self, result):
                """
                시뮬레이션 결과를 부모 노드로 전파.
                :param result: 시뮬레이션 결과 (승리/패배)
                """
                self._number_of_visits += 1  # 방문 횟수 증가
                self._results[result] += 1  # 결과 업데이트
                if self.parent:
                    self.parent.backpropagate(result)  # 부모 노드로 결과 전파

            def best_child(self, c_param=1.4):
                """
                UCB1 공식을 사용하여 가장 좋은 자식 노드를 선택.
                :param c_param: 탐험과 이용 간의 균형을 조절하는 상수
                """
                choices_weights = [
                    (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
                    for c in self.children
                ]
                return self.children[np.argmax(choices_weights)]  # 가장 높은 값을 가진 자식 반환

        # MCTS 알고리즘 클래스
        class MCTS:
            def __init__(self, node):
                """
                MCTS 초기화
                :param node: 루트 노드 (현재 상태에서 시작)
                """
                self.root = node

            def best_action(self, total_simulation_seconds):
                """
                주어진 시간 동안 시뮬레이션을 수행하여 최적의 행동을 반환.
                :param total_simulation_seconds: 시뮬레이션에 사용할 최대 시간(초 단위)
                """
                end_time = time.time() + total_simulation_seconds  # 종료 시간 계산
                while time.time() < end_time:  # 시간 내에 반복
                    v = self._tree_policy()  # 탐색 정책에 따라 노드 선택
                    reward = v.rollout()  # 선택한 노드에서 시뮬레이션 진행
                    v.backpropagate(reward)  # 시뮬레이션 결과를 전파
                return self.root.best_child(c_param=0.0)  # 최종적으로 가장 좋은 자식 선택

            def _tree_policy(self):
                """
                탐색할 노드를 선택.
                :return: 선택된 노드
                """
                current_node = self.root
                while not current_node.is_terminal_node():  # 터미널 노드가 아닐 때까지
                    if not current_node.is_fully_expanded():  # 모든 행동을 시도하지 않았다면
                        return current_node.expand()  # 노드를 확장
                    else:
                        current_node = current_node.best_child()  # 가장 좋은 자식으로 이동
                return current_node

        # MCTS 실행
        root_node = MCTSNode(board.get_state())  # 루트 노드 생성
        mcts = MCTS(root_node)  # MCTS 초기화
        best_node = mcts.best_action(time_limit - time.time())  # 최적의 행동 계산
        return best_node.state.last_action  # 선택된 행동 반환



