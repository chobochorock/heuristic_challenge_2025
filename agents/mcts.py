from pathlib import Path
from random import choice
from typing import List, Literal, Union

from action import *
from board import GameBoard


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
        import numpy as np
        from collections import defaultdict
        from abc import ABC, abstractmethod


        class MonteCarloTreeSearchNode(ABC):

            def __init__(self, state, parent=None):
                """
                Parameters
                ----------
                state : mctspy.games.common.TwoPlayersAbstractGameState
                parent : MonteCarloTreeSearchNode
                """
                self.state = state
                self.parent = parent
                self.children = []

            @property
            @abstractmethod
            def untried_actions(self):
                """

                Returns
                -------
                list of mctspy.games.common.AbstractGameAction

                """
                pass

            @property
            @abstractmethod
            def q(self):
                pass

            @property
            @abstractmethod
            def n(self):
                pass

            @abstractmethod
            def expand(self):
                pass

            @abstractmethod
            def is_terminal_node(self):
                pass

            @abstractmethod
            def rollout(self):
                pass

            @abstractmethod
            def backpropagate(self, reward):
                pass

            def is_fully_expanded(self):
                return len(self.untried_actions) == 0

            def best_child(self, c_param=1.4):
                choices_weights = [
                    (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
                    for c in self.children
                ]
                return self.children[np.argmax(choices_weights)]

            def rollout_policy(self, possible_moves):        
                return possible_moves[np.random.randint(len(possible_moves))]

        class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

            def __init__(self, state, parent=None):
                super().__init__(state, parent)
                self._number_of_visits = 0.
                self._results = defaultdict(int)
                self._untried_actions = None

            @property
            def untried_actions(self):
                if self._untried_actions is None:
                    self._untried_actions = self.state.get_legal_actions()
                return self._untried_actions

            @property
            def q(self):
                wins = self._results[self.parent.state.next_to_move]
                loses = self._results[-1 * self.parent.state.next_to_move]
                return wins - loses

            @property
            def n(self):
                return self._number_of_visits

            def expand(self):
                action = self.untried_actions.pop()
                next_state = self.state.move(action)
                child_node = TwoPlayersGameMonteCarloTreeSearchNode(
                    next_state, parent=self
                )
                self.children.append(child_node)
                return child_node

            def is_terminal_node(self):
                return self.state.is_game_over()

            def rollout(self):
                current_rollout_state = self.state
                while not current_rollout_state.is_game_over():
                    possible_moves = current_rollout_state.get_legal_actions()
                    action = self.rollout_policy(possible_moves)
                    current_rollout_state = current_rollout_state.move(action)
                return current_rollout_state.game_result

            def backpropagate(self, result):
                self._number_of_visits += 1.
                self._results[result] += 1.
                if self.parent:
                    self.parent.backpropagate(result)

        class MonteCarloTreeSearch(object):

            def __init__(self, node):
                """
                MonteCarloTreeSearchNode
                Parameters
                ----------
                node : mctspy.tree.nodes.MonteCarloTreeSearchNode
                """
                self.root = node

            def best_action(self, simulations_number=None, total_simulation_seconds=None):
                """

                Parameters
                ----------
                simulations_number : int
                    number of simulations performed to get the best action

                total_simulation_seconds : float
                    Amount of time the algorithm has to run. Specified in seconds

                Returns
                -------

                """

                if simulations_number is None :
                    assert(total_simulation_seconds is not None)
                    end_time = time.time() + total_simulation_seconds
                    while True:
                        v = self._tree_policy()
                        reward = v.rollout()
                        v.backpropagate(reward)
                        if time.time() > end_time:
                            break
                else :
                    for _ in range(0, simulations_number):            
                        v = self._tree_policy()
                        reward = v.rollout()
                        v.backpropagate(reward)
                # to select best child go for exploitation only
                return self.root.best_child(c_param=0.)

            def _tree_policy(self):
                """
                selects node to run rollout/playout for

                Returns
                -------

                """
                current_node = self.root
                while not current_node.is_terminal_node():
                    if not current_node.is_fully_expanded():
                        return current_node.expand()
                    else:
                        current_node = current_node.best_child()
                return current_node
        
        return action

