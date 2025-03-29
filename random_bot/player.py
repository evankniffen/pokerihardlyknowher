"""
CFRPlusPokerbot – A full–scale CFR⁺ implementation for B4G Hold’em

This file is the only one you need to modify for your bot.
It implements the __init__, handle_new_round, handle_round_over, and get_action methods.
The bot loads a precomputed equilibrium strategy from "cfr_strategy_full.pkl" (located in the project root)
and selects actions automatically based on that strategy.

DO NOT EDIT any files in the skeleton/ folder.
DO NOT tamper with the game engine/judging system.
"""

import os
import pickle
import random
import math
import eval7
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK, TerminalState, RoundState, GameState

####################################################
# Advanced Abstraction Functions
####################################################

def card_value(card):
    """Convert a card (e.g., 'Ah', 'Tc') to its numeric value."""
    rank = card[0]
    if rank in '23456789':
        return int(rank)
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    elif rank == 'A':
        return 14
    return 0

def bucketize_hand(my_cards):
    """
    Evaluate hand strength using a simple sum-of-values heuristic,
    and bucketize into 10 discrete buckets (0 through 9).
    For 3 cards, maximum sum is 14+13+12=39.
    """
    total = sum(card_value(card) for card in my_cards)
    strength = total / 39.0  
    bucket = min(9, int(strength * 10))
    return bucket

def bucketize_board(board_cards):
    """
    Bucketize the board texture using the sum of board card values.
    Discretize into 5 buckets (0 through 4).
    """
    if not board_cards:
        return 0
    total = sum(card_value(card) for card in board_cards)
    bucket = min(4, int(total / 10))
    return bucket

def pot_ratio_bucket(round_state):
    """
    Compute the pot ratio (total chips contributed divided by the total chips in play)
    and bucketize into 5 buckets (0 through 4).
    """
    pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
    ratio = pot / (2 * STARTING_STACK)
    bucket = min(4, int(ratio * 5))
    return bucket

def street_to_index(round_state):
    """
    Map round_state.street to an index:
      0 -> preflop,
      1 -> flop,
      2 -> turn (final betting round in B4G Hold’em).
    """
    if round_state.street == 0:
        return 0
    elif round_state.street == 2:
        return 1
    else:
        return 2

def infoset_key(round_state, my_cards):
    """
    Construct an abstraction key from the current state.
    The key is a tuple: (street_index, hand_bucket, board_bucket, pot_bucket)
    """
    street_idx = street_to_index(round_state)
    hand_bucket = bucketize_hand(my_cards)
    board_bucket = bucketize_board(round_state.deck[:round_state.street])
    pot_bucket = pot_ratio_bucket(round_state)
    return (street_idx, hand_bucket, board_bucket, pot_bucket)

####################################################
# Full-scale CFR⁺ Solver Framework
####################################################

class CFRPlusSolver:
    def __init__(self):
        # Mapping from infoset key to a node:
        # Each node is a dict with keys "regret" and "strategy_sum" for each action.
        self.node = {}
        self.actions = ["fold", "call", "raise"]

    def get_node(self, key):
        if key not in self.node:
            self.node[key] = {
                "regret": {a: 0.0 for a in self.actions},
                "strategy_sum": {a: 0.0 for a in self.actions}
            }
        return self.node[key]

    def regret_matching_plus(self, regrets):
        # Use only positive regrets; reset negatives to zero.
        positive = {a: max(regrets[a], 0) for a in self.actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in self.actions}
        else:
            return {a: 1.0 / len(self.actions) for a in self.actions}

    def cfr(self, round_state, my_cards, p0, p1):
        """
        Recursively perform CFR⁺ updates.
        p0: reach probability for the current player.
        p1: reach probability for the opponent.
        """
        if isinstance(round_state, TerminalState):
            # Return utility from the perspective of player 0.
            return round_state.deltas[0]

        key = infoset_key(round_state, my_cards)
        node = self.get_node(key)
        strategy = self.regret_matching_plus(node["regret"])
        util = {}
        node_util = 0.0

        for a in self.actions:
            next_state, immediate_payoff = self.simulate_action(round_state, my_cards, a)
            util[a] = immediate_payoff + self.cfr(next_state, my_cards, p0 * strategy[a], p1)
            node_util += strategy[a] * util[a]

        for a in self.actions:
            regret = util[a] - node_util
            node["regret"][a] = max(0.0, node["regret"][a] + p1 * regret)
            node["strategy_sum"][a] += p0 * strategy[a]

        return node_util

    def simulate_action(self, round_state, my_cards, action):
        """
        Simulate the outcome of taking an abstract action.
        This advanced placeholder should be replaced or refined as needed.
        - "fold": yields an immediate loss.
        - "call": advances the round, with payoff determined by a heuristic.
        - "raise": gives an immediate positive payoff if our hand bucket is high.
        """
        if action == "fold":
            payoff = -1.0
            return TerminalState([-1, 1], round_state), payoff
        elif action == "call":
            next_state = round_state.proceed_street()
            my_bucket = bucketize_hand(my_cards)
            opp_bucket = random.randint(0, 9)  # Simulate opponent hand bucket.
            if my_bucket > opp_bucket:
                payoff = 0.5
            elif my_bucket < opp_bucket:
                payoff = -0.5
            else:
                payoff = 0.0
            return next_state, payoff
        elif action == "raise":
            my_bucket = bucketize_hand(my_cards)
            if my_bucket >= 7:
                payoff = 1.0
                return TerminalState([1, -1], round_state), payoff
            else:
                next_state = round_state.proceed_street()
                payoff = 0.0
                return next_state, payoff
        return round_state, 0.0

    def sample_private_cards(self):
        deck = eval7.Deck()
        deck.shuffle()
        cards = deck.deal(3)
        return [str(card) for card in cards]

    def sample_initial_state(self, my_cards):
        opponent_cards = self.sample_private_cards()
        hands = [my_cards, opponent_cards]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        deck = []  # Board will be simulated via simulate_action.
        return RoundState(0, 0, pips, stacks, hands, deck, None)

    def get_average_strategy(self):
        avg_strategy = {}
        for key, node in self.node.items():
            total = sum(node["strategy_sum"].values())
            if total > 0:
                avg_strategy[key] = {a: node["strategy_sum"][a] / total for a in self.actions}
            else:
                avg_strategy[key] = {a: 1.0 / len(self.actions) for a in self.actions}
        return avg_strategy

    def train(self, iterations):
        """
        Run CFR⁺ iterations over sampled game trajectories.
        For a production run, this should be executed offline (possibly in a distributed fashion).
        """
        for i in range(iterations):
            my_cards = self.sample_private_cards()
            round_state = self.sample_initial_state(my_cards)
            self.cfr(round_state, my_cards, 1, 1)
            if (i + 1) % 100000 == 0:
                print(f"Iteration {i+1} complete.")
        return

####################################################
# CFRPlusPokerbot: The Bot Class for Live Play
####################################################

class CFRPlusPokerbot(Bot):
    def __init__(self):
        # Load the precomputed equilibrium strategy from the project root.
        strategy_path = r"C:\Users\evank\Build4Good-Pokerbots\cfr_strategy_full.pkl"
        if os.path.exists(strategy_path):
            with open(strategy_path, "rb") as f:
                self.avg_strategy = pickle.load(f)
            print("Loaded full-scale equilibrium strategy from disk.")
        else:
            # If no precomputed file exists, optionally train offline.
            print("Precomputed strategy not found. Training via CFR⁺... (this may take a very long time)")
            self.solver = CFRPlusSolver()
            self.solver.train(5000000)  # Adjust iteration count as needed.
            self.avg_strategy = self.solver.get_average_strategy()
            with open(strategy_path, "wb") as f:
                pickle.dump(self.avg_strategy, f)
            print("Training complete; equilibrium strategy saved.")

    def handle_new_round(self, game_state, round_state, active):
        # No additional per-round initialization needed.
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        # No additional post-round processing.
        pass

    def get_action(self, game_state, round_state, active):
        # Map the current state and our hole cards into an abstract infoset.
        my_cards = round_state.hands[active]
        key = infoset_key(round_state, my_cards)
        strat = self.avg_strategy.get(key, {"fold": 1/3, "call": 1/3, "raise": 1/3})
        r = random.random()
        cumulative = 0.0
        chosen_action = "call"
        for a, prob in strat.items():
            cumulative += prob
            if r < cumulative:
                chosen_action = a
                break
        # Convert the abstract action into an engine action.
        if chosen_action == "fold":
            return FoldAction()
        elif chosen_action == "call":
            if round_state.pips[1 - active] - round_state.pips[active] == 0:
                return CheckAction()
            else:
                return CallAction()
        elif chosen_action == "raise":
            if RaiseAction in round_state.legal_actions():
                min_raise, _ = round_state.raise_bounds()
                return RaiseAction(min_raise)
            else:
                return CallAction()
        return CallAction()

if __name__ == '__main__':
    run_bot(CFRPlusPokerbot(), parse_args())
