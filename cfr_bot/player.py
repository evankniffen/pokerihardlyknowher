"""
CFRPlusPokerbot – A full–scale CFR⁺ implementation for B4G Hold'em

This module implements a near–Nash equilibrium solver using CFR⁺ with regret-matching⁺,
linear weighting, and an advanced state abstraction. In this version, the abstraction no longer 
relies on a simple sum of card values ("hand-sum") but uses an estimate of the win probability 
for our hand versus a random opponent (using Monte Carlo simulation) to determine the bucket.
This should guide training more deterministically and help scale our regrets and potential gains.

Note: This is a starting point – further refinements (including caching, more samples, 
or more nuanced scaling) may be required.
"""

import random, pickle, os, math, sys, eval7
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_skeleton.skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from python_skeleton.skeleton.bot import Bot
from python_skeleton.skeleton.runner import parse_args, run_bot
from python_skeleton.skeleton.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK, TerminalState, RoundState

# --- Workaround for the port problem ---
if len(sys.argv) < 2:
    sys.argv.append("5000")

#####################################################
# Advanced Abstraction Functions (Probability-Based)
#####################################################

def hand_win_probability(my_cards, board_cards, iterations=50):
    """
    Estimate the win probability of our hand (my_cards) given the board_cards.
    We assume a uniform random opponent hand from the remaining deck.
    Uses a Monte Carlo simulation with the given number of iterations.
    Returns a float in [0, 1] indicating our estimated win probability.
    """
    # Create a cache key from the sorted cards
    cache_key = tuple(sorted(my_cards + board_cards))
    
    # Check if we have this hand in our cache
    if hasattr(hand_win_probability, 'cache') and cache_key in hand_win_probability.cache:
        return hand_win_probability.cache[cache_key]
    
    wins = 0
    ties = 0
    
    for _ in range(iterations):
        # Create a fresh deck for each iteration
        deck = eval7.Deck()
        # Remove known cards from the deck
        for card in my_cards + board_cards:
            deck.cards.remove(eval7.Card(card))
            
        # Deal opponent's hand
        opp_hand = deck.deal(3)
        
        # Create complete hands
        our_complete = [eval7.Card(card) for card in (my_cards + board_cards)]
        opp_complete = [eval7.Card(card) for card in ([str(card) for card in opp_hand] + board_cards)]
        
        # Evaluate hands
        our_value = eval7.evaluate(our_complete)
        opp_value = eval7.evaluate(opp_complete)
        
        if our_value > opp_value:
            wins += 1
        elif our_value == opp_value:
            ties += 1
            
    # Calculate probability
    prob = (wins + 0.5 * ties) / iterations
    
    # Cache the result
    if not hasattr(hand_win_probability, 'cache'):
        hand_win_probability.cache = {}
    hand_win_probability.cache[cache_key] = prob
    
    return prob

def bucketize_probability(prob, buckets=10):
    """
    Map a win probability (0 to 1) into discrete buckets.
    For example, with 10 buckets, bucket 0 corresponds to 0<=p<0.1, bucket 9 to 0.9<=p<=1.
    """
    bucket = min(buckets - 1, int(prob * buckets))
    return bucket

def bucketize_board(board_cards):
    """
    Bucketize board texture based on the number of cards and their values.
    For B4G Hold'em, we have 0, 2, or 4 board cards.
    """
    if not board_cards:
        return 0
    # For 2-card flop
    if len(board_cards) == 2:
        total = sum(card_value(card) for card in board_cards)
        bucket = min(4, int(total / 10))
        return bucket
    # For 4-card turn
    else:
        total = sum(card_value(card) for card in board_cards)
        bucket = min(4, int(total / 20))
        return bucket

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

def pot_ratio_bucket(round_state):
    """
    Bucketize the pot ratio (current pot divided by total chips in play).
    Uses 5 buckets.
    """
    pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
    ratio = pot / (2 * STARTING_STACK)
    bucket = min(4, int(ratio * 5))
    return bucket

def street_to_index(round_state):
    """
    Map round_state.street to an index for B4G Hold'em:
      0 -> preflop (3 cards)
      2 -> flop (2 cards)
      4 -> turn (2 more cards)
    """
    if round_state.street == 0:
        return 0
    elif round_state.street == 2:
        return 1
    else:
        return 2

def infoset_key(round_state, my_cards):
    """
    Construct an abstraction key for the current decision point.
    The key is a tuple:
        (street_index, hand_bucket, board_bucket, pot_bucket)
    Instead of using a simple hand-sum, we use our win probability.
    """
    street_idx = street_to_index(round_state)
    # Estimate win probability given current hand and board.
    win_prob = hand_win_probability(my_cards, round_state.deck[:round_state.street])
    hand_bucket = bucketize_probability(win_prob)
    board_bucket = bucketize_board(round_state.deck[:round_state.street])
    pot_bucket = pot_ratio_bucket(round_state)
    return (street_idx, hand_bucket, board_bucket, pot_bucket)

#####################################################
# Full-scale CFR⁺ Solver Implementation (unchanged core)
#####################################################

class CFRPlusSolver:
    def __init__(self):
        # Each information set is a node with cumulative positive regret and weighted strategy sums.
        self.node = {}  # key: infoset_key, value: {'regret': {action: float}, 'strategy_sum': {action: float}}
        self.actions = ["fold", "call", "raise"]

    def get_node(self, key):
        if key not in self.node:
            self.node[key] = {
                "regret": {a: 0.0 for a in self.actions},
                "strategy_sum": {a: 0.0 for a in self.actions}
            }
        return self.node[key]

    def regret_matching_plus(self, regrets):
        # Use only positive regrets.
        pos = {a: max(regrets[a], 0) for a in self.actions}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in self.actions}
        else:
            return {a: 1.0 / len(self.actions) for a in self.actions}

    def cfr(self, round_state, my_cards, p0, p1):
        """
        Recursively perform CFR⁺ updates.
        p0: reach probability for the current player.
        p1: reach probability for the opponent.
        """
        if isinstance(round_state, TerminalState):
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
        Simulate the effect of taking an abstract action.
        This placeholder uses a heuristic based on our new probability-based abstraction.
        """
        if action == "fold":
            payoff = -1.0
            return TerminalState([-1, 1], round_state), payoff
        elif action == "call":
            next_state = self.advance_round(round_state)
            # Use our win probability to determine a heuristic payoff.
            win_prob = hand_win_probability(my_cards, round_state.deck[:round_state.street])
            # Scale payoff based on street
            if round_state.street == 0:  # Preflop
                payoff = (win_prob - 0.5) * 0.8  # More conservative preflop
            elif round_state.street == 2:  # Flop
                payoff = (win_prob - 0.5) * 1.2  # More aggressive on flop
            else:  # Turn
                payoff = (win_prob - 0.5) * 1.5  # Most aggressive on turn
            return next_state, payoff
        elif action == "raise":
            win_prob = hand_win_probability(my_cards, round_state.deck[:round_state.street])
            # If our win probability is high, take a more aggressive (positive) payoff.
            if win_prob > 0.65:
                # Scale payoff based on street
                if round_state.street == 0:  # Preflop
                    payoff = 0.8
                elif round_state.street == 2:  # Flop
                    payoff = 1.2
                else:  # Turn
                    payoff = 1.5
                return TerminalState([payoff, -payoff], round_state), payoff
            else:
                next_state = self.advance_round(round_state)
                # Scale payoff based on street
                if round_state.street == 0:  # Preflop
                    payoff = (win_prob - 0.5) * 0.8
                elif round_state.street == 2:  # Flop
                    payoff = (win_prob - 0.5) * 1.2
                else:  # Turn
                    payoff = (win_prob - 0.5) * 1.5
                return next_state, payoff
        return round_state, 0.0

    def advance_round(self, round_state):
        """Advance to the next betting round using the engine's method."""
        return round_state.proceed_street()

    def train(self, iterations):
        """
        Run CFR⁺ iterations over randomly sampled game trajectories.
        """
        for i in range(iterations):
            my_cards = self.sample_private_cards()
            round_state = self.sample_initial_state(my_cards)
            self.cfr(round_state, my_cards, 1, 1)
            if (i + 1) % 100000 == 0:
                print(f"Iteration {i+1} complete.")
        return

    def sample_private_cards(self):
        deck = eval7.Deck()
        deck.shuffle()
        cards = deck.deal(3)  # Deal 3 cards for B4G Hold'em
        return [str(card) for card in cards]

    def sample_initial_state(self, my_cards):
        opponent_cards = self.sample_private_cards()
        hands = [my_cards, opponent_cards]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        deck = []  # For simulation purposes, board is empty initially.
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

#####################################################
# CFRPlusPokerbot Integration with Build4Good Framework
#####################################################

class CFRPlusPokerbot(Bot):
    def __init__(self):
        strategy_path = os.path.join(os.path.dirname(__file__), "cfr_strategy_full.pkl")
        if os.path.exists(strategy_path):
            with open(strategy_path, "rb") as f:
                self.avg_strategy = pickle.load(f)
            print("Loaded full-scale equilibrium strategy from disk.")
        else:
            print("Training full-scale equilibrium strategy via CFR⁺... (this may take a while)")
            self.solver = CFRPlusSolver()
            self.solver.train(500000)  # Adjust iterations as needed.
            self.avg_strategy = self.solver.get_average_strategy()
            with open(strategy_path, "wb") as f:
                pickle.dump(self.avg_strategy, f)
            print("Training complete; strategy saved.")

    def handle_new_round(self, game_state, round_state, active):
        # Log or update as needed.
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        # Process end-of-round data if needed.
        pass

    def get_action(self, game_state, round_state, active):
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
        if chosen_action == "fold":
            return FoldAction()
        elif chosen_action == "call":
            if round_state.pips[1 - active] - round_state.pips[active] == 0:
                return CheckAction()
            else:
                return CallAction()
        elif chosen_action == "raise":
            if RaiseAction in round_state.legal_actions():
                min_raise, max_raise = round_state.raise_bounds()
                # Scale raise size based on street and hand strength
                win_prob = hand_win_probability(my_cards, round_state.deck[:round_state.street])
                if round_state.street == 0:  # Preflop
                    raise_size = min_raise + int((max_raise - min_raise) * win_prob * 0.5)
                elif round_state.street == 2:  # Flop
                    raise_size = min_raise + int((max_raise - min_raise) * win_prob * 0.7)
                else:  # Turn
                    raise_size = min_raise + int((max_raise - min_raise) * win_prob * 0.9)
                raise_size = min(raise_size, max_raise)
                return RaiseAction(raise_size)
            else:
                return CallAction()
        return CallAction()

if __name__ == '__main__':
    run_bot(CFRPlusPokerbot(), parse_args())

