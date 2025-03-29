import itertools
import random
import pickle
import time
import math
from cfr import (
    CFRPlusSolver, CFRPlusPokerbot, 
    hand_win_probability, bucketize_probability, 
    bucketize_board, pot_ratio_bucket,
    street_to_index
)
from all_in_bot.skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from all_in_bot.skeleton.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK, TerminalState, RoundState
from all_in_bot.skeleton.bot import Bot
import eval7

class RandomBot(Bot):
    def __init__(self, strategy=None, params=None):
        self.strategy = strategy
        self.params = params or {}
        
    def handle_new_round(self, game_state, round_state, active):
        pass
        
    def handle_round_over(self, game_state, terminal_state, active):
        pass
        
    def get_action(self, round_state, active):
        """Get action based on the loaded strategy with dynamic adjustments."""
        legal_actions = list(round_state.legal_actions())
        if not legal_actions:
            return None
            
        # Get our cards and calculate key metrics
        my_cards = round_state.hands[active]
        win_prob = hand_win_probability(my_cards, round_state.deck[:round_state.street], self.params["monte_carlo_iterations"])
        
        # Calculate pot odds and implied odds
        pot = sum(round_state.pips)
        to_call = max(0, round_state.pips[1-active] - round_state.pips[active])
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        # Calculate effective stack and position advantage
        effective_stack = min(round_state.stacks[0], round_state.stacks[1]) / BIG_BLIND
        is_button = round_state.button == active
        
        # Calculate pot size relative to stack for aggression scaling
        pot_to_stack_ratio = pot / (STARTING_STACK * 2)
        
        # Street-based strategy adjustments
        if round_state.street == 0:  # Preflop
            if win_prob < 0.35:  # Weak hand
                if CheckAction in legal_actions:
                    return CheckAction()
                elif pot_odds < 0.12 and effective_stack > 25:  # More selective with deep stack calls
                    return CallAction()
                elif is_button and random.random() < 0.15 and effective_stack > 30:  # Increased button steals
                    min_raise, max_raise = round_state.raise_bounds()
                    # Smaller steals to minimize risk
                    raise_size = min_raise + int((max_raise - min_raise) * 0.25)
                    return RaiseAction(raise_size)
                else:
                    return FoldAction()
            elif win_prob < 0.6:  # Medium hand
                if RaiseAction in legal_actions:
                    if is_button or random.random() < 0.65:  # More aggressive on button
                        min_raise, max_raise = round_state.raise_bounds()
                        # Size based on stack depth and position
                        raise_mult = 0.55 if is_button and effective_stack > 25 else 0.4
                        raise_size = min_raise + int((max_raise - min_raise) * raise_mult)
                        return RaiseAction(raise_size)
                if CheckAction in legal_actions:
                    return CheckAction()
                elif pot_odds < win_prob * 1.25:  # More selective calling
                    return CallAction()
                else:
                    return FoldAction()
            else:  # Strong hand
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Dynamic raise sizing based on position and stack
                    if is_button and effective_stack > 35:
                        raise_size = max_raise  # Maximum pressure
                    elif effective_stack > 25:
                        raise_size = min_raise + int((max_raise - min_raise) * 0.9)
                    else:
                        raise_size = min_raise + int((max_raise - min_raise) * 0.8)
                    return RaiseAction(raise_size)
                else:
                    return CallAction()
        else:  # Postflop
            if win_prob < 0.45:  # Weak hand
                if CheckAction in legal_actions:
                    return CheckAction()
                elif pot_odds < 0.2 and effective_stack > 30:  # More selective with deep stacks
                    return CallAction()
                elif is_button and random.random() < 0.25 and pot_to_stack_ratio < 0.25:  # More bluffs in small pots
                    min_raise, max_raise = round_state.raise_bounds()
                    # Small bluff raises
                    raise_size = min_raise + int((max_raise - min_raise) * 0.35)
                    return RaiseAction(raise_size)
                else:
                    return FoldAction()
            elif win_prob < 0.7:  # Medium hand
                if RaiseAction in legal_actions:
                    if (is_button and random.random() < 0.75) or random.random() < 0.55:
                        min_raise, max_raise = round_state.raise_bounds()
                        # Size based on pot and position
                        raise_mult = 0.65 if is_button else 0.5
                        raise_size = min_raise + int((max_raise - min_raise) * raise_mult)
                        return RaiseAction(raise_size)
                if CheckAction in legal_actions:
                    if random.random() < 0.25:  # Increased check-raise frequency
                        if RaiseAction in legal_actions:
                            min_raise, max_raise = round_state.raise_bounds()
                            raise_size = min_raise + int((max_raise - min_raise) * 0.45)
                            return RaiseAction(raise_size)
                    return CheckAction()
                elif pot_odds < win_prob * 1.15:  # More selective calling
                    return CallAction()
                else:
                    return FoldAction()
            else:  # Strong hand
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Dynamic sizing based on multiple factors
                    if effective_stack > 40:
                        raise_size = max_raise
                    elif pot_to_stack_ratio > 0.6:  # Large pot
                        raise_size = min_raise + int((max_raise - min_raise) * 0.95)
                    else:
                        raise_size = min_raise + int((max_raise - min_raise) * 0.85)
                    return RaiseAction(raise_size)
                elif CallAction in legal_actions:
                    return CallAction()
                else:
                    return CheckAction()

    def sample_private_cards(self):
        """Sample 3 random cards for B4G Hold'em."""
        deck = eval7.Deck()
        deck.shuffle()
        cards = deck.deal(3)
        return [str(card) for card in cards]

class SimpleBot(Bot):
    """A simple bot implementation for testing purposes."""
    def __init__(self):
        pass
        
    def handle_new_round(self, game_state, round_state, active):
        pass
        
    def handle_round_over(self, game_state, terminal_state, active):
        pass
        
    def get_action(self, game_state, round_state, active):
        """Simple strategy: Check when possible, call small bets, rarely raise."""
        legal_actions = list(round_state.legal_actions())
        if not legal_actions:
            return None
            
        if CheckAction in legal_actions:
            return CheckAction()
            
        # Calculate pot odds for calling
        if CallAction in legal_actions:
            # Calculate the amount needed to call
            to_call = 0
            if hasattr(round_state, 'pips') and len(round_state.pips) > active:
                our_pip = round_state.pips[active]
                their_pip = round_state.pips[1-active] if len(round_state.pips) > (1-active) else 0
                to_call = their_pip - our_pip
            
            if to_call > 0:
                pot = sum(round_state.pips) if hasattr(round_state, 'pips') else 0
                pot_odds = to_call / (pot + to_call)
                # Call if the pot odds are good
                if pot_odds < 0.3:
                    return CallAction()
            else:
                return CallAction()  # Free call
            
        # Occasionally raise (10% chance) if possible
        if RaiseAction in legal_actions and random.random() < 0.1:
            min_raise, max_raise = round_state.raise_bounds()
            # Make a small raise
            raise_size = min_raise + int((max_raise - min_raise) * 0.3)
            return RaiseAction(raise_size)
            
        # Otherwise fold if possible, else call
        if FoldAction in legal_actions:
            return FoldAction()
        else:
            return CallAction()  # If we can't fold, we have to call

def simulate_game(bot):
    """Simulate a single hand of poker."""
    # Create initial state
    deck = eval7.Deck()
    deck.shuffle()
    
    # Deal cards
    my_cards = [str(card) for card in deck.deal(3)]
    opp_cards = [str(card) for card in deck.deal(3)]
    
    # Create initial round state
    pips = [SMALL_BLIND, BIG_BLIND]
    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
    hands = [my_cards, opp_cards]
    round_state = RoundState(0, 0, pips, stacks, hands, [], None)
    
    # Track actions and profit
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    profit = 0
    
    # Track opponent tendencies
    opp_aggression = 0  # Track opponent's aggression level
    opp_calls = 0      # Track opponent's calling frequency
    total_opp_actions = 0
    
    # Play the hand
    while not isinstance(round_state, TerminalState):
        active = round_state.button
        
        if active == 0:  # Our turn
            action = bot.get_action(round_state, active)
            if isinstance(action, FoldAction):
                action_counts["fold"] += 1
            elif isinstance(action, CallAction) or isinstance(action, CheckAction):
                action_counts["call"] += 1
            elif isinstance(action, RaiseAction):
                action_counts["raise"] += 1
        else:  # Opponent's turn - exploitable strategy
            legal_actions = list(round_state.legal_actions())
            opp_win_prob = hand_win_probability(opp_cards, round_state.deck[:round_state.street], 100)
            
            # Calculate pot odds and effective stack
            pot = sum(round_state.pips)
            to_call = max(0, round_state.pips[0] - round_state.pips[1])
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
            effective_stack = min(round_state.stacks[0], round_state.stacks[1]) / BIG_BLIND
            
            # Calculate pot to stack ratio for opponent decision making
            pot_to_stack = pot / (STARTING_STACK * 2)
            
            # More exploitable opponent strategy with tendencies
            if round_state.street == 0:  # Preflop
                if opp_win_prob < 0.3:  # Very weak hand
                    if CheckAction in legal_actions:
                        action = CheckAction()
                    elif pot_odds < 0.2:  # Call too often with weak hands
                        action = CallAction()
                        opp_calls += 1
                    else:
                        action = FoldAction()
                elif opp_win_prob < 0.5:  # Weak to medium hand
                    if CheckAction in legal_actions:
                        if random.random() < 0.3:  # Sometimes raise as bluff
                            if RaiseAction in legal_actions:
                                min_raise, max_raise = round_state.raise_bounds()
                                raise_size = min_raise + int((max_raise - min_raise) * 0.3)
                                action = RaiseAction(raise_size)
                                opp_aggression += 1
                            else:
                                action = CheckAction()
                        else:
                            action = CheckAction()
                    elif pot_odds < 0.35:  # Call too liberally
                        action = CallAction()
                        opp_calls += 1
                    else:
                        action = FoldAction()
                else:  # Strong hand
                    if RaiseAction in legal_actions:
                        min_raise, max_raise = round_state.raise_bounds()
                        # Predictable sizing based on hand strength
                        raise_size = min_raise + int((max_raise - min_raise) * opp_win_prob)
                        action = RaiseAction(raise_size)
                        opp_aggression += 1
                    else:
                        action = CallAction()
                        opp_calls += 1
            else:  # Postflop
                if opp_win_prob < 0.4:  # Weak hand
                    if CheckAction in legal_actions:
                        action = CheckAction()
                    elif pot_odds < 0.3 and pot_to_stack < 0.4:  # Call too often in small pots
                        action = CallAction()
                        opp_calls += 1
                    else:
                        action = FoldAction()
                elif opp_win_prob < 0.6:  # Medium hand
                    if CheckAction in legal_actions:
                        if random.random() < 0.4 and effective_stack > 20:  # Semi-bluff frequently
                            if RaiseAction in legal_actions:
                                min_raise, max_raise = round_state.raise_bounds()
                                raise_size = min_raise + int((max_raise - min_raise) * 0.4)
                                action = RaiseAction(raise_size)
                                opp_aggression += 1
                            else:
                                action = CheckAction()
                        else:
                            action = CheckAction()
                    elif pot_odds < 0.4:  # Call too often
                        action = CallAction()
                        opp_calls += 1
                    else:
                        action = FoldAction()
                else:  # Strong hand
                    if RaiseAction in legal_actions and effective_stack > 15:
                        min_raise, max_raise = round_state.raise_bounds()
                        # Always raise max with strong hands
                        raise_size = max_raise
                        action = RaiseAction(raise_size)
                        opp_aggression += 1
                    else:
                        action = CallAction()
                        opp_calls += 1
            
            total_opp_actions += 1
        
        if action is None:
            break
            
        round_state = round_state.proceed(action)
        
        if isinstance(round_state, TerminalState):
            profit = round_state.deltas[0] / BIG_BLIND
            break
            
    # Calculate opponent tendencies for future reference
    if total_opp_actions > 0:
        opp_aggression_freq = opp_aggression / total_opp_actions
        opp_calling_freq = opp_calls / total_opp_actions
    else:
        opp_aggression_freq = 0
        opp_calling_freq = 0
            
    return {
        "profit": profit,
        "action_counts": action_counts,
        "opp_tendencies": {
            "aggression_freq": opp_aggression_freq,
            "calling_freq": opp_calling_freq
        }
    }

def test_parameter_combination(params):
    """Test a single parameter combination."""
    print("Starting test...")
    start_time = time.time()
    
    # Load the strategy
    print("Loading strategy...")
    strategy_load_start = time.time()
    with open("cfr_strategy_full.pkl", "rb") as f:
        strategy = pickle.load(f)
    print(f"Strategy loaded in {time.time() - strategy_load_start:.2f}s")
    
    # Create a bot with the current parameters
    bot = RandomBot(strategy, params)
    
    # Run simulations
    print("Starting simulations...")
    sim_start = time.time()
    total_profit = 0
    wins = 0
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    
    # Use number of hands from params
    num_hands = params["test_hands"]
    for i in range(num_hands):
        if i % 100 == 0:  # Print progress every 100 hands for 2000 hand test
            print(f"Hand {i+1}/{num_hands}...")
        result = simulate_game(bot)
        total_profit += result["profit"]
        if result["profit"] > 0:
            wins += 1
        for action, count in result["action_counts"].items():
            action_counts[action] += count
    
    print(f"Simulations completed in {time.time() - sim_start:.2f}s")
    
    # Calculate statistics
    total_actions = sum(action_counts.values())
    action_percentages = {action: (count / total_actions) * 100 for action, count in action_counts.items()}
    win_rate = (wins / num_hands) * 100
    avg_profit_per_hand = total_profit / num_hands
    
    results = {
        "total_profit": total_profit,
        "avg_profit_per_hand": avg_profit_per_hand,
        "win_rate": win_rate,
        "action_counts": action_counts,
        "action_percentages": action_percentages,
        "total_time": time.time() - start_time,
        "num_hands": num_hands
    }
    
    print(f"Test completed in {results['total_time']:.2f}s")
    return results

def optimize_parameters():
    # Parameter ranges to test - optimized for 2000 hands
    parameter_ranges = {
        "monte_carlo_iterations": [150, 200],  # Increased iterations for better accuracy
        "prob_buckets": [8, 10],  # Added more granular probability buckets
        "test_hands": [2000],  # Keep at 2000 for comprehensive statistics
        "raising_threshold": [0.2, 0.25, 0.3],  # More granular raising thresholds
        "payoff_scaling": [1.8, 2.0, 2.2]  # More granular payoff scaling
    }
    
    # Generate all combinations
    keys = parameter_ranges.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*[parameter_ranges[k] for k in keys])]
    
    best_result = None
    best_params = None
    best_profit = float('-inf')
    
    print(f"Testing {len(combinations)} parameter combinations...")
    print("Estimated time: 45-60 minutes")
    print("Each combination will test 2000 hands")
    
    for i, params in enumerate(combinations):
        print(f"\nTesting combination {i+1}/{len(combinations)}")
        print("Parameters:", params)
        
        start_time = time.time()
        results = test_parameter_combination(params)
        end_time = time.time()
        
        print(f"Results: {results}")
        print(f"Time taken: {end_time - start_time:.2f}s")
        
        if results["total_profit"] > best_profit:
            best_profit = results["total_profit"]
            best_result = results
            best_params = params
            print("New best parameters found!")
            print(f"Current best profit: {best_profit:.2f} BB")
            print(f"Current best win rate: {best_result['win_rate']:.1f}%")
    
    print("\nBest parameters found:")
    print("Parameters:", best_params)
    print("Results:", best_result)
    
    # Save best parameters
    with open("best_parameters.pkl", "wb") as f:
        pickle.dump({"params": best_params, "results": best_result}, f)

def test_vs_skeleton(num_hands=10000):
    """Test our bot against the skeleton bot for specified number of hands."""
    print(f"\nTesting against simple bot for {num_hands} hands...")
    print("Loading strategy...")
    
    # Load strategy and create our bot
    with open("cfr_strategy_full.pkl", "rb") as f:
        strategy = pickle.load(f)
    
    params = {
        "monte_carlo_iterations": 100,
        "prob_buckets": 8,
        "test_hands": num_hands,
        "raising_threshold": 0.25,
        "payoff_scaling": 2.0
    }
    
    our_bot = RandomBot(strategy, params)
    
    # Run simulations
    print("Starting simulations...")
    start_time = time.time()
    total_profit = 0
    wins = 0
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    profit_by_position = {"button": 0, "bb": 0}
    hands_by_position = {"button": 0, "bb": 0}
    street_wins = {0: 0, 2: 0, 4: 0}  # Wins by street
    max_profit = float('-inf')
    min_profit = float('inf')
    profit_per_hand = []
    
    for i in range(num_hands):
        if i % 500 == 0:  # Print progress every 500 hands
            print(f"Hand {i+1}/{num_hands}...")
            if i > 0:
                current_profit_per_hand = total_profit / i
                print(f"Current profit per hand: {current_profit_per_hand:.3f} BB")
        
        result = simulate_game_vs_skeleton(our_bot)
        total_profit += result["profit"]
        profit_per_hand.append(result["profit"])
        
        # Track max and min profits
        max_profit = max(max_profit, result["profit"])
        min_profit = min(min_profit, result["profit"])
        
        # Track position results
        position = "button" if i % 2 == 0 else "bb"
        profit_by_position[position] += result["profit"]
        hands_by_position[position] += 1
        
        if result["profit"] > 0:
            wins += 1
            if "last_street" in result:
                street_wins[result["last_street"]] += 1
                
        for action, count in result["action_counts"].items():
            action_counts[action] += count
    
    # Calculate statistics
    total_actions = sum(action_counts.values())
    action_percentages = {action: (count / total_actions) * 100 for action, count in action_counts.items()}
    win_rate = (wins / num_hands) * 100
    avg_profit_per_hand = total_profit / num_hands
    
    # Calculate position-based stats
    button_profit_per_hand = profit_by_position["button"] / hands_by_position["button"]
    bb_profit_per_hand = profit_by_position["bb"] / hands_by_position["bb"]
    
    # Calculate standard deviation
    mean = avg_profit_per_hand
    variance = sum((x - mean) ** 2 for x in profit_per_hand) / len(profit_per_hand)
    std_dev = math.sqrt(variance)
    
    results = {
        "total_profit": total_profit,
        "avg_profit_per_hand": avg_profit_per_hand,
        "win_rate": win_rate,
        "action_counts": action_counts,
        "action_percentages": action_percentages,
        "position_stats": {
            "button_profit_per_hand": button_profit_per_hand,
            "bb_profit_per_hand": bb_profit_per_hand
        },
        "street_wins": street_wins,
        "max_profit": max_profit,
        "min_profit": min_profit,
        "std_dev": std_dev,
        "total_time": time.time() - start_time,
        "num_hands": num_hands
    }
    
    print("\nDetailed Results vs Simple Bot:")
    print(f"Total profit: {results['total_profit']:.1f} BB")
    print(f"Average profit per hand: {results['avg_profit_per_hand']:.3f} BB")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print("\nAction distribution:")
    for action, percentage in results['action_percentages'].items():
        print(f"  {action}: {percentage:.1f}%")
    print("\nPosition statistics:")
    print(f"  Button profit per hand: {results['position_stats']['button_profit_per_hand']:.3f} BB")
    print(f"  BB profit per hand: {results['position_stats']['bb_profit_per_hand']:.3f} BB")
    print("\nProfit statistics:")
    print(f"  Max profit in a hand: {results['max_profit']:.1f} BB")
    print(f"  Min profit in a hand: {results['min_profit']:.1f} BB")
    print(f"  Standard deviation: {results['std_dev']:.3f} BB")
    print(f"\nTotal time: {results['total_time']:.2f}s")
    print(f"Average time per hand: {results['total_time']/num_hands:.3f}s")
    
    return results

def simulate_game_vs_skeleton(our_bot):
    """Simulate a single hand of poker against the simple bot."""
    # Create initial state
    deck = eval7.Deck()
    deck.shuffle()
    
    # Deal cards
    my_cards = [str(card) for card in deck.deal(3)]
    opp_cards = [str(card) for card in deck.deal(3)]
    
    # Create initial round state
    pips = [SMALL_BLIND, BIG_BLIND]
    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
    hands = [my_cards, opp_cards]
    round_state = RoundState(0, 0, pips, stacks, hands, [], None)
    
    # Create simple bot opponent
    simple_bot = SimpleBot()
    
    # Track actions and profit
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    profit = 0
    last_street = 0
    
    # Play the hand
    while not isinstance(round_state, TerminalState):
        active = round_state.button
        last_street = round_state.street
        
        if active == 0:  # Our turn
            action = our_bot.get_action(round_state, active)
            if isinstance(action, FoldAction):
                action_counts["fold"] += 1
            elif isinstance(action, CallAction) or isinstance(action, CheckAction):
                action_counts["call"] += 1
            elif isinstance(action, RaiseAction):
                action_counts["raise"] += 1
        else:  # Simple bot's turn
            action = simple_bot.get_action(None, round_state, active)
        
        if action is None:
            break
            
        round_state = round_state.proceed(action)
        
        if isinstance(round_state, TerminalState):
            profit = round_state.deltas[0] / BIG_BLIND
            break
            
    return {
        "profit": profit,
        "action_counts": action_counts,
        "last_street": last_street
    }

if __name__ == "__main__":
    test_vs_skeleton(10000) 