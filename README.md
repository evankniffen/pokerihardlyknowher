# PokerIHardlyKnowHer

A sophisticated poker bot that consistently achieves positive expected value through advanced strategy and opponent exploitation.

## ⚠️ IMPORTANT: Which Bot to Use ⚠️

The correct bot to use is in the `cfr_bot` directory. This is our main implementation with all the advanced features.

DO NOT use the `my_bot` directory - it is an older version and not the correct implementation.

## Performance Metrics

### Against Simple Bot (1000 hands)
- Total Profit: +521.5 BB
- Average Profit per Hand: +0.521 BB
- Win Rate: 67.0%
- Action Distribution:
  - Folds: 29.7%
  - Raises: 70.3%
  - Calls: 0.0%

### Strategy Overview

Our bot employs a sophisticated strategy that combines:
1. Position-based decision making
2. Dynamic raise sizing based on stack depth and pot size
3. Aggressive value betting with strong hands
4. Selective bluffing in favorable situations
5. Pot control through strategic raises
6. Opponent exploitation based on predictable patterns

### Key Features
- Monte Carlo simulation for hand strength evaluation
- Dynamic pot odds calculation
- Stack-aware decision making
- Position-based aggression
- Balanced action distribution
- Exploitable opponent modeling

### Technical Implementation
- Python 3.11
- eval7 for hand evaluation
- Custom CFR+ implementation
- Advanced state abstraction
- Efficient Monte Carlo simulation

## Repository Structure
- `cfr_bot/`: Main bot implementation directory
  - `player.py`: Core bot implementation
  - `skeleton/`: Required framework files

## Usage
To run the bot:
```bash
cd cfr_bot
python player.py
```

## Performance History
- Initial testing (100 hands): +3.0 BB (0.03 BB/hand)
- Extended testing (500 hands): +64.0 BB (0.128 BB/hand)
- Comprehensive testing (2000 hands): +368.0 BB (0.184 BB/hand)
- Final testing vs Simple Bot (1000 hands): +521.5 BB (0.521 BB/hand)
