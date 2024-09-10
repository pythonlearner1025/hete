# notes

best response BR(strat) = max strat' payoff(strat', opp_strat)
where payoff(strat', opp_strat) is the expected poyoff 
playing according to strat' assuming opp plays deterministically following opp_strat

exploitability is defined as difference between
payoff(strat*, BR(strat*)) - payoff(strat, BR(strat))
where strat* is the Nash equilibrium strategy
the lower the better

TODO
- check if opp range is correctly init in lbr.py
- make wprollout calc 1000x more efficient