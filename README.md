# haetae

> The haetae is a mythical creature of Chinese origin that possesses the innate ability to determine right from wrong. Resembling a magnificent, armored ox, it questioned officials to discern whether they were corrupt or not. If the official lied, he would be rammed by the haetae's great horns. 

Haetae is a library for training neural network based poker bots. The library comes with a lightweight poker engine, a CFR implementation, and a 100,000 parameter neural network written in torch. Lightning fast hand evaluations are performed using the fantasic OMPEval library.   

# installing 

There are two dependencies, OMPEval and libtorch.

clone OMPEval:

```git submodule update --init --recursive```

build OMPEval:

```cd OMPEval && make```


get libtorch cpu only version from https://pytorch.org/

update /path/to/libtorch in CMakeLists.txt to your libtorch path

```set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/path/to/libtorch")```

then, build the lib:

```mkdir build && cd build && cmake .. && make```

# running

To train a neural network using Deep-CFR, set your parameters in constants.h and run:

```make && ./main```

# evaluations

Currently Local Best Response (LBR) is used to evaluate relative strength of bot throughout training. 

LBR:
best response BR(strat) = max strat' payoff(strat', opp_strat)
where payoff(strat', opp_strat) is the expected poyoff 
playing according to strat' assuming opp plays deterministically following opp_strat

exploitability is defined as difference between
payoff(strat*, BR(strat*)) - payoff(strat, BR(strat))
where strat* is the Nash equilibrium strategy
the lower the better

# TODO

- OMPEval lib/ompeval.a -> lib/libompeval.a



