# Hete

Hete is a performant Deep-CFR network trainer in c++ using [mlx](https://github.com/ml-explore/mlx).

Features: 
- Decoder only poker GPT in one file ```src/models/model.h``` 
- Minimal 2+ player poker engine
- Monte carlo outcome sampler for game rollouts 
- CFR training loop
- Fast hand evaluations using OMPEval library.   

# performance

terminology:
- mbb = 1/1000 * big blind
- baseline statistics = difference between outcome of you played vs. how slumbot would have played. more positive is better.
- mbb_per_hand = average mbb winning across 10000 games

Winning by 50mbb or more per hand on average is considered a significant win between professionals.

Performance of my latest run against [slumbot.com](https://www.slumbot.com/)

![mbb_per_hand](https://github.com/pythonlearner1025/hete/blob/master/mbb.png?raw=true)

![baseline_avg](https://github.com/pythonlearner1025/hete/blob/master/baseline.png?raw=true)

There's work to do.

# installing 

There are two dependencies, OMPEval and mlx.

clone mlx:

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



