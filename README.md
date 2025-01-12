# Hete
Hete is a performant Deep-CFR network trainer in c++ using [mlx](https://github.com/ml-explore/mlx).

## Features
- Decoder only poker GPT in one file ```src/models/model.h``` 
- Minimal 2+ player poker engine
- Monte carlo outcome sampler for game rollouts 
- CFR training loop
- Fast hand evaluations using OMPEval library   

## Performance
terminology:
- mbb = 1/1000 * big blind
- baseline statistics = difference between outcome of you played vs. how slumbot would have played. more positive is better.
- mbb_per_hand = average mbb winning across 10000 games

winning by 50mbb or more per hand on average is considered a significant win between professionals.

performance of my latest run against [slumbot.com](https://www.slumbot.com/):

![mbb_per_hand](https://github.com/pythonlearner1025/hete/blob/master/mbb.png?raw=true)
![baseline_avg](https://github.com/pythonlearner1025/hete/blob/master/baseline.png?raw=true)

there's work to do.

## Installation 

first clone hete:
```
git clone https://github.com/pythonlearner1025/hete.git && 
cd hete  
```

install dependencies:

1. build OMPEval:
```
git clone https://github.com/pythonlearner1025/OMPEval.git &&
cd OMPEval &&
make
```

2. build and install mlx:
```
git clone https://github.com/ml-explore/mlx.git && 
cd mlx &&
mkdir -p build && cd build && 
cmake .. && make -j &&
make install
```

3. build hete:
```
mkdir build && cd build && cmake ..
```

4. build python extension:
```
python setup.py build_ext --inplace
```

## Training
set your parameters in constants.h and run:
```
make && ./main
```

trained models will be saved under:
```
/hete/out/<timestamp>/<cfr_iteration_index>/<player_index>
```

## Evaluation

create and activate virtual environment:
```
python -m venv env && . env/bin/activate
pip install -r requirements.txt
```

evaluate against slumbot:
```
python eval.py --wandb 0 --num_hands 1000 --auto 1
```
optionally stream to wandb with ```--wandb 1```

## Plotting
plot performance against slumbot:
```
python plot.py
```