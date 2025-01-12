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

```
git clone https://github.com/pythonlearner1025/hete.git && 
cd hete  
```

clone and build OMPEval in /hete:

```
git clone https://github.com/pythonlearner1025/OMPEval.git &&
cd OMPEval &&
make
```

clone and build mlx in /hete:

```
git clone https://github.com/ml-explore/mlx.git && 
cd mlx &&
mkdir -p build && cd build && 
cmake .. && make -j 
```

then install mlx:
```make install```

then, build hete in /hete:

```mkdir build && cd build && cmake ..```

finally, set your training parameters in constants.h and run:

```make && ./main```

building the python extension:

```python setup.py build_ext --inplace```

# running

To train a neural network using Deep-CFR, set your parameters in constants.h and run:

```make && ./main```

# evaluating

Make a virtual env in /hete and activate it

```python -m venv env && . env/bin/activate```

Install the required dependencies

```pip install -r requirements.txt```

Run eval.py to automatically pit the most recently trained model against slumbot at slumbot.com for 1000 hands (you need trained models for both player 1 and player 2).   

```python eval.py --wandb 0 --num_hands 1000 --auto 1```

Optionally stream performance to wandb with ```--wandb 1```