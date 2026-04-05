# Traffic Signal RL Environment

A custom Gymnasium-compatible reinforcement learning environment 
simulating traffic signal control on a 2x2 grid.

## Features
- 4 intersections, each with 4 independent signal directions
- Cars flow between neighboring intersections realistically
- Compatible with Stable Baselines3 for training

## Setup
pip install gymnasium stable-baselines3

## Run
python traffic_env_v2.py

# traffic-RL-env
using this RL environment an AI agent can learn to manage traffic signals on 2x2 grid of road intersections. this is a simulated situation of course but this idea can be deployed on a whole city by using multi agent RL which will be much more complex than this but this is the building block for it.
