# RL-AERPAW-DT
## A UAV Digital Twin for Communication Optimization with Reinforcement Learning

## Research Poster
![A research poster updated on March 18th, 2025](presentations/research_poster_final_img.jpg)
This research poster was presented at the Goodnight Research Symposium at NC State on March 18th, 2025.

## Digital Twin Demonstrations
The two GIFs below show the evolution of the simulated digital twin environment including
* (Top) Coverage Map with Ground Users color-coded based on their UAV assignment
* (Far Left) The number of Ground Users assigned to each UAV
* (Middle Left) The total theoretical maximum data rate for each UAV
* (Middle Right) The total actual data rate through each UAV
* (Far Right) The power consumption of each UAV including movement, hovering, and transmitter power

### Case 2: Demonstration of What-if Analysis on Network Performance (Consider both system and user performance)
![A Demonstration of the simulation showing Data Rate, UAV Energy Consumption, Receiver Assignments, and the Coverage Map using an optimized load-balancing algorithm.](gifs/0_assignGUs.gif)

Case 2 showcases a bi-objective algorithm that seeks to balance the cumulative actual data rate while including a load-balancing compensation. This algorithm attempts to maximize the minimum number of Ground Users assigned to any UAV. Because the total number of GUs is bounded, this results in a more equal distribution of network load. We can see from this demonstration that the total coverage, meaning the percentage of GUs assigned to a UAV, is almost always 100%. Furthermore, the variance in the load for each UAV is stable compared to Case 1. Finally, the acutal data rate is relatively unaffected by the change, meaning that we gain load-balancing without losing communication rate.

### Case 1: Demonstration of What-if Analysis on Network Performance (Prioritize user QoS)
![A Demonstration of the simulation showing Data Rate, UAV Energy Consumption, Receiver Assignments, and the Coverage Map.](gifs/1_assignGUsWithLoadBalancing.gif)

Case 1 shows a basic throughput maximization algorithm to assign the Ground Users. It uses a Constraint-Programming Solver from Google OR-Tools to optimize a linear integer constraint problem. Case 1 shows that the actual data rate is high, but the number of assignments is incredibly variable and unstable over time, which Case 1 attempts to fix.

## Goals
- Construct a pipeline for the simulation of pedestrian data using SUMO
- Import building data from OpenStreetMaps and use Sionna for ray tracing
- Create a realistic simulation for simulating UAV communication with ground users in urban settings.
- Train a reinforcement learning model to control the UAVs to maximize the communication capacity of the UAV network with the Ground Users.

## Current Task
- Working on a minimial control framework to simulation UAV and pedestrian movement and provide a wrapper for machine learning algorithms.

## What's Done Already
- Generating pedestrian data with SUMO and parsing it into a usable format.
- Importing data from OpenStreetMaps into Sionna
- Running Sionna ray tracing algorithms to determine communication metrics like path loss between a UAV and a ground user.


