# Defuant Model Network 

## Overview
This script simulates the formation and evolution of opinions within a small-world network using the Defuant model.

## Installation Guide
Ensure that your environment has Pycharm 2023.2.1 or above installed. Additionally, the following libraries are required:
- **numpy** : For array and matrix operations.
- **matplotlib**: For generating charts.

## Features Description
### Network Construction
- **population = Network()** creates a new network instance.
- **population.make_small_world_network (N, re_wire_prob)** initializes this network using a small-world network model
- **N** represents the number of nodes in the network and **re_wire_prob'** is the rewiring probability, affecting the randomness of the node connections and are configurable

### Opinion Update Process
- A for loop iterates through each node performing the following steps:
-Randomly select a node **position_random_element** from the network.
  -Randomly select a neighbor **random_element_neighbor** from this node’s connections.
  -Calculate the opinion difference between these two nodes.
		  If this difference is less than a given threshold, then adjust the opinions of these two nodes based on the coupling coefficient to reduce their opinion difference.)
  -The average value of all nodes' opinions is recorded and plotted to show its evolution over iterations.

### Visualization
-At the end of each iteration, calculate the average opinion value across all nodes in the network and add this value to the **array**.
-Use the **matplotlib** library to plot these average values to visualize the evolution of opinion values across the network.

## Execution Instructions
-The script can be run from the command line using the following format:
```
    python FCP_assignment_combine.py --use_network 1 --N 20
```
- **N** : Sets the number of nodes in the network.   

## Usage Example
```
def defuant_main_network(Threshold = 0.5, Coupling = 0.5, N=20, re_wire_prob=0.2)
```
-**Threshold **: Controls the maximum difference in opinions for acceptance.
-**Coupling** : Determines the speed of opinion adjustment.
-**N** (Number of nodes): The total number of nodes in the network.
-**re_wire_prob**(Rewiring probability): Affects the network topology through the probability of rewiring.
