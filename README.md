# Ant Colony Optimization for Bin Packing Problems

This project implements an Ant Colony Optimization (ACO) algorithm to solve two variations of the **Bin Packing Problem (BPP)**. The ACO algorithm is a bio-inspired metaheuristic that simulates the behavior of ants searching for food to find optimal or near-optimal solutions to computational problems.

---

## Problem Description
The **Bin Packing Problem (BPP)** involves distributing a set of items into a given number of bins such that the load of the bins is balanced. The goal is to minimize the difference between the heaviest and lightest bins.

### Variations in the Project:
1. **BPP1**: Items have weights ranging from 1 to 500.
2. **BPP2**: Items have weights computed as \( \frac{i^2}{2} \), where \( i \) ranges from 1 to 500.

---

## Algorithm Overview
The ACO algorithm uses multiple ants to explore possible solutions. Each ant constructs a path (assignment of items to bins) based on pheromone levels, which represent the desirability of certain choices. The algorithm consists of the following steps:

1. **Initialization**:
   - Randomly initialize the pheromone table.

2. **Ant Path Construction**:
   - Each ant selects a bin for each item based on pheromone levels and probabilistic rules.

3. **Fitness Evaluation**:
   - Fitness is calculated as the difference between the heaviest and lightest bin loads.

4. **Pheromone Update**:
   - Pheromone levels are updated based on the fitness of the solutions.
   - Pheromone evaporation encourages exploration.

5. **Termination**:
   - The algorithm stops after a specified number of fitness evaluations.

---

## Features
- Simulates multiple ant paths and updates pheromones based on their fitness.
- Implements evaporation to encourage exploration of the solution space.
- Evaluates performance across multiple trials to compute average fitness.
- Generates separate visualizations for BPP1 and BPP2.

---

## Requirements
The project requires the following Python libraries:
- `random`
- `matplotlib`
- `numpy`

To install missing dependencies, run:
```bash
pip install matplotlib numpy
```

---

## Usage

### Running the Project
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```
3. Run the script:
   ```bash
   python <script_name>.py
   ```

### Key Functions
1. **`ant_colony_optimization(weights, num_bins, p, e)`**
   - Main function to run the ACO algorithm.
   - Parameters:
     - `weights`: List of item weights.
     - `num_bins`: Number of bins.
     - `p`: Number of ants (paths).
     - `e`: Evaporation rate.

2. **`run_and_average_trials(num_trials, weights, num_bins, num_ants, evaporation_rate, label)`**
   - Runs multiple trials of the ACO algorithm and computes average fitness across trials.

3. **`plot_separate_trials()`**
   - Plots separate graphs for BPP1 and BPP2.

---

## Results
The algorithm generates:
- **Average Best Fitness Graphs** for BPP1 and BPP2.
  - X-axis: Number of Fitness Evaluations.
  - Y-axis: Average Best Fitness.

Graphs demonstrate the performance of the ACO algorithm under different parameter settings.

---

## Parameters
The script includes predefined parameter sets for testing:
- Number of trials: `5`
- Evaporation rates: `0.9` and `0.6`
- Number of ants: `100` and `10`
- Bin counts:
  - BPP1: `10`
  - BPP2: `50`

---

## Customization
You can customize the algorithm parameters by modifying the `bpp1_parameter_sets` and `bpp2_parameter_sets` lists in the `plot_separate_trials` function.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
This project is inspired by the Ant Colony Optimization metaheuristic, which is widely studied in optimization problems.

---

## Contact
For questions or suggestions, please contact:
- **Author**: Adam Okeahalam 
- **Email**: adamokeahalam@gmail.com

Feel free to contribute or raise issues in this repository!

