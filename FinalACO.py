import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

"""e: evaporation rate"""
"""p: number of ant paths"""

# Problem parameters
n_items = 500  # Number of items for both BPP problems
bins_bpp1 = 10  # Number of bins for BPP1
bins_bpp2 = 50  # Number of bins for BPP2

# Generate item weights for BPP1 and BPP2
weights_bpp1 = [i for i in range(1, n_items + 1)]  # Weights from 1 to 500
weights_bpp2 = [(i ** 2) / 2 for i in range(1, n_items + 1)]  # Weights of (i^2)/2

# ACO parameters
max_fitness_evaluations = 10000  # Termination criterion: maximum number of fitness evaluations

# Function to initialize pheromone tables
# Initializes a pheromone table with random values for each item and bin.
def initialize_pheromone_table(num_items, num_bins):
    pheromone_table = [[random.uniform(0, 1) for _ in range(num_bins)] for _ in range(num_items)]
    return pheromone_table

# Function to generate an ant's path
# Simulates an ant's decision-making process, choosing a bin for each item based on pheromone levels.
def generate_ant_path(pheromone_table, num_bins):
    path = []
    for item_index in range(len(pheromone_table)):
        pheromones = pheromone_table[item_index]
        total_pheromone = sum(pheromones)
        # Calculate selection probabilities
        probabilities = [pheromone / total_pheromone for pheromone in pheromones]
        # Choose a bin based on probabilities
        bin_choice = random.choices(range(num_bins), weights=probabilities, k=1)[0]
        path.append(bin_choice)
    return path

# Function to calculate fitness of a path
# Calculates the fitness of an ant's path by finding the difference between the heaviest and lightest bin loads.
def calculate_fitness(path, weights, num_bins):
    bin_weights = [0] * num_bins
    for item_index, bin_index in enumerate(path):
        bin_weights[bin_index] += weights[item_index]
    heaviest = max(bin_weights)
    lightest = min(bin_weights)
    fitness = heaviest - lightest  # Difference between heaviest and lightest bins
    return fitness, bin_weights

# Function to update pheromone table
# Updates the pheromone table based on the fitness of each ant's path; better fitness leads to a higher deposit.
def update_pheromone(pheromone_table, ant_paths, ant_fitnesses):
    for ant_index, path in enumerate(ant_paths):
        fitness = ant_fitnesses[ant_index]
        pheromone_deposit = 100 / fitness if fitness != 0 else 0  # Avoid division by zero
        for item_index, bin_index in enumerate(path):
            pheromone_table[item_index][bin_index] += pheromone_deposit
    return pheromone_table

# Function to evaporate pheromones
# Reduces pheromone levels over time, which encourages exploration by weakening the influence of previous paths.
def evaporate_pheromones(pheromone_table, e):
    for item_index in range(len(pheromone_table)):
        for bin_index in range(len(pheromone_table[item_index])):
            pheromone_table[item_index][bin_index] *= (e)
    return pheromone_table

# Main ACO function
# Runs the Ant Colony Optimization (ACO) algorithm, simulating multiple ants' paths and updating pheromones.
def ant_colony_optimization(weights, num_bins, p, e):
    num_items = len(weights)
    pheromone_table = initialize_pheromone_table(num_items, num_bins)
    evaluations = 0
    best_fitness = float('inf')
    best_bin_weights = []
    best_path = []
    
    # Histories to store data for plotting
    best_fitness_history = []     # List to store best fitness per evaluation
    evaluations_history = []      # List to store the number of evaluations
    
    # Loop until maximum fitness evaluations are reached
    while evaluations < max_fitness_evaluations:
        ant_paths = []
        ant_fitnesses = []
        
        # Generate paths for each ant
        for _ in range(p):
            if evaluations >= max_fitness_evaluations:
                break  # Ensure we do not exceed the maximum evaluations
            path = generate_ant_path(pheromone_table, num_bins)
            fitness, bin_weights = calculate_fitness(path, weights, num_bins)
            ant_paths.append(path)
            ant_fitnesses.append(fitness)
            evaluations += 1

            # Update best solution found
            if fitness < best_fitness:
                best_fitness = fitness
                best_bin_weights = bin_weights.copy()
                best_path = path.copy()

            # Record every evaluation in the histories
            evaluations_history.append(evaluations)
            best_fitness_history.append(best_fitness)
        
        # Update pheromone table based on ant paths and fitnesses
        pheromone_table = update_pheromone(pheromone_table, ant_paths, ant_fitnesses)
        # Evaporate pheromones
        pheromone_table = evaporate_pheromones(pheromone_table, e)
    
    # Return the best solution found and histories for plotting
    return best_fitness, best_bin_weights, best_path, evaluations_history, best_fitness_history

# Function to run and average trials
# Runs multiple trials of the ACO algorithm, averaging the fitness history over trials for each parameter set.
def run_and_average_trials(num_trials, weights, num_bins, num_ants, evaporation_rate, parameter_set_label):
    # Lists to store histories for all trials
    all_best_fitness_histories = []
    best_fitness_per_trial = []  # List to store best fitness from each trial
    
    # Run the algorithm for the specified number of trials and collect data
    for trial in range(num_trials):
        print(f"Solving {parameter_set_label}: Trial {trial + 1}")
        best_fitness, _, _, evaluations_history, best_fitness_history = ant_colony_optimization(
            weights, num_bins, num_ants, evaporation_rate
        )
        all_best_fitness_histories.append(best_fitness_history)
        best_fitness_per_trial.append(best_fitness)  # Store the best fitness of each trial

    # Print the best fitness for each trial
    for i, fitness in enumerate(best_fitness_per_trial, 1):
        print(f"Best fitness for {parameter_set_label}, Trial {i}: {fitness}")

    # Compute the average best fitness over trials
    min_len = min(map(len, all_best_fitness_histories))  # Ensure equal length for averaging
    truncated_histories = [history[:min_len] for history in all_best_fitness_histories]
    avg_best_fitness = np.mean(truncated_histories, axis=0)
    
    return evaluations_history[:min_len], avg_best_fitness, f"{parameter_set_label} (p={num_ants}, e={evaporation_rate})"

# Function to plot average fitness separately for BPP1 and BPP2
def plot_separate_trials():
    # Define parameter sets for BPP1 and BPP2
    bpp1_parameter_sets = [
        (5, weights_bpp1, bins_bpp1, 100, 0.9, "BPP1"),
        (5, weights_bpp1, bins_bpp1, 100, 0.6, "BPP1"),
        (5, weights_bpp1, bins_bpp1, 10, 0.9, "BPP1"),
        (5, weights_bpp1, bins_bpp1, 10, 0.6, "BPP1")
    ]
    
    bpp2_parameter_sets = [
        (5, weights_bpp2, bins_bpp2, 100, 0.9, "BPP2"),
        (5, weights_bpp2, bins_bpp2, 100, 0.6, "BPP2"),
        (5, weights_bpp2, bins_bpp2, 10, 0.9, "BPP2"),
        (5, weights_bpp2, bins_bpp2, 10, 0.6, "BPP2")
    ]
    
    # Plot for BPP1
    plt.figure(figsize=(10, 6))
    for num_trials, weights, num_bins, num_ants, evaporation_rate, label in bpp1_parameter_sets:
        evaluations, avg_best_fitness, param_label = run_and_average_trials(
            num_trials, weights, num_bins, num_ants, evaporation_rate, label
        )
        plt.plot(evaluations, avg_best_fitness, label=param_label)
    
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel(r'Average Best Fitness($\overline{f^x}$)')
    plt.title(r'Average Best Fitness($\overline{f^x}$) for BPP1 Across Multiple Trials')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot for BPP2 with scientific notation on y-axis
    plt.figure(figsize=(10, 6))
    for num_trials, weights, num_bins, num_ants, evaporation_rate, label in bpp2_parameter_sets:
        evaluations, avg_best_fitness, param_label = run_and_average_trials(
            num_trials, weights, num_bins, num_ants, evaporation_rate, label
        )
        plt.plot(evaluations, avg_best_fitness, label=param_label)
    
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel(r'Average Best Fitness($\overline{f^x}$)')
    plt.title(r'Average Best Fitness($\overline{f^x}$) for BPP2 Across Multiple Trials')
    plt.legend()
    plt.grid(True)
    
    # Set y-axis to scientific notation with 10^5
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e5:.2f}e5'))
    plt.show()

# Execute the function to plot separate graphs for BPP1 and BPP2
plot_separate_trials()

