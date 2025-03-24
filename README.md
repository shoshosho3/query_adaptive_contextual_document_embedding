<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg" alt="Technion Logo" height="100">
  <br>
  Advanced Information Retrieval - Final Project
</h1>

<p align="center">
  <em>
    Query Adaptive Contextual Document Embedding
  </em>
</p>

<p align="center">
  <strong>Technion - Israel Institute of Technology</strong> <br>
  Faculty of Data Science and Decisions
</p>

<h1 align="center">
  <img src="https://github.com/shoshosho3/query_adaptive_contextual_document_embedding/blob/main/pictures_QACDE/IR_Logo_1.png" alt="IR Logo 1" height="200">
  <img src="https://github.com/shoshosho3/query_adaptive_contextual_document_embedding/blob/main/pictures_QACDE/IR_Logo_2.png" alt="IR Logo 2" height="200">
</h1>

---

<details open>
<summary><strong>Table of Contents</strong> ⚙️</summary>

1. [About the Article](#link-of-the-article)
2. [Project Overview](#project-overview)  
3. [About the code](#about-the-code)
4. [Implemented Classes](#implemented-classes)
5. [Running Instructions](#running-instructions)  
6. [Results & Comments](#results-&-comments)  

</details>

---

## About the article
Our project is an extension on the following article:
<blockquote>
  <a href="https://arxiv.org/abs/2410.02525">Contextual Document Embedders</a> by John X. Morris and Alexander M. Rush.</blockquote>

The article "Contextual Document Embeddings" by John X. Morris and Alexander M. Rush from Cornell University proposes a method to improve document embeddings by incorporating context from surrounding documents. Traditional embeddings, which rely solely on the individual document, can be insufficient for highly specific information retrieval tasks. This paper addresses this by incorporating neighboring documents into the embedding process, making the representation more context-aware. The authors introduce a contextual training objective and an architecture that explicitly encodes information from neighboring documents, demonstrating improved performance in various scenarios.

## Project Overview
In this project, we aimed to extend the "Contextual Document Embeddings" model by introducing a query-adaptive approach. While the original model effectively contextualizes document embeddings by considering neighboring documents, it does not account for the user's needs expressed in the query, which is essential for information retrieval tasks.

To achieve this, we developed two models to address this. The first model makes the document embedding query-adaptive, meaning that the document embedding is influenced by the query's embedding, following the same embedding method proposed in the original paper. However, upon reflection, we realized that the document embedding greatly depends on how the query is represented. This led us to develop a second model that computes the document embedding using multiple query embeddings, such as BERT and TF-IDF. The goal of this approach is to combine the strengths of different query embedding methods to create more robust and informative document embeddings. By leveraging multiple representations of the query, we aim to capture various aspects of the document-query relationship, improving the relevance of the information retrieval process.


## About the code
The code replicates the experimental results from the paper. The goal is to create a virtual environment for blocking bandits by reproducing the different simulations described in Section V of the paper. We replicated three types of simulations: the first involves arms with uniformly sampled short delays, the second with arms having uniformly sampled long delays, and the third with fixed delays. The code runs the two algorithms presented in the paper, namely Oracle Greedy and UCB Greedy, across these different simulations. It then generates graphs showing the evolution of cumulative rewards and cumulative regret for each algorithm depending on the type of simulation.


## Implemented Classes
- **bandits.py**
  - ### BanditArm Class:
    - `__init__(self, mean_reward)`: Initializes the bandit arm with the given mean reward and tracks the number of uses.
    - `__repr__(self)`: Returns a string representation of the bandit arm's mean reward.
    - `__eq__(self, other)`: Checks if two bandit arms have the same mean reward.
    - `sample_reward(self)`: Samples a binary reward from a binomial distribution based on the arm’s mean reward.

  - ### BlockingBanditArm Class (inherits from BanditArm):
    - `__init__(self, mean_reward, blocking_delay)`: Initializes a blocking bandit arm with a mean reward and a blocking delay.
    - `__repr__(self)`: Returns a string representation of the blocking bandit arm, including its blocking delay.
    - `__eq__(self, other)`: Checks if two blocking bandit arms have the same mean reward and blocking delay.
    - `sample_reward(self)`: Samples a reward if the arm is available, otherwise returns 0.

- **simulations.py**
  - ### BlockingBanditSimulation Class:

    - `__init__(self, K, T, sim_type, fixed_delays, seed)`: Initializes the simulation with the number of arms, horizon, type of simulation, fixed delays, and a random seed.
    - `generate_bandits(self)`: Generates bandit arms based on the type of simulation (small delays, large delays, or fixed delays).
    - `oracle_greedy_algorithm(self)`: Selects the best available arm based on the oracle greedy algorithm and updates its state.
    - `UCB_greedy_algorithm(self, timestamp)`: Implements the UCB greedy algorithm, selecting arms and updating their state based on the UCB index.
    - `update_unavailable_arms(self)`: Decreases the blocking delay countdown for unavailable arms and makes them available when the delay ends.
    - `simulate(self)`: Runs the simulation for T time steps, returning the cumulative rewards and the arms chosen by both algorithms.
    - `calculate_cumulative_regret(self, ucb_arms)`: Computes the cumulative regret for the UCB algorithm.
    - `reset_simulation(self)`: Resets the state of all arms, making them available for reuse.
    - `calculate_k_star(self)`: Calculates the optimal number of arms, `K*`, based on the sum of inverse delays.
    - `calculate_k_g(self, oracle_chosen_arms)`: Computes `K_g`, the number of arms used by the Oracle Greedy algorithm.

- **main.py**
  - `parse_arguments()`: Parses the command-line arguments such as the number of arms, rounds, fixed delay, number of simulations, and seed.
  - `plot_graph_cumulative_rewards(simulation_types, results_values, algorithms, T, k_stars_types, k_g_types)`: Plots the cumulative rewards for different algorithms and simulation types.
  - `plot_graph_cumulative_regrets(simulation_types, regrets_values, T, k_stars_types, k_g_types)`: Plots the cumulative regrets for the UCB algorithm across different simulation types.
  - `deterministic_delays_simulations(simulation_types, K, T, fixed_delays, num_sims, seed)`: Runs the simulations for different types of delays and collects results on rewards and regrets.
  - `__main__`: Parses the arguments, sets up the seed, and runs the deterministic delays simulations.


## Running Instructions
This repository contains the code and instructions to run the experiment presented in the associated paper. Please follow the steps below to set up and run the experiment

1. Download and install the dependencies by running the following command in your terminal:
```python
pip install -r requirements.txt
```

2. Run the following command in your terminal to execute the experiment:
```python
python "your_local_path"/project_files/main.py -K 20 -T 10000 -seed 42 -fixed_delay 10 -num_sim 250
```
Those hyperparameters match exactly those of the paper's experiment.

4. You can modify the values of the following parameters to generate different simulations:

- `-K`: The number of arms of the simulation. For example, `-K 20` will run a simulation with 20 arms.
- `-T`: The time duration for each simulation in arbitrary units. For example, `-T 10000` will run each simulation for 10,000 timeslots.
- `-seed`: The random seed for reproducibility. For example, `-seed 42` will set the seed to 42.
- `-fixed_delay`: The value of the delay for the simulation with fixed delay. For example, `-fixed_delay 10` sets a fixed delay of 10 timeslots.
- `-num_sim`: The number of individual simulations to run. For example, `-num_sim 250` will run 250 simulations.

Feel free to adjust these values to explore different scenarios or to generate alternative results based on your preferences.


## Results & Comments
Here are the graphs generated by the simulation:
<h1 align="center">
  <img src="https://github.com/tombijaoui/Sequential-Decision-Making-Project/blob/main/pictures_MAB/Cumulative%20Rewards.png" alt="Cumulative Rewards" height="350">
  <img src="https://github.com/tombijaoui/Sequential-Decision-Making-Project/blob/main/pictures_MAB/Cumulative%20Regrets.png" alt="Cumulative Regrets" height="350">
</h1>

1. **Cumulative Reward**: 
   - Simulations with smaller delays achieve higher cumulative rewards compared to those with fixed or larger delays. This is because smaller delays allow more frequent optimal arm pulls, increasing exploitation.
   - All cumulative rewards grow linearly with time, likely due to the low variance in arm rewards.
   - The Oracle algorithm consistently outperforms UCB since it knows the optimal arm at each step, unlike UCB, which must estimate it. However, with longer delays, UCB and Oracle results converge, as Oracle is forced to explore suboptimal arms.

2. **Cumulative Regret**: 
   - Cumulative regret decreases over time across all simulations, as UCB explores enough to estimate rewards more accurately.
   - Lower delays lead to higher cumulative regret due to the difficulty in exploration, potentially resulting in premature suboptimal decisions.
   - Slight negative cumulative regret at the end suggests UCB's arm selection surpasses Oracle's greedy strategy in some cases.

