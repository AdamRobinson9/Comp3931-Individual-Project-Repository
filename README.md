# COMP3931 - Individual Project: Code Repository

This repository contains the full code implementation and experimental results to investigate *Reinforcement Learning for Strategy Development in a Novel
Pursuit Evasion Game*.

The software implements the *Capture the Flag* game as described in the project report, played on an 8x8 grid graph. This is achieved in various stages as follows:
* A playable game environment, allowing two players to play against each other. ([Version1-Playable.py](Version1-Playable.py))
* Fixed strategies, introducing artificial players to make strategical decisions. ([Version2-Fixed_Strategies.py](Version2-Fixed_Strategies.py), [Strategies.py](Strategies.py), [Placement.py](Placement.py))
* Adaptive strategies, using a contextual bandit & hyper-heuristic framework. ([Version3.1-RL_Evaders.py](Version3.1-RL_Evaders.py), [Version3.2-RL_Pursuers.py](Version3.2-RL_Pursuers.py)).
* Automated experiment runners, to collect performance data of different strategies across 9 team configurations ([Baseline_Results.py](./Experiment%20Results/Baseline_Results.py), [RL_Pursuers_Results.py](./Experiment%20Results/RL_Pursuers_Results.py), [RL_Evaders_Results.py](./Experiment%20Results/RL_Evaders_Results.py))

## Repository Structure

```
├── Placement.py
├── Strategies.py
├── Version1-Playable.py
├── Version2-Fixed_Strategies.py
├── Version3.1-RL_Evaders.py
├── Version3.2-RL_Pursuers.py
│
├── Experiment Results/
│   ├── Baseline_Results.py
│   ├── RL_Evaders_Results.py
│   ├── RL_Pursuers_Results.py
│   ├── Learning_Curves.ipynb
│   ├── Data_Aggregation.ipynb
│   ├── *.csv files (Simulation results)
│   └── Q-Tables/
└──
```

## Results Files
• The results of all of the experiments can be found in [final_results.csv](./Experiment%20Results/final_results.csv)
* This contains data aggregated into one dataset from:
    * [Baseline_Results.csv](./Experiment%20Results/Baseline_Results.csv)
    * [rl_pursuers_results.csv](./Experiment%20Results/rl_pursuers_results.csv)
    * [rl_evaders_results.csv](./Experiment%20Results/rl_evaders_results.csv)
    
    Using [Data_Aggregation.ipynb](./Experiment%20Results/Data_Aggregation.ipynb).

The results of RL experiments using different epsilon values can also be found in:
* [rl_evaders_results_epsilon1.csv](./Experiment%20Results/rl_evaders_results_epsilon1.csv) / [rl_pursuers_results_epsilon1.csv](./Experiment%20Results/rl_pursuers_results_epsilon1.csv)
* [rl_evaders_results_epsilon3.csv](./Experiment%20Results/rl_evaders_results_epsilon3.csv) / [rl_pursuers_results_epsilon3.csv](./Experiment%20Results/rl_pursuers_results_epsilon3.csv)
* [rl_evaders_results_epsilon4.csv](./Experiment%20Results/rl_evaders_results_epsilon4.csv) / [rl_pursuers_results_epsilon4.csv](./Experiment%20Results/rl_pursuers_results_epsilon4.csv)

These files are used to plot the learning curves in [Learning_Curves.ipynb](./Experiment%20Results/Learning_Curves.ipynb)

The Q-tables used in section 5 of the project report can also be found in [Q-Tables](./Experiment%20Results/Q-Tables/).

## Dependencies

The code is written using Python 3.12, utilising the following libraries:

| Library  | Use |
| ------------- |:-------------:|
| NetworkX      | Graph representation & operations   |
| Matplotlib      | Graph visualisation     |
| Numpy      | Median calculations & data handling     |
| Scipy | Used within NetworkX|

No external datasets were used and all data was collected through simulations.

## Running the Code

1. Install `git` and `python` (>=3.12)
2. Open the terminal / command prompt.
3. Install dependencies:
```
pip install networkx matplotlib numpy scipy
```
4. Clone the repository:
```
git clone https://github.com/AdamRobinson9/Comp3931-Individual-Project-Repository.git
cd Comp3931-Individual-Project-Repository
```
5. Run the code:

    **Playing the game**
    ```
    python Version1-Playable.py
    ```
    Launches a command line game, with the graph displayed using matplotlib, where both teams are controlled manually.
    Pursuers are red, evaders are blue and the flag carrier has a yellow ring. Enter vertex numbers when prompted to move each agent.

    **Running fixed strategies**
    ```
    python Version2-Fixed_Strategies.py
    ```
    Runs a single game between fixed strategy agents, use enter to advance one turn. 
    
    Note, the fixed strategies in this version are arbitrary and may not represent results displayed in the project report. 
    Edit the team sizes and strategies by changing the `num_pursuers`/`num_evaders` and `pursuer_roles`/`evader_roles` variables within the file.

    **Running the reinforcement learning agents**
    ```
    python Version3.1-RL_Evaders.py
    python Version3.2-RL_Pursuers.py
    ```
    Trains the RL agent for 200 episodes against the fixed opponent that it is playing aginst and runs a single visualised game using the trained policy. Use enter to advance one turn. 
    
    Note, the fixed strategies in this version are arbitrary and may not represent results displayed in the project report. 
    Edit the team sizes and strategies by changing the `num_pursuers`/`num_evaders` and `pursuer_roles`/`evader_roles` variables within the file.

    **Running the experiments**

    ```
    cd "Experiment Results"

    python Baseline_Results.py
    python RL_Evaders_Results.py
    python RL_Pursuers_Results.py
    ```
    Runs the experiments as described in the project report and saves the files to a csv file.
    Results can be found in [Experiment Results](./Experiment%20Results/) folder.

    Note, each experiment runner file is likely to take several minutes to run.
