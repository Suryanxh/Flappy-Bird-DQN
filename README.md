# Flappy Bird - Deep Q-Learning

A Reinforcement Learning project that trains an AI agent to play Flappy Bird using a Deep Q-Network (DQN).

The agent learns by interacting with the environment, receiving rewards, storing experiences, and updating its neural network to improve its gameplay decisions.

## Features

- Deep Q-Network (DQN)
- Experience Replay
- Epsilon-Greedy Exploration
- Target Network Synchronization
- Configurable Hyperparameters
- Best Model Checkpoint Saving
- Training Logs
- CPU / CUDA / Apple MPS Support
- Training and Evaluation Modes

## Tech Stack

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- NumPy
- PyYAML

## Project Structure

```text
Flapp-Bird/
│
├── agent.py                 # Main training and evaluation script
├── dqn.py                   # Deep Q-Network architecture
├── experience_replay.py     # Replay memory implementation
├── parameters.yaml          # Hyperparameter configuration
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
└── runs/                    # Saved models and training logs
```

## How It Works

The agent follows the Deep Q-Learning pipeline:

1. Observe the current game state.
2. Select an action using an epsilon-greedy policy.
3. Execute the action in the environment.
4. Receive a reward and the next state.
5. Store the experience in replay memory.
6. Sample a mini-batch of experiences.
7. Train the policy network.
8. Periodically synchronize the target network.
9. Save the best-performing model.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Flappy-Bird-DQN.git
cd Flappy-Bird-DQN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the agent in training mode:

```bash
python agent.py flappybirdv0 --train
```

The parameter name must match the configuration key present in `parameters.yaml`.

## Evaluation

After training, run the saved model:

```bash
python agent.py flappybirdv0
```

The trained agent will play Flappy Bird using the saved model.

## Hyperparameters

The training configuration is stored in `parameters.yaml`.

It includes:

- Learning rate
- Discount factor
- Initial epsilon
- Minimum epsilon
- Epsilon decay
- Replay memory size
- Mini-batch size
- Target network synchronization rate
- Reward threshold

## Model Saving

The best-performing model and training logs are saved in the `runs/` directory.

Example:

```text
runs/
├── flappybirdv0.pt
└── flappybirdv0.log
```

## Reinforcement Learning Concepts

This project demonstrates:

- Markov Decision Processes
- Q-Learning
- Deep Q-Networks
- Bellman Equation
- Experience Replay
- Target Networks
- Epsilon-Greedy Policy
- Temporal Difference Learning

## Future Improvements

- Add training reward graphs
- Track average reward over episodes
- Implement Double DQN
- Implement Dueling DQN
- Add TensorBoard monitoring
- Improve checkpoint management
- Compare different hyperparameter configurations

## Author

**Suryansh**

B.Tech Computer Science and Engineering

## License

This project is intended for educational and fun purposes.
