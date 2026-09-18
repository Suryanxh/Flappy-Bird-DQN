# Flappy Bird - Deep Q-Learning

A Reinforcement Learning project that trains an AI agent to play Flappy Bird using Deep Q-Learning (DQN).

The agent learns by interacting with the environment, collecting rewards, and improving its action-selection policy through experience replay and a target network.

## Features

- Deep Q-Network (DQN)
- Experience Replay
- Epsilon-Greedy Exploration
- Target Network Synchronization
- Configurable Hyperparameters
- Model Checkpoint Saving
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
├── parameters.yaml          # Hyperparameters
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
└── runs/                    # Saved models and logs
