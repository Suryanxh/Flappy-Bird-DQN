# agent.py

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
import torch
import torch.nn as nn
import torch.optim as optim
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse
import os


# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


# Directory for saved models and logs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):

        self.param_set = param_set

        # Load hyperparameters
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # File paths
        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def run(self, is_training=True, render=False):

        # Create environment
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=True
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Policy network
        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        if is_training:

            # Replay memory
            memory = ReplayMemory(
                self.replay_memory_size
            )

            epsilon = self.epsilon_init

            # Target network
            target_dqn = DQN(
                num_states,
                num_actions
            ).to(device)

            # Initially synchronize target network
            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            # Optimizer
            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            steps = 0
            best_reward = float("-inf")

        else:

            # Evaluation mode
            if not os.path.exists(self.MODEL_FILE):

                raise FileNotFoundError(
                    f"Model not found: {self.MODEL_FILE}\n"
                    f"Train the agent first using --train."
                )

            # Load trained model
            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )

            policy_dqn.eval()

            # No exploration during testing
            epsilon = 0

        # Run episodes
        for episode in itertools.count():

            state, _ = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )

            episode_rewards = 0
            terminated = False

            while not terminated:

                # Exploration
                if is_training and random.random() < epsilon:

                    action = env.action_space.sample()

                # Exploitation
                else:

                    with torch.no_grad():

                        action = policy_dqn(
                            state.unsqueeze(0)
                        ).argmax(dim=1).item()

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )

                # Episode ends if terminated or truncated
                terminated = terminated or truncated

                # Convert next state to tensor
                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=device
                )

                reward_tensor = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=device
                )

                # Store experience
                if is_training:

                    memory.append(
                        (
                            state,
                            torch.tensor(
                                action,
                                dtype=torch.long,
                                device=device
                            ),
                            next_state,
                            reward_tensor,
                            terminated
                        )
                    )

                    steps += 1

                # Move to next state
                state = next_state

                episode_rewards += reward

            # Print episode results
            print(
                f"Episode = {episode + 1}, "
                f"Reward = {episode_rewards}, "
                f"Epsilon = {epsilon:.4f}"
            )

            if is_training:

                # Epsilon decay
                epsilon = max(
                    epsilon * self.epsilon_decay,
                    self.epsilon_min
                )

                # Train the network
                if len(memory) >= self.mini_batch_size:

                    mini_batch = memory.sample(
                        self.mini_batch_size
                    )

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )

                    # Synchronize target network
                    if steps >= self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        steps = 0

                # Save best model
                if episode_rewards > best_reward:

                    log_msg = (
                        f"Best reward = {episode_rewards} "
                        f"for episode = {episode + 1}"
                    )

                    print(log_msg)

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )

                    best_reward = episode_rewards

                # Stop training if threshold reached
                if episode_rewards >= self.reward_threshold:

                    print("Reward threshold reached!")
                    break

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Unpack experiences
        states, actions, next_states, rewards, terminations = zip(
            *mini_batch
        )

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(
            terminations,
            dtype=torch.float32,
            device=device
        )

        # Calculate target Q-values
        with torch.no_grad():

            target_q = rewards + (
                1 - terminations
            ) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        # Calculate current Q-values
        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # Calculate loss
        loss = self.loss_fn(
            current_q,
            target_q
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test DQN agent."
    )

    parser.add_argument(
        "hyperparameters",
        help="Name of the hyperparameter set in parameters.yaml"
    )

    parser.add_argument(
        "--train",
        help="Training mode",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(
        param_set=args.hyperparameters
    )

    if args.train:

        dql.run(
            is_training=True,
            render=False
        )

    else:

        dql.run(
            is_training=False,
            render=True
        )