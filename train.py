import gymnasium as gym

from algorithms.trpo_dda import TRPODDAAgent
from utils.trainer import OnPolicyTrainer

def main():

    env_name = "Swimmer-v5"

    env = gym.make(env_name)

    agent = TRPODDAAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=128
    )

    trainer = OnPolicyTrainer(
        env=env,
        agent=agent,
        num_episodes=6000
    )

    trainer.train()

if __name__ == "__main__":
    main()
