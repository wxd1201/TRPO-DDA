import gymnasium as gym

from algorithms.trpo_dda import TRPODDAAgent

def main():

    env = gym.make("Swimmer-v5")

    agent = TRPODDAAgent(
        state_space=env.observation_space,
        action_space=env.action_space
    )

    print("TRPO-DDA training framework initialized.")

    # training loop omitted

if __name__ == "__main__":
    main()
