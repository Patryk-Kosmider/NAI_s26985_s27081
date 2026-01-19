"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

System RL do gry Freeway w Atari
System wykorzystuje algorytm Deep Q-Network (DQN) do nauki gry Freeway w środowisku Atari.
Model jest trenowany wraz z zapisem postępów w folder /logs, co określoną liczbę kroków.
W przypadku przerwania treningu (Ctrl+C), model jest zapisywany lokalnie.
W trybie treningowym środowisko działa w trybie "rgb_array" - konsola (minimalizacja zasobów potrzebnych do treningu by zmniejszyć czas)
, natomiast w trybie testowym w trybie "human" - wizualny podgląd, jak model radzi sobie w grze.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install gymnasium[atari] stable-baselines3 ale-py shimmy autorom

Przykładowe uruchomienia:
    python rl_freeway.py --mode train --steps 1000000 (rozpoczęcie nauki)
    python rl_freeway.py --mode test (uruchomienie podglądu wyuczonego agenta)

"""

import argparse
import ale_py
import gymnasium as gym
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

gym.register_envs(ale_py)

def initialize_environment(render_mode="rgb_array"):
    env = gym.make("ALE/Freeway-v5", render_mode=render_mode)
    env = AtariWrapper(env)
    return env

def build_model(env):
    return DQN(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.4,
        exploration_final_eps=0.01,
        verbose=1,
        device="auto",
    )

def train_model(steps=1000000):
    env = initialize_environment()
    model = build_model(env)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, save_path="logs", name_prefix="model_freeway"
    )

    print(f"Trening modelu DQN: {steps} kroków")
    try:
        model.learn(total_timesteps=steps, callback=checkpoint_callback)
        params = model.get_parameters()
        with open("dqn_freeway_weights.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Koniec treningu, wagi modelu zapisane do dqn_freeway_weights.pkl")
    except KeyboardInterrupt:
        params = model.get_parameters()
        with open("dqn_freeway_weights.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Trening przerwany, wagi zapisane")
    env.close()

def test_and_evaluate_model(weights_path="dqn_freeway_weights.pkl", episodes=5):
    env = initialize_environment(render_mode="human")
    
    model = build_model(env)
    
    try:
        with open(weights_path, "rb") as f:
            params = pickle.load(f)
        model.set_parameters(params)
        print(f"Pomyślnie wczytano wagi z {weights_path}")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku wag {weights_path}")
        return

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Epizod {episode + 1}: Zdobyte punkty: {total_reward}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Freeway Agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()

    if args.mode == "train":
        train_model(steps=args.steps)
    elif args.mode == "test":
        test_and_evaluate_model()

