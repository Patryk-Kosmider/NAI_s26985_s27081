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
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

gym.register_envs(ale_py)


def initialize_environment(render_mode="rgb_array"):
    """
    Inicjalizuje środowisko gry Freeway z Atari.
    :param render_mode: tryb renderowania (domyślnie "rgb_array" dla treningu, "human" dla testów)
    :return: Srodowisko gry
    """
    env = gym.make("ALE/Freeway-v5", render_mode=render_mode)
    # AtariWrapper - skalowanie do 84x84, frame stacking
    env = AtariWrapper(env)
    return env


def build_model(env):
    """
    Buduje model DQN dla podanego środowiska.
    :param env: Srodowisko gry
    :return: Model DQN
    """
    return DQN(
        "CnnPolicy",  # Sieć konwolucyjna dla obrazów
        env,
        learning_rate=1e-4,
        buffer_size=100000,  # Przechowuje 100k ostatnich przejść by zapełnić bufor replay
        learning_starts=5000,  # 5k losowych ruchów przed rozpoczęciem treningu
        batch_size=32,
        tau=1.0,  # Aktualizacja sieci docelowej
        gamma=0.99,  # Dyskont przyszłych nagród, bliższe 1 oznacza większą wagę dla przyszłych nagród. U nas nagroda jest tylko na samej górze, więc musimy mieć wysoką gammę
        train_freq=4,  # Aktualizacja co 4 kroki
        target_update_interval=1000,  # Aktualizacja sieci co 1000 kroków
        exploration_fraction=0.15,  # Przez początkowe 15% szansa eksploracji maleje liniowo
        exploration_final_eps=0.01,  # Docelowa szansa na losowy ruch
        verbose=1,
        device="auto",
    )


def train_model(steps=1000000):
    """
    Trenuje model DQN przez określoną liczbę kroków.
    Zapisuje model po zakończeniu treningu lub przerwaniu.
    :param steps: Ilość kroków treningu
    :return: None
    """
    env = initialize_environment()
    model = build_model(env)
    # Callback do zapisywania modelu co określoną liczbę kroków
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, save_path="logs", name_prefix="model_freeway"
    )

    print(f"Trening modelu DQN: {steps} kroków")
    try:
        model.learn(total_timesteps=steps, callback=checkpoint_callback)
        model.save("dqn_freeway_model")
        print("Koniec treningu, model zapisany")
    except KeyboardInterrupt:
        model.save("dqn_freeway_model")
        print("Trening przerwany, model zapisany")
    env.close()


def test_and_evaluate_model(model_path="dqn_freeway_model", episodes=5):
    """
    Testuje wytrenowany model DQN w trybie wizualnym.
    :param model_path: Scieżka do zapisanego modelu
    :param episodes: Ilość epizodów testowych
    :return: None
    """
    env = initialize_environment(render_mode="human")
    model = DQN.load(model_path, env=env)

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # deterministyczny wybór - wybiera akcje o najwyższej przewidywanej wartości Q
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Epizod {episode + 1}: Zdobyte punkty: {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Freeway Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Tryb pracy",
    )
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Liczba kroków treningu"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model(steps=args.steps)
    elif args.mode == "test":
        test_and_evaluate_model(episodes=5)
