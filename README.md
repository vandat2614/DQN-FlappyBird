# DQN-FlappyBird

A reinforcement learning agent trained with **Deep Q-Network (DQN)** to play **Flappy Bird**, built using PyTorch and Gymnasium.

![Gameplay Demo](assets\demo.gif)
---

## 🌍 Environment

This project uses the external environment [`FlappyBird-v0`](https://github.com/markub3327/flappy-bird-gymnasium), a Gymnasium-compatible implementation of the classic Flappy Bird game.

This code is designed to work with the vector-based observation space, which is enabled by setting `use_lidar=False`.  If you're interested in the meaning of each feature, please refer to the original repository.


---

## 📂 Project Structure

```
DQN-FlappyBird/
├── config.yaml              # Contains settings for training, evaluation, and model architecture
├── main.py                  # Entry point to load env, model, and run
├── src/
│   ├── train.py             # Training logic for training the agent
│   ├── test.py              # Demo: run a single episode with a trained agent
│   ├── eval.py              # Generates statistics and evaluation metrics
│   ├── neural_network.py
│   └── experience_replay.py # Experience replay buffer
├── runs/
    ├── train/
    │   ├── weights/         # Checkpoints: best.pt, last.pt
    │   └── log              # Text log of training performance
    └── eval/
        └── results.json     #  Stores evaluation results:

```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vandat2614/DQN-FlappyBird.git
cd DQN-FlappyBird
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training

```bash
python main.py --mode train
```

- Best model is saved at: `runs/train/weights/best.pt`
- Training parameters (e.g., learning rate, epsilon decay...) are specified under the `train` section of config
---

## 🧪 Evaluate

```bash
python main.py --mode eval --config_path config.yaml --weights_path path/to/your/weights.pt
```

- Loads a trained `.pt` model to evaluate the agent's performance in the environment.
- Results are saved to `runs/eval/eval_log.json`
- Evaluation parameters (e.g., number of episodes) are specified under the `evaluate` section of config
- It’s recommended to set a `limit_score` in the `env` section of the config to prevent episodes from running indefinitely.
---

## 🎮 Test


```bash
python main.py --mode test --config_path config.yaml --weights_path runs/train/weights/best.pt
```

- Runs a single episode with the trained agent and renders it in real-time.