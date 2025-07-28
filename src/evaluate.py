import os
import torch
import json
import numpy as np

def evaluate(env, model, config, device):

    model = model.to(device)
    model.eval()

    num_episodes = config['num_episodes']
    base_dir = config['base_dir']
    eval_dir = os.path.join(base_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    scores = []

    for _ in range(num_episodes):
        state, _ = env.reset()

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                scores.append(info['score'])
                break

    stats = {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "avg": float(np.mean(scores)),
        "std": float(np.std(scores)),
        # "scores": scores,
    }

    with open(os.path.join(eval_dir, 'results.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    print("Evaluation completed. Results saved to", os.path.join(eval_dir, 'results.json'))
