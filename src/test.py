import torch
import time

def test(env, model, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = model.to(device)
    model.eval()

    state, info = env.reset()
    while True:

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            break
