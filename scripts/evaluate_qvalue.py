import torch
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym   # ✅ usar gymnasium
import sinergym
 
# 1. Cargar el entorno de Sinergym
env = gym.make('Eplus-demo-v1')   # cambia por tu entorno específico
 
# 2. Cargar el modelo entrenado (bestmodel.zip)
model = SAC.load("./train/local_confs/Eplus-SAC-training-nuestro_2025-10-31_00-40-res1/evaluation/best_model.zip")
 
 
# 3. Función para calcular Q1 y Q2 de un estado y acción
def get_q_values(model, state, action):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    q1, q2 = model.critic(state_tensor, action_tensor)
    return q1.item(), q2.item()
 
# 4. Función para re‑escalar acciones de [-1,1] al rango del entorno
def rescale_action(action, env):
    low = env.action_space.low
    high = env.action_space.high
    return low + (0.5 * (action + 1.0) * (high - low))
 
# 5. Evaluar un episodio y calcular Q-values paso a paso
obs, _ = env.reset()   # Gymnasium devuelve (obs, info)
done = False
episode_q_values = []
 
while not done:
    # Acción de la política (en [-1,1])
    action, _ = model.predict(obs, deterministic=True)
    # Re‑escalar al rango válido del entorno
    action = rescale_action(action, env).astype(np.float32)
 
    # Calcular Q1 y Q2 para este estado-acción
    q1, q2 = get_q_values(model, obs, action)
    q_min = min(q1, q2)
    episode_q_values.append(q_min)
 
    # Avanzar en el entorno
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
 
# 6. Promedio del Q-value en el episodio
avg_q_value = np.mean(episode_q_values)
print(f"Q-value promedio del episodio: {avg_q_value:.3f}")