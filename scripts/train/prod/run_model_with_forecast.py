#!/usr/bin/env python3
"""
Ejemplo: correr model.zip solo e ingresar el forecast como variable paso a paso.
Si entrenaste con nuestroMultrizona.yaml (ForecastWrapper horizon=24), la observación
es: base (19) + forecast (24*6=144) = 163 dimensiones.
"""
import numpy as np
from stable_baselines3 import SAC  # o PPO, TD3, según tu modelo

# Config igual que nuestroMultrizona.yaml (ForecastWrapper)
MODEL_PATH = "model.zip"
FORECAST_CSV = "/workspaces/sinergym/sinergym/data/weather/forecast/forecast_24h.csv"
HORIZON = 24  # mismo que en nuestroMultrizona.yaml
BASE_DIM = 19  # 3 time + 15 variables + 1 meter

# Importar la utilidad de forecast (mismo directorio que este script)
import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from forecast_utils import get_forecast_vector


def build_observation(obs_base: np.ndarray, timestep: int) -> np.ndarray:
    """
    Construye la observación completa: base + forecast.
    obs_base: vector de sensores (ej. 19 dims: time + variables + meter).
    timestep: paso actual (0, 1, 2, ...).
    """
    forecast_vec = get_forecast_vector(timestep, FORECAST_CSV, HORIZON)
    return np.concatenate([np.asarray(obs_base, dtype=np.float32), forecast_vec], axis=0)


def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    # Modelo entrenado con nuestroMultrizona: espera obs de 163 dims (19 + 24*6)
    expected_dim = BASE_DIM + HORIZON * 6
    print(f"Observación esperada: {expected_dim} dims (base {BASE_DIM} + forecast {HORIZON}*6)")

    timestep = 0
    # Orden igual que nuestroMultrizona: month, day_of_month, hour + 15 variables + 1 meter
    obs_base = np.zeros(BASE_DIM, dtype=np.float32)
    # obs_base[0]=month, [1]=day_of_month, [2]=hour
    # [3:7]=heating_rate N/S/E/W, [7]=outdoor_temperature, [8]=outdoor_humidity
    # [9:13]=air_temperature N/S/E/W, [13:17]=air_humidity N/S/E/W, [17]=heat_pump_power, [18]=total_electricity_HVAC

    obs_full = build_observation(obs_base, timestep)
    assert obs_full.shape[0] == expected_dim, f"Obs {obs_full.shape[0]} != {expected_dim}"
    action, _ = model.predict(obs_full, deterministic=True)
    print("Action:", action)

    timestep += 1
    obs_full = build_observation(obs_base, timestep)
    action, _ = model.predict(obs_full, deterministic=True)
    print("Action:", action)


if __name__ == "__main__":
    main()
