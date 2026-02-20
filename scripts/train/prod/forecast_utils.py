"""
Utilidad para inyectar forecast meteorológico como variable paso a paso (sin ForecastWrapper).
Uso: en cada paso llamas get_forecast_vector(timestep, ...) y concatenas el resultado a tu observación.
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def get_forecast_vector(
    timestep: int,
    forecast_csv: str,
    horizon: int,
    index_col: str = "timestep",
) -> np.ndarray:
    """
    Devuelve el vector de forecast para el timestep actual (próximas `horizon` filas del CSV).

    Así puedes añadir el forecast como variable paso a paso sin usar el ForecastWrapper:
    en cada paso llamas esta función con el timestep actual y concatenas el resultado
    a tu vector de observación.

    Args:
        timestep: Paso actual de simulación (0, 1, 2, ...).
        forecast_csv: Ruta al CSV con columnas de clima e índice timestep.
        horizon: Número de filas de forecast (ej. 5 = próximas 5 horas si el CSV es horario).
        index_col: Nombre de la columna que es el índice (por defecto "timestep").

    Returns:
        Vector 1D de shape (horizon * n_columnas,) con las filas t, t+1, ..., t+horizon-1
        aplanadas. Si no hay suficientes filas, se rellena con la última disponible.
    """
    if not os.path.isfile(forecast_csv):
        return np.zeros(horizon * 6, dtype=np.float32)  # fallback: 6 columnas típicas

    df = pd.read_csv(forecast_csv, index_col=index_col)
    feat_n = df.shape[1]
    block = df.loc[timestep : timestep + horizon - 1]

    if len(block) < horizon:
        last = block.iloc[[-1]] if len(block) > 0 else pd.DataFrame(
            np.zeros((1, feat_n)), columns=df.columns
        )
        pad = pd.concat([last] * (horizon - len(block)), ignore_index=True)
        block = pd.concat([block.reset_index(drop=True), pad], ignore_index=True)

    return block.values.reshape(-1).astype(np.float32)


def get_forecast_extra_dim(forecast_csv: str, horizon: int, index_col: str = "timestep") -> int:
    """
    Devuelve cuántas dimensiones añade el forecast al vector de observación.
    Útil para definir observation_space: obs_size_base + get_forecast_extra_dim(...).
    """
    if not os.path.isfile(forecast_csv):
        return horizon * 6
    df = pd.read_csv(forecast_csv, index_col=index_col, nrows=1)
    return horizon * df.shape[1]
