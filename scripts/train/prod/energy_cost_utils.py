"""
Utilidad para obtener el precio de la energía (energy_cost) cuando ejecutás
el modelo por fuera del entrenamiento. El modelo entrenado con EnergyCostWrapper
espera una observación que termina con ese valor.
"""
import os
from typing import Union

import pandas as pd
import numpy as np


def get_energy_cost_value(
    month: int,
    day: int,
    hour: int,
    energy_cost_csv: str,
    sep: str = ";",
) -> float:
    """
    Devuelve el valor de precio (columna `value`) para la hora dada desde el CSV.

    Mismo criterio que EnergyCostWrapper: el CSV debe tener columnas
    datetime y value; se derivan Month, Day, Hour y se busca la fila que coincida.

    Args:
        month: 1-12
        day: 1-31
        hour: 0-23
        energy_cost_csv: Ruta al CSV (ej. PVPC_active_energy_billing_...csv).
        sep: Separador del CSV (por defecto ";").

    Returns:
        Valor de la columna 'value' para esa hora, o 0.0 si no hay fila o falla.
    """
    if not os.path.isfile(energy_cost_csv):
        return 0.0
    try:
        df = pd.read_csv(energy_cost_csv, sep=sep)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] += pd.DateOffset(hours=1)
        df["Month"] = df["datetime"].dt.month
        df["Day"] = df["datetime"].dt.day
        df["Hour"] = df["datetime"].dt.hour
        mask = (
            (df["Month"] == month)
            & (df["Day"] == day)
            & (df["Hour"] == hour)
        )
        rows = df.loc[mask, "value"]
        if len(rows) == 0:
            return 0.0
        return float(rows.iloc[0])
    except Exception:
        return 0.0


def load_energy_cost_table(energy_cost_csv: str, sep: str = ";") -> pd.DataFrame:
    """
    Carga el CSV y devuelve un DataFrame con columnas Month, Day, Hour, value.
    Útil si querés consultar muchas horas sin releer el archivo en cada paso.
    """
    if not os.path.isfile(energy_cost_csv):
        return pd.DataFrame(columns=["Month", "Day", "Hour", "value"])
    df = pd.read_csv(energy_cost_csv, sep=sep)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime"] += pd.DateOffset(hours=1)
    df["Month"] = df["datetime"].dt.month
    df["Day"] = df["datetime"].dt.day
    df["Hour"] = df["datetime"].dt.hour
    return df[["Month", "Day", "Hour", "value"]].copy()


def get_energy_cost_from_table(
    table: pd.DataFrame,
    month: int,
    day: int,
    hour: int,
) -> float:
    """Dado un DataFrame ya cargado con load_energy_cost_table, devuelve value para (month, day, hour)."""
    if table is None or len(table) == 0:
        return 0.0
    mask = (
        (table["Month"] == month)
        & (table["Day"] == day)
        & (table["Hour"] == hour)
    )
    rows = table.loc[mask, "value"]
    if len(rows) == 0:
        return 0.0
    return float(rows.iloc[0])
