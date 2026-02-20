#!/usr/bin/env python3
"""
Generador de gráficas de resultados de entrenamiento Sinergym.
Genera 6 PNGs en la misma carpeta del CSV de observaciones.

Uso:
  python generar_graficas.py <ruta_al_observations.csv>

Ejemplo:
  python generar_graficas.py Eplus-SAC-training-.../episode-5/monitor/observations.csv
"""

import sys
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Backend sin ventana
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DPI = 150
YEAR = 2024  # Año de referencia para construir timestamps

# Tarifas eléctricas ($/kWh)
TARIFA_PUNTA = 14.493
TARIFA_FUERA_PUNTA = 4.556

# Colores
COLOR_EAST = '#E74C3C'      # Rojo
COLOR_WEST = '#3498DB'      # Azul
COLOR_OUTDOOR = '#2ECC71'   # Verde
COLOR_NORTH = '#9B59B6'     # Púrpura
COLOR_SOUTH = '#F39C12'     # Naranja
COLOR_POWER = '#E67E22'     # Naranja oscuro
COLOR_ENERGY = '#2980B9'    # Azul medio
COLOR_PUNTA = '#E74C3C'     # Rojo
COLOR_FUERA = '#27AE60'     # Verde
COLOR_ACUM = '#8E44AD'      # Púrpura


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Carga el CSV y construye columna datetime."""
    df = pd.read_csv(csv_path)

    # Construir timestamp: E+ marca el FINAL del intervalo
    # hour=0 → corresponde a 00:00 (medianoche)
    # Ajustar: hora 0 de un día es realmente la hora 24 del día anterior en E+
    timestamps = []
    for _, row in df.iterrows():
        m = int(row['month'])
        d = int(row['day_of_month'])
        h = int(row['hour'])
        try:
            ts = pd.Timestamp(year=YEAR, month=m, day=d, hour=h)
        except ValueError:
            # Fecha inválida (ej. 31 de febrero) → usar última válida
            ts = timestamps[-1] + pd.Timedelta(hours=1) if timestamps else pd.Timestamp(year=YEAR, month=1, day=1)
        timestamps.append(ts)

    df['timestamp'] = timestamps
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Lunes, 6=Domingo
    return df


def es_hora_punta(row) -> bool:
    """Determina si un paso es hora punta.
    Hora punta: Lunes a Viernes, intervalo 17-21h.
    El timestamp de E+ marca el FINAL del intervalo,
    entonces hora 18,19,20,21 en día hábil = punta.
    """
    h = int(row['hour'])
    dow = row['day_of_week']
    return (dow < 5) and (h >= 18) and (h <= 21)


def calcular_pmv(tdb, rh):
    """Calcula PMV simplificado."""
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh


def guardar(fig, carpeta, nombre):
    """Guarda la figura como PNG."""
    path = os.path.join(carpeta, nombre)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Guardado: {path}")


# ============================================================================
# GRÁFICA 1: Temperaturas anuales
# ============================================================================
def grafica_temperaturas_anual(df, carpeta):
    print("\n📊 Gráfica 1: Temperaturas anuales")
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(df['timestamp'], df['east_perimeter_air_temperature'],
            color=COLOR_EAST, linewidth=0.4, alpha=0.8, label='Zona East')
    ax.plot(df['timestamp'], df['west_perimeter_air_temperature'],
            color=COLOR_WEST, linewidth=0.4, alpha=0.8, label='Zona West')
    ax.plot(df['timestamp'], df['outdoor_temperature'],
            color=COLOR_OUTDOOR, linewidth=0.4, alpha=0.6, label='Exterior')

    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Temperatura (°C)', fontsize=12)
    ax.set_title('Temperatura anual — Zonas East, West y Exterior', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    guardar(fig, carpeta, '1_temperaturas_anuales.png')


# ============================================================================
# GRÁFICA 2: Temperaturas día 1 de Julio
# ============================================================================
def grafica_temperaturas_dia(df, carpeta, mes, dia, nombre_dia, num_grafica):
    print(f"\n📊 Gráfica {num_grafica}: Temperaturas {nombre_dia}")
    mask = (df['month'] == mes) & (df['day_of_month'] == dia)
    dd = df[mask].copy()

    if dd.empty:
        print(f"  ⚠️  No hay datos para {nombre_dia} (mes={mes}, día={dia})")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dd['hour'], dd['east_perimeter_air_temperature'],
            color=COLOR_EAST, marker='o', markersize=5, linewidth=1.5, label='Zona East')
    ax.plot(dd['hour'], dd['west_perimeter_air_temperature'],
            color=COLOR_WEST, marker='s', markersize=5, linewidth=1.5, label='Zona West')
    ax.plot(dd['hour'], dd['outdoor_temperature'],
            color=COLOR_OUTDOOR, marker='^', markersize=5, linewidth=1.5, label='Exterior')

    ax.set_xlabel('Hora del día', fontsize=12)
    ax.set_ylabel('Temperatura (°C)', fontsize=12)
    ax.set_title(f'Temperaturas hora a hora — {nombre_dia}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 25))
    ax.set_xlim(-0.5, 24.5)

    guardar(fig, carpeta, f'{num_grafica}_temperaturas_{nombre_dia.replace(" ", "_").replace("/", "-")}.png')


# ============================================================================
# GRÁFICA 4: Consumo eléctrico Heat Pump
# ============================================================================
def grafica_consumo_electrico(df, carpeta):
    print("\n📊 Gráfica 4: Consumo eléctrico Heat Pump")

    # Potencia instantánea en kW
    potencia_kW = df['heat_pump_power'] / 1000.0

    # Energía por paso en kWh (meter en Joules → kWh)
    energia_kWh_paso = df['total_electricity_HVAC'] / 3.6e6
    energia_acum_kWh = energia_kWh_paso.cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # --- Subplot superior: Potencia instantánea ---
    ax1.plot(df['timestamp'], potencia_kW,
             color=COLOR_POWER, linewidth=0.4, alpha=0.8)
    ax1.fill_between(df['timestamp'], 0, potencia_kW,
                     color=COLOR_POWER, alpha=0.15)
    ax1.set_ylabel('Potencia (kW)', fontsize=12)
    ax1.set_title('Consumo eléctrico de la Heat Pump', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    max_pot = potencia_kW.max()
    mean_pot = potencia_kW[potencia_kW > 0].mean() if (potencia_kW > 0).any() else 0
    ax1.axhline(y=mean_pot, color='gray', linestyle='--', alpha=0.5,
                label=f'Promedio (activa): {mean_pot:.1f} kW')
    ax1.legend(fontsize=10, loc='upper right')

    textstr = f'Máx: {max_pot:.1f} kW\nProm (activa): {mean_pot:.1f} kW'
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Subplot inferior: Energía acumulada ---
    ax2.plot(df['timestamp'], energia_acum_kWh,
             color=COLOR_ENERGY, linewidth=1.5)
    ax2.fill_between(df['timestamp'], 0, energia_acum_kWh,
                     color=COLOR_ENERGY, alpha=0.15)
    ax2.set_xlabel('Mes', fontsize=12)
    ax2.set_ylabel('Energía acumulada (kWh)', fontsize=12)
    ax2.set_title('Energía acumulada', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    total_kWh = energia_acum_kWh.iloc[-1]
    ax2.text(0.02, 0.95, f'Total anual: {total_kWh:,.0f} kWh',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.autofmt_xdate()
    plt.tight_layout()
    guardar(fig, carpeta, '4_consumo_electrico_heat_pump.png')


# ============================================================================
# GRÁFICA 5: Costo eléctrico Heat Pump
# ============================================================================
def grafica_costo_electrico(df, carpeta):
    print("\n📊 Gráfica 5: Costo eléctrico Heat Pump")

    # Energía por paso en kWh
    df_cost = df.copy()
    df_cost['energia_kWh'] = df_cost['total_electricity_HVAC'] / 3.6e6

    # Clasificar punta / fuera de punta
    df_cost['es_punta'] = df_cost.apply(es_hora_punta, axis=1)

    # Tarifa y costo
    df_cost['tarifa'] = np.where(df_cost['es_punta'], TARIFA_PUNTA, TARIFA_FUERA_PUNTA)
    df_cost['costo'] = df_cost['energia_kWh'] * df_cost['tarifa']
    df_cost['costo_acum'] = df_cost['costo'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    # --- Subplot superior: Costo horario con franjas ---
    mask_punta = df_cost['es_punta']
    mask_fuera = ~df_cost['es_punta']

    ax1.bar(df_cost.loc[mask_fuera, 'timestamp'], df_cost.loc[mask_fuera, 'costo'],
            width=0.04, color=COLOR_FUERA, alpha=0.6, label=f'Fuera de punta (${TARIFA_FUERA_PUNTA}/kWh)')
    ax1.bar(df_cost.loc[mask_punta, 'timestamp'], df_cost.loc[mask_punta, 'costo'],
            width=0.04, color=COLOR_PUNTA, alpha=0.6, label=f'Hora punta (${TARIFA_PUNTA}/kWh)')

    ax1.set_ylabel('Costo horario ($)', fontsize=12)
    ax1.set_title('Costo eléctrico de la Heat Pump por franja tarifaria', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # --- Subplot inferior: Costo acumulado + cuadro resumen ---
    ax2.plot(df_cost['timestamp'], df_cost['costo_acum'],
             color=COLOR_ACUM, linewidth=2)
    ax2.fill_between(df_cost['timestamp'], 0, df_cost['costo_acum'],
                     color=COLOR_ACUM, alpha=0.1)
    ax2.set_xlabel('Mes', fontsize=12)
    ax2.set_ylabel('Costo acumulado ($)', fontsize=12)
    ax2.set_title('Costo acumulado anual', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    # Calcular resumen
    kWh_punta = df_cost.loc[mask_punta, 'energia_kWh'].sum()
    kWh_fuera = df_cost.loc[mask_fuera, 'energia_kWh'].sum()
    costo_punta = df_cost.loc[mask_punta, 'costo'].sum()
    costo_fuera = df_cost.loc[mask_fuera, 'costo'].sum()
    kWh_total = kWh_punta + kWh_fuera
    costo_total = costo_punta + costo_fuera

    resumen = (
        f'{"Franja":<16} {"kWh":>10} {"Costo ($)":>12}\n'
        f'{"─"*40}\n'
        f'{"Hora punta":<16} {kWh_punta:>10,.1f} {costo_punta:>12,.1f}\n'
        f'{"Fuera de punta":<16} {kWh_fuera:>10,.1f} {costo_fuera:>12,.1f}\n'
        f'{"─"*40}\n'
        f'{"TOTAL":<16} {kWh_total:>10,.1f} {costo_total:>12,.1f}'
    )

    ax2.text(0.02, 0.95, resumen, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    fig.autofmt_xdate()
    plt.tight_layout()
    guardar(fig, carpeta, '5_costo_electrico_heat_pump.png')


# ============================================================================
# GRÁFICA 6: PMV anual
# ============================================================================
def grafica_pmv_anual(df, carpeta):
    print("\n📊 Gráfica 6: PMV anual")

    pmv_east = calcular_pmv(df['east_perimeter_air_temperature'], df['east_perimeter_air_humidity'])
    pmv_west = calcular_pmv(df['west_perimeter_air_temperature'], df['west_perimeter_air_humidity'])
    pmv_outdoor = calcular_pmv(df['outdoor_temperature'], df['outdoor_humidity'])

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df['timestamp'], pmv_east,
            color=COLOR_EAST, linewidth=0.4, alpha=0.8, label='Zona East')
    ax.plot(df['timestamp'], pmv_west,
            color=COLOR_WEST, linewidth=0.4, alpha=0.8, label='Zona West')
    ax.plot(df['timestamp'], pmv_outdoor,
            color=COLOR_OUTDOOR, linewidth=0.4, alpha=0.6, label='Exterior')

    # Bandas de confort
    ax.axhspan(-0.5, 0.5, alpha=0.15, color='green', label='Confortable (±0.5)')
    ax.axhspan(-1.0, -0.5, alpha=0.08, color='yellow')
    ax.axhspan(0.5, 1.0, alpha=0.08, color='yellow', label='Ligeramente incómodo (±1.0)')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axhline(y=-0.5, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=-1.0, color='orange', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='orange', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('PMV', fontsize=12)
    ax.set_title('Índice PMV anual — Zonas East, West y Exterior', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Calcular porcentaje en confort
    pct_east = ((pmv_east.abs() <= 0.5).sum() / len(pmv_east)) * 100
    pct_west = ((pmv_west.abs() <= 0.5).sum() / len(pmv_west)) * 100
    textstr = (f'% tiempo en confort (|PMV|≤0.5):\n'
               f'  East:  {pct_east:.1f}%\n'
               f'  West:  {pct_west:.1f}%')
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    fig.autofmt_xdate()
    guardar(fig, carpeta, '6_pmv_anual.png')


# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Uso: python generar_graficas.py <ruta_al_observations.csv>")
        print("Ejemplo: python generar_graficas.py episode-5/monitor/observations.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró: {csv_path}")
        sys.exit(1)

    carpeta = os.path.dirname(os.path.abspath(csv_path))
    print(f"📂 CSV: {csv_path}")
    print(f"📁 PNGs se guardarán en: {carpeta}")

    # Cargar datos
    print("\n⏳ Cargando datos...")
    df = load_data(csv_path)
    print(f"   {len(df)} filas cargadas")
    print(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Columnas: {list(df.columns)}")

    # Generar gráficas
    grafica_temperaturas_anual(df, carpeta)
    grafica_temperaturas_dia(df, carpeta, mes=7, dia=1, nombre_dia='1 de Julio', num_grafica=2)
    grafica_temperaturas_dia(df, carpeta, mes=1, dia=10, nombre_dia='10 de Enero', num_grafica=3)
    grafica_consumo_electrico(df, carpeta)
    grafica_costo_electrico(df, carpeta)
    grafica_pmv_anual(df, carpeta)

    print(f"\n✅ Todas las gráficas generadas en: {carpeta}")


if __name__ == '__main__':
    main()
