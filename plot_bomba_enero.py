import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

csv_path = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor/observations.csv"

df = pd.read_csv(csv_path)

df_jan = df[df['month'] == 1.0].copy()

df_jan['heat_pump_kW'] = df_jan['heat_pump_power'] / 1000.0

df_jan['datetime'] = pd.to_datetime(
    '2025-01-' + df_jan['day_of_month'].astype(int).astype(str) + ' ' + df_jan['hour'].astype(int).astype(str) + ':00',
    format='%Y-%m-%d %H:%M'
)
df_jan = df_jan.sort_values('datetime')

fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(df_jan['datetime'], df_jan['heat_pump_kW'], color='#2196F3', linewidth=0.7, alpha=0.85)
ax.fill_between(df_jan['datetime'], 0, df_jan['heat_pump_kW'], color='#2196F3', alpha=0.15)

ax.set_xlabel('Fecha (Enero)', fontsize=12)
ax.set_ylabel('Potencia Bomba de Calor (kW)', fontsize=12)
ax.set_title('Consumo de la Bomba de Calor - Enero\n(SAC λ_T=25, Evaluación episodio 20)', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Ene'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
plt.xticks(rotation=45, ha='right')

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(df_jan['datetime'].min(), df_jan['datetime'].max())
ax.set_ylim(bottom=0)

total_kwh = df_jan['heat_pump_kW'].sum()
max_kw = df_jan['heat_pump_kW'].max()
mean_kw = df_jan['heat_pump_kW'].mean()
hours_on = (df_jan['heat_pump_kW'] > 0).sum()
total_hours = len(df_jan)

stats_text = (
    f"Energía total: {total_kwh:,.0f} kWh\n"
    f"Potencia máx: {max_kw:.1f} kW\n"
    f"Potencia media: {mean_kw:.1f} kW\n"
    f"Horas encendida: {hours_on}/{total_hours} ({100*hours_on/total_hours:.0f}%)"
)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('/workspaces/sinergym/bomba_enero.png', dpi=150, bbox_inches='tight')
print("Gráfico guardado en /workspaces/sinergym/bomba_enero.png")
print(f"\nEstadísticas de Enero:")
print(f"  Energía total: {total_kwh:,.0f} kWh")
print(f"  Potencia máxima: {max_kw:.1f} kW")
print(f"  Potencia media: {mean_kw:.1f} kW")
print(f"  Horas con bomba encendida: {hours_on} de {total_hours} ({100*hours_on/total_hours:.0f}%)")
