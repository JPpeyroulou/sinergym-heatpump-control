import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

base = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"

obs = pd.read_csv(f"{base}/observations.csv")
act = pd.read_csv(f"{base}/simulated_actions.csv")

min_len = min(len(obs), len(act))
obs = obs.iloc[:min_len].copy()
act = act.iloc[:min_len].copy()

df = pd.concat([obs.reset_index(drop=True), act.reset_index(drop=True)], axis=1)

df_jan = df[df['month'] == 1.0].copy()
df_jan['heat_pump_kW'] = df_jan['heat_pump_power'] / 1000.0
df_jan['datetime'] = pd.to_datetime(
    '2025-01-' + df_jan['day_of_month'].astype(int).astype(str) + ' ' + df_jan['hour'].astype(int).astype(str) + ':00',
    format='%Y-%m-%d %H:%M'
)
df_jan = df_jan.sort_values('datetime').reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1.5, 2]})

# --- Panel 1: Consumo bomba (kW) ---
ax1 = axes[0]
ax1.plot(df_jan['datetime'], df_jan['heat_pump_kW'], color='#2196F3', linewidth=0.8, alpha=0.9)
ax1.fill_between(df_jan['datetime'], 0, df_jan['heat_pump_kW'], color='#2196F3', alpha=0.15)
ax1.set_ylabel('Potencia (kW)', fontsize=12)
ax1.set_title('Consumo de la Bomba de Calor - Enero\n(SAC λ_T=25, Evaluación episodio 20)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

total_kwh = df_jan['heat_pump_kW'].sum()
max_kw = df_jan['heat_pump_kW'].max()
mean_kw = df_jan['heat_pump_kW'].mean()
stats1 = f"Total: {total_kwh:,.0f} kWh | Máx: {max_kw:.1f} kW | Media: {mean_kw:.1f} kW"
ax1.text(0.5, 0.95, stats1, transform=ax1.transAxes, fontsize=10,
         ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# --- Panel 2: Setpoint bomba (°C) ---
ax2 = axes[1]
ax2.step(df_jan['datetime'], df_jan['bomba'], where='post', color='#E91E63', linewidth=1.0)
ax2.fill_between(df_jan['datetime'], 0, df_jan['bomba'], step='post', color='#E91E63', alpha=0.1)
ax2.set_ylabel('Setpoint Bomba (°C)', fontsize=12)
ax2.set_title('Setpoint de Temperatura de la Bomba de Calor', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

sp_min = df_jan['bomba'].min()
sp_max = df_jan['bomba'].max()
sp_mean = df_jan['bomba'].mean()
stats2 = f"Mín: {sp_min:.0f}°C | Máx: {sp_max:.0f}°C | Media: {sp_mean:.1f}°C"
ax2.text(0.5, 0.95, stats2, transform=ax2.transAxes, fontsize=10,
         ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', edgecolor='gray', alpha=0.9))

# --- Panel 3: Electroválvulas ---
ax3 = axes[2]
valves = {
    'electrovalve_north': ('Norte', '#4CAF50'),
    'electrovalve_south': ('Sur', '#FF9800'),
    'electrovalve_east':  ('Este', '#9C27B0'),
    'electrovalve_west':  ('Oeste', '#00BCD4'),
}

offsets = {'electrovalve_north': 3, 'electrovalve_south': 2, 'electrovalve_east': 1, 'electrovalve_west': 0}

for col, (label, color) in valves.items():
    offset = offsets[col]
    y_vals = df_jan[col] * 0.7 + offset
    y_base = np.full(len(df_jan), offset)
    ax3.fill_between(df_jan['datetime'], y_base, y_vals, step='post',
                     color=color, alpha=0.6, label=label)
    ax3.step(df_jan['datetime'], y_vals, where='post', color=color, linewidth=0.5, alpha=0.8)
    ax3.text(df_jan['datetime'].iloc[0] - pd.Timedelta(hours=12), offset + 0.35, label,
             fontsize=10, fontweight='bold', ha='right', va='center', color=color)

    pct_on = 100 * df_jan[col].mean()
    ax3.text(df_jan['datetime'].iloc[-1] + pd.Timedelta(hours=6), offset + 0.35,
             f'{pct_on:.0f}%', fontsize=9, ha='left', va='center', color=color, fontweight='bold')

ax3.set_ylabel('Estado Electroválvulas', fontsize=12)
ax3.set_title('Estado de Electroválvulas por Zona (ON/OFF)', fontsize=12, fontweight='bold')
ax3.set_yticks([0.35, 1.35, 2.35, 3.35])
ax3.set_yticklabels(['Oeste', 'Este', 'Sur', 'Norte'])
ax3.set_ylim(-0.2, 4.2)
ax3.grid(True, alpha=0.2, linestyle='--', axis='x')

ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d Ene'))
ax3.xaxis.set_minor_locator(mdates.DayLocator())
plt.xticks(rotation=45, ha='right')
ax3.set_xlabel('Fecha (Enero)', fontsize=12)

for ax in axes:
    ax.set_xlim(df_jan['datetime'].min(), df_jan['datetime'].max())

plt.tight_layout()
plt.savefig('/workspaces/sinergym/bomba_enero_completo.png', dpi=150, bbox_inches='tight')
print("Gráfico guardado en /workspaces/sinergym/bomba_enero_completo.png")

print(f"\n=== Estadísticas Enero ===")
print(f"\nBomba de calor:")
print(f"  Energía total: {total_kwh:,.0f} kWh")
print(f"  Potencia máx: {max_kw:.1f} kW | media: {mean_kw:.1f} kW")
print(f"\nSetpoint bomba:")
print(f"  Rango: {sp_min:.0f}°C - {sp_max:.0f}°C | Media: {sp_mean:.1f}°C")
print(f"\nElectroválvulas (% tiempo ON):")
for col, (label, _) in valves.items():
    pct = 100 * df_jan[col].mean()
    print(f"  {label}: {pct:.1f}%")
