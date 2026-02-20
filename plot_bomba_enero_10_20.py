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

d = df_jan[(df_jan['day_of_month'] >= 10) & (df_jan['day_of_month'] <= 20)].copy()

fig, axes = plt.subplots(4, 1, figsize=(20, 18), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1.3, 2.5, 2]})

# --- Panel 1: Consumo bomba (kW) ---
ax1 = axes[0]
ax1.plot(d['datetime'], d['heat_pump_kW'], color='#2196F3', linewidth=1.0)
ax1.fill_between(d['datetime'], 0, d['heat_pump_kW'], color='#2196F3', alpha=0.2)
ax1.set_ylabel('Potencia (kW)', fontsize=12)
ax1.set_title('Consumo, Setpoint, Temperaturas y Electroválvulas — 10 al 20 de Enero\n(SAC λ_T=25, Evaluación episodio 20)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)
total_kwh = d['heat_pump_kW'].sum()
ax1.text(0.5, 0.93, f"Total: {total_kwh:,.0f} kWh | Máx: {d['heat_pump_kW'].max():.1f} kW | Media: {d['heat_pump_kW'].mean():.1f} kW",
         transform=ax1.transAxes, fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# --- Panel 2: Setpoint bomba (°C) ---
ax2 = axes[1]
ax2.step(d['datetime'], d['bomba'], where='post', color='#E91E63', linewidth=1.2)
ax2.fill_between(d['datetime'], 0, d['bomba'], step='post', color='#E91E63', alpha=0.1)
ax2.set_ylabel('Setpoint (°C)', fontsize=12)
ax2.set_title('Setpoint de Temperatura de la Bomba', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.text(0.5, 0.93, f"Mín: {d['bomba'].min():.0f}°C | Máx: {d['bomba'].max():.0f}°C | Media: {d['bomba'].mean():.1f}°C",
         transform=ax2.transAxes, fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', edgecolor='gray', alpha=0.9))

# --- Panel 3: Temperaturas por zona + exterior ---
ax3 = axes[2]
zones = {
    'north_perimeter_air_temperature': ('Norte', '#4CAF50'),
    'south_perimeter_air_temperature': ('Sur', '#FF9800'),
    'east_perimeter_air_temperature':  ('Este', '#9C27B0'),
    'west_perimeter_air_temperature':  ('Oeste', '#00BCD4'),
}
for col, (label, color) in zones.items():
    ax3.plot(d['datetime'], d[col], color=color, linewidth=1.0, label=label, alpha=0.9)

ax3.plot(d['datetime'], d['outdoor_temperature'], color='#F44336', linewidth=1.5,
         linestyle='--', label='Exterior', alpha=0.8)

ax3.axhspan(20, 26, color='green', alpha=0.07, label='Rango confort (20-26°C)')
ax3.axhline(y=20, color='green', linewidth=0.5, linestyle=':', alpha=0.5)
ax3.axhline(y=26, color='green', linewidth=0.5, linestyle=':', alpha=0.5)

ax3.set_ylabel('Temperatura (°C)', fontsize=12)
ax3.set_title('Temperaturas por Zona y Exterior', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')

for col, (label, _) in zones.items():
    t_mean = d[col].mean()
    t_min = d[col].min()
    t_max = d[col].max()
ext_mean = d['outdoor_temperature'].mean()
stats3 = (f"T media — N:{d['north_perimeter_air_temperature'].mean():.1f}°C  "
          f"S:{d['south_perimeter_air_temperature'].mean():.1f}°C  "
          f"E:{d['east_perimeter_air_temperature'].mean():.1f}°C  "
          f"O:{d['west_perimeter_air_temperature'].mean():.1f}°C  "
          f"Ext:{ext_mean:.1f}°C")
ax3.text(0.5, 0.07, stats3, transform=ax3.transAxes, fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# --- Panel 4: Electroválvulas ---
ax4 = axes[3]
valves = {
    'electrovalve_north': ('Norte', '#4CAF50'),
    'electrovalve_south': ('Sur', '#FF9800'),
    'electrovalve_east':  ('Este', '#9C27B0'),
    'electrovalve_west':  ('Oeste', '#00BCD4'),
}
offsets = {'electrovalve_north': 3, 'electrovalve_south': 2, 'electrovalve_east': 1, 'electrovalve_west': 0}

for col, (label, color) in valves.items():
    off = offsets[col]
    y_vals = d[col] * 0.7 + off
    y_base = np.full(len(d), off)
    ax4.fill_between(d['datetime'], y_base, y_vals, step='post', color=color, alpha=0.6, label=label)
    ax4.step(d['datetime'], y_vals, where='post', color=color, linewidth=0.5, alpha=0.8)
    pct = 100 * d[col].mean()
    ax4.text(d['datetime'].iloc[-1] + pd.Timedelta(hours=3), off + 0.35,
             f'{pct:.0f}%', fontsize=10, ha='left', va='center', color=color, fontweight='bold')

ax4.set_ylabel('Electroválvulas', fontsize=12)
ax4.set_title('Estado de Electroválvulas por Zona (ON/OFF)', fontsize=12, fontweight='bold')
ax4.set_yticks([0.35, 1.35, 2.35, 3.35])
ax4.set_yticklabels(['Oeste', 'Este', 'Sur', 'Norte'])
ax4.set_ylim(-0.2, 4.2)
ax4.grid(True, alpha=0.2, linestyle='--', axis='x')
ax4.set_xlabel('Fecha (Enero)', fontsize=12)

ax4.xaxis.set_major_locator(mdates.DayLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d Ene'))
ax4.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
plt.xticks(rotation=45, ha='right')

for ax in axes:
    ax.set_xlim(d['datetime'].min(), d['datetime'].max())

plt.tight_layout()
plt.savefig('/workspaces/sinergym/bomba_enero_10_20.png', dpi=150, bbox_inches='tight')
print("Gráfico guardado en /workspaces/sinergym/bomba_enero_10_20.png")

print(f"\n=== Estadísticas 10-20 Enero ===")
print(f"\nBomba: {total_kwh:,.0f} kWh | Máx: {d['heat_pump_kW'].max():.1f} kW")
print(f"Setpoint: {d['bomba'].min():.0f}-{d['bomba'].max():.0f}°C (media {d['bomba'].mean():.1f}°C)")
print(f"\nTemperaturas medias:")
print(f"  Norte: {d['north_perimeter_air_temperature'].mean():.1f}°C")
print(f"  Sur:   {d['south_perimeter_air_temperature'].mean():.1f}°C")
print(f"  Este:  {d['east_perimeter_air_temperature'].mean():.1f}°C")
print(f"  Oeste: {d['west_perimeter_air_temperature'].mean():.1f}°C")
print(f"  Exterior: {d['outdoor_temperature'].mean():.1f}°C")
print(f"\nElectroválvulas (% ON):")
for col, (label, _) in valves.items():
    print(f"  {label}: {100*d[col].mean():.1f}%")
