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

df['heat_pump_kW'] = df['heat_pump_power'] / 1000.0
df['datetime'] = pd.NaT

months_days = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
base_date = pd.Timestamp('2025-01-01')
day_offset = []
for _, row in df.iterrows():
    m = int(row['month'])
    dd = int(row['day_of_month'])
    h = int(row['hour'])
    total_days = sum(months_days[i] for i in range(1, m)) + dd - 1
    day_offset.append(total_days)

df['datetime'] = [base_date + pd.Timedelta(days=d, hours=int(df.iloc[i]['hour']))
                  for i, d in enumerate(day_offset)]

d = df[(df['month'] == 2) & (df['day_of_month'] >= 7) & (df['day_of_month'] <= 9)].copy()

def pmv_from_T_RH(T, RH):
    return -7.4928 + 0.2882 * T - 0.0020 * RH + 0.0004 * T * RH

def T_from_pmv_RH(pmv_target, RH):
    return (pmv_target + 7.4928 + 0.0020 * RH) / (0.2882 + 0.0004 * RH)

zones = {
    'north_perimeter_air_temperature': ('Norte', '#4CAF50', 'north_perimeter_air_humidity'),
    'south_perimeter_air_temperature': ('Sur', '#FF9800', 'south_perimeter_air_humidity'),
    'east_perimeter_air_temperature':  ('Este', '#9C27B0', 'east_perimeter_air_humidity'),
    'west_perimeter_air_temperature':  ('Oeste', '#00BCD4', 'west_perimeter_air_humidity'),
}

for tcol, (label, color, hcol) in zones.items():
    d[f'pmv_{label}'] = pmv_from_T_RH(d[tcol].values, d[hcol].values)
    d[f'T_low_{label}'] = T_from_pmv_RH(-0.5, d[hcol].values)
    d[f'T_high_{label}'] = T_from_pmv_RH(0.5, d[hcol].values)

fig, axes = plt.subplots(5, 1, figsize=(20, 22), sharex=True,
                          gridspec_kw={'height_ratios': [1.8, 1.0, 2.5, 2.0, 1.8]})

# --- Panel 1: Consumo bomba ---
ax1 = axes[0]
ax1.plot(d['datetime'], d['heat_pump_kW'], color='#2196F3', linewidth=1.2)
ax1.fill_between(d['datetime'], 0, d['heat_pump_kW'], color='#2196F3', alpha=0.2)
ax1.set_ylabel('Potencia (kW)', fontsize=11)
ax1.set_title('3 Días Más Calurosos del Año: 7-9 de Febrero (T ext media 25.2°C)\n(SAC λ_T=25, Evaluación ep. 20)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)
total_kwh = d['heat_pump_kW'].sum()
ax1.text(0.5, 0.92, f"Total: {total_kwh:,.0f} kWh | Máx: {d['heat_pump_kW'].max():.1f} kW | Media: {d['heat_pump_kW'].mean():.1f} kW",
         transform=ax1.transAxes, fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# --- Panel 2: Setpoint bomba ---
ax2 = axes[1]
ax2.step(d['datetime'], d['bomba'], where='post', color='#E91E63', linewidth=1.4)
ax2.fill_between(d['datetime'], 0, d['bomba'], step='post', color='#E91E63', alpha=0.1)
ax2.set_ylabel('Setpoint (°C)', fontsize=11)
ax2.set_title('Setpoint de Temperatura de la Bomba', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# --- Panel 3: Temperaturas + banda PMV ---
ax3 = axes[2]

rh_avg = d[['east_perimeter_air_humidity', 'west_perimeter_air_humidity',
            'north_perimeter_air_humidity', 'south_perimeter_air_humidity']].mean(axis=1)
T_lo_avg = T_from_pmv_RH(-0.5, rh_avg)
T_hi_avg = T_from_pmv_RH(0.5, rh_avg)

for tcol, (label, color, hcol) in zones.items():
    T_lo = d[f'T_low_{label}'].values
    T_hi = d[f'T_high_{label}'].values
    ax3.fill_between(d['datetime'], T_lo, T_hi, color=color, alpha=0.08)

for tcol, (label, color, hcol) in zones.items():
    T_lo = d[f'T_low_{label}'].values
    T_hi = d[f'T_high_{label}'].values
    ax3.plot(d['datetime'], T_lo, color=color, linewidth=0.5, linestyle=':', alpha=0.5)
    ax3.plot(d['datetime'], T_hi, color=color, linewidth=0.5, linestyle=':', alpha=0.5)

for tcol, (label, color, hcol) in zones.items():
    ax3.plot(d['datetime'], d[tcol], color=color, linewidth=1.3, label=label, alpha=0.9)

ax3.plot(d['datetime'], d['outdoor_temperature'], color='#F44336', linewidth=1.8,
         linestyle='--', label='Exterior', alpha=0.8)

ax3.fill_between(d['datetime'], T_lo_avg, T_hi_avg, color='green', alpha=0.12,
                 label=f'Confort PMV [-0.5, 0.5]\n(~{T_lo_avg.mean():.1f}-{T_hi_avg.mean():.1f}°C)')

ax3.set_ylabel('Temperatura (°C)', fontsize=11)
ax3.set_title('Temperaturas por Zona con Rango de Confort PMV [-0.5, +0.5]', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')

stats3 = (f"T media — N:{d['north_perimeter_air_temperature'].mean():.1f}°C  "
          f"S:{d['south_perimeter_air_temperature'].mean():.1f}°C  "
          f"E:{d['east_perimeter_air_temperature'].mean():.1f}°C  "
          f"O:{d['west_perimeter_air_temperature'].mean():.1f}°C  "
          f"Ext:{d['outdoor_temperature'].mean():.1f}°C")
ax3.text(0.5, 0.05, stats3, transform=ax3.transAxes, fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# --- Panel 4: PMV ---
ax4 = axes[3]
ax4.axhspan(-0.5, 0.5, color='green', alpha=0.15, label='Confort PMV [-0.5, +0.5]')
ax4.axhline(y=-0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
ax4.axhline(y=0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)

for tcol, (label, color, hcol) in zones.items():
    ax4.plot(d['datetime'], d[f'pmv_{label}'], color=color, linewidth=1.2, label=label, alpha=0.9)

ax4.set_ylabel('PMV', fontsize=11)
ax4.set_title('Índice PMV por Zona (confort térmico)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9, ncol=4, framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')

pct_parts = []
for tcol, (label, color, hcol) in zones.items():
    pmv_vals = d[f'pmv_{label}']
    pct = 100 * ((pmv_vals >= -0.5) & (pmv_vals <= 0.5)).sum() / len(pmv_vals)
    pct_parts.append(f"{label}: {pct:.0f}%")
ax4.text(0.5, 0.05, "% horas en confort PMV → " + "  |  ".join(pct_parts),
         transform=ax4.transAxes, fontsize=10, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew', edgecolor='green', alpha=0.9),
         fontweight='bold')

# --- Panel 5: Electroválvulas ---
ax5 = axes[4]
valves = {
    'electrovalve_north': ('Norte', '#4CAF50'),
    'electrovalve_south': ('Sur', '#FF9800'),
    'electrovalve_east':  ('Este', '#9C27B0'),
    'electrovalve_west':  ('Oeste', '#00BCD4'),
}
offsets_v = {'electrovalve_north': 3, 'electrovalve_south': 2, 'electrovalve_east': 1, 'electrovalve_west': 0}

for col, (label, color) in valves.items():
    off = offsets_v[col]
    y_vals = d[col] * 0.7 + off
    y_base = np.full(len(d), off)
    ax5.fill_between(d['datetime'], y_base, y_vals, step='post', color=color, alpha=0.6)
    ax5.step(d['datetime'], y_vals, where='post', color=color, linewidth=0.5, alpha=0.8)
    pct = 100 * d[col].mean()
    ax5.text(d['datetime'].iloc[-1] + pd.Timedelta(hours=1), off + 0.35,
             f'{pct:.0f}%', fontsize=10, ha='left', va='center', color=color, fontweight='bold')

ax5.set_ylabel('Electroválvulas', fontsize=11)
ax5.set_title('Estado de Electroválvulas por Zona (ON/OFF)', fontsize=12, fontweight='bold')
ax5.set_yticks([0.35, 1.35, 2.35, 3.35])
ax5.set_yticklabels(['Oeste', 'Este', 'Sur', 'Norte'])
ax5.set_ylim(-0.2, 4.2)
ax5.grid(True, alpha=0.2, linestyle='--', axis='x')
ax5.set_xlabel('Fecha', fontsize=12)

ax5.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d Feb %Hh'))
ax5.xaxis.set_minor_locator(mdates.HourLocator())
plt.xticks(rotation=45, ha='right')

for ax in axes:
    ax.set_xlim(d['datetime'].min(), d['datetime'].max())

plt.tight_layout()
plt.savefig('/workspaces/sinergym/bomba_hottest_3days.png', dpi=150, bbox_inches='tight')
print("Gráfico guardado en /workspaces/sinergym/bomba_hottest_3days.png")

print(f"\n=== Estadísticas 7-9 Feb (3 días más calurosos) ===")
print(f"Bomba: {total_kwh:,.0f} kWh | Máx: {d['heat_pump_kW'].max():.1f} kW")
print(f"T exterior: media {d['outdoor_temperature'].mean():.1f}°C, máx {d['outdoor_temperature'].max():.1f}°C")
print(f"\nConfort PMV [-0.5, 0.5]:")
for tcol, (label, color, hcol) in zones.items():
    pmv_v = d[f'pmv_{label}']
    pct = 100 * ((pmv_v >= -0.5) & (pmv_v <= 0.5)).sum() / len(pmv_v)
    print(f"  {label}: {pct:.1f}% en confort (PMV medio: {pmv_v.mean():.2f})")
