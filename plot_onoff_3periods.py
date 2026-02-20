import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ===================== CARGAR DATOS ON/OFF =====================
onoff = pd.read_csv('/workspaces/sinergym/baseline_onoff/eplusout.csv')
onoff.columns = onoff.columns.str.strip()
dt_list = []
for dt_str in onoff['Date/Time']:
    dt_str = dt_str.strip()
    parts = dt_str.split()
    md = parts[0].split('/')
    month, day = int(md[0]), int(md[1])
    hour = int(parts[1].split(':')[0]) if len(parts) > 1 else 0
    if hour == 24: hour = 0
    try: dt_list.append(pd.Timestamp(2025, month, day, hour))
    except: dt_list.append(pd.NaT)
onoff['datetime'] = dt_list
onoff['heat_pump_kW'] = onoff['BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)'] / 1000.0
onoff['east_temp'] = onoff['EAST PERIMETER:Zone Air Temperature [C](Hourly)']
onoff['west_temp'] = onoff['WEST PERIMETER:Zone Air Temperature [C](Hourly)']
onoff['east_rh'] = onoff['EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)']
onoff['west_rh'] = onoff['WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)']
onoff['outdoor_temp'] = onoff['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']
onoff['month'] = onoff['datetime'].dt.month
onoff['day'] = onoff['datetime'].dt.day

# Heating rates para inferir electroválvulas
onoff['hr_east'] = onoff['LOZARADIANTE_ZONAEAST:Zone Radiant HVAC Heating Rate [W](Hourly)']
onoff['hr_west'] = onoff['LOZARADIANTE_ZONAWEST:Zone Radiant HVAC Heating Rate [W](Hourly)']
onoff['ev_east'] = (onoff['hr_east'] > 0).astype(int)
onoff['ev_west'] = (onoff['hr_west'] > 0).astype(int)

# Setpoint inferido desde la temperatura de salida de la bomba
onoff['setpoint'] = onoff['NODO_SALIDA_AIRE_HP:System Node Temperature [C](Hourly)']

def pmv(T, RH):
    return -7.4928 + 0.2882 * T - 0.0020 * RH + 0.0004 * T * RH
def T_from_pmv(target, RH):
    return (target + 7.4928 + 0.0020 * RH) / (0.2882 + 0.0004 * RH)
def pct_comfort(vals):
    return 100 * ((vals >= -0.5) & (vals <= 0.5)).sum() / len(vals)

onoff['pmv_east'] = pmv(onoff['east_temp'], onoff['east_rh'])
onoff['pmv_west'] = pmv(onoff['west_temp'], onoff['west_rh'])

# ===================== FUNCIÓN PLOT =====================
def plot_onoff(d, title_suffix, date_fmt, fname):
    fig, axes = plt.subplots(5, 1, figsize=(20, 22), sharex=True,
                              gridspec_kw={'height_ratios': [1.8, 1.0, 2.5, 2.2, 1.8]})

    # --- Panel 1: Consumo bomba ---
    ax1 = axes[0]
    ax1.plot(d['datetime'], d['heat_pump_kW'], color='#FF5722', linewidth=1.2)
    ax1.fill_between(d['datetime'], 0, d['heat_pump_kW'], color='#FF5722', alpha=0.2)
    ax1.set_ylabel('Potencia (kW)', fontsize=11)
    ax1.set_title(f'{title_suffix}\nBaseline ON/OFF', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    kwh = d['heat_pump_kW'].sum()
    ax1.text(0.5, 0.92,
             f"Total: {kwh:,.0f} kWh | Máx: {d['heat_pump_kW'].max():.1f} kW | Media: {d['heat_pump_kW'].mean():.1f} kW",
             transform=ax1.transAxes, fontsize=10, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # --- Panel 2: Setpoint / T salida bomba ---
    ax2 = axes[1]
    ax2.plot(d['datetime'], d['setpoint'], color='#E91E63', linewidth=1.2)
    ax2.fill_between(d['datetime'], 0, d['setpoint'], color='#E91E63', alpha=0.1)
    ax2.set_ylabel('T salida HP (°C)', fontsize=11)
    ax2.set_title('Temperatura de Salida de la Bomba de Calor', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # --- Panel 3: Temperaturas ---
    ax3 = axes[2]
    rh_avg = d[['east_rh', 'west_rh']].mean(axis=1)
    T_lo = T_from_pmv(-0.5, rh_avg); T_hi = T_from_pmv(0.5, rh_avg)
    ax3.fill_between(d['datetime'], T_lo, T_hi, color='green', alpha=0.12, label='Confort PMV [-0.5, 0.5]')
    ax3.plot(d['datetime'], d['east_temp'], color='#9C27B0', linewidth=1.5, label='Este', alpha=0.9)
    ax3.plot(d['datetime'], d['west_temp'], color='#00BCD4', linewidth=1.5, label='Oeste', alpha=0.9)
    ax3.plot(d['datetime'], d['outdoor_temp'], color='#F44336', linewidth=1.8, linestyle='--', label='Exterior', alpha=0.7)
    ax3.set_ylabel('Temperatura (°C)', fontsize=11)
    ax3.set_title('Temperaturas Zona Este y Oeste + Exterior', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10, ncol=4, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    stats = f"T media Este: {d['east_temp'].mean():.1f}°C  |  T media Oeste: {d['west_temp'].mean():.1f}°C  |  T media Ext: {d['outdoor_temp'].mean():.1f}°C"
    ax3.text(0.5, 0.05, stats, transform=ax3.transAxes, fontsize=9, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # --- Panel 4: PMV ---
    ax4 = axes[3]
    ax4.axhspan(-0.5, 0.5, color='green', alpha=0.15, label='Confort [-0.5, +0.5]')
    ax4.axhline(y=-0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
    ax4.axhline(y=0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
    ax4.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    ax4.plot(d['datetime'], d['pmv_east'], color='#9C27B0', linewidth=1.3, label='Este', alpha=0.9)
    ax4.plot(d['datetime'], d['pmv_west'], color='#00BCD4', linewidth=1.3, label='Oeste', alpha=0.9)
    ax4.set_ylabel('PMV', fontsize=11)
    ax4.set_title('Índice PMV Zona Este y Oeste', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    pe = pct_comfort(d['pmv_east'].values)
    pw = pct_comfort(d['pmv_west'].values)
    ax4.text(0.5, 0.05, f"% en confort PMV → Este: {pe:.0f}%  |  Oeste: {pw:.0f}%",
             transform=ax4.transAxes, fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew', edgecolor='green', alpha=0.9), fontweight='bold')

    # --- Panel 5: Electroválvulas (inferidas del heating rate) ---
    ax5 = axes[4]
    for col, label, color, off in [('ev_east','Este','#9C27B0',1), ('ev_west','Oeste','#00BCD4',0)]:
        y_vals = d[col] * 0.7 + off
        y_base = np.full(len(d), off)
        ax5.fill_between(d['datetime'], y_base, y_vals, step='post', color=color, alpha=0.6)
        ax5.step(d['datetime'], y_vals, where='post', color=color, linewidth=0.5, alpha=0.8)
        pct = 100 * d[col].mean()
        ax5.text(d['datetime'].iloc[-1] + pd.Timedelta(hours=1), off + 0.35,
                 f'{pct:.0f}%', fontsize=11, ha='left', va='center', color=color, fontweight='bold')
    ax5.set_ylabel('Calefacción activa', fontsize=11)
    ax5.set_title('Calefacción Activa por Zona (heating rate > 0)', fontsize=12, fontweight='bold')
    ax5.set_yticks([0.35, 1.35]); ax5.set_yticklabels(['Oeste', 'Este'])
    ax5.set_ylim(-0.2, 2.2); ax5.grid(True, alpha=0.2, linestyle='--', axis='x')
    ax5.set_xlabel('Fecha', fontsize=12)
    ax5.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax5.xaxis.set_minor_locator(mdates.HourLocator())
    plt.xticks(rotation=45, ha='right')
    for ax in axes: ax.set_xlim(d['datetime'].min(), d['datetime'].max())
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Guardado: {fname}")
    print(f"  Consumo: {kwh:.0f} kWh | Confort Este: {pe:.0f}% | Confort Oeste: {pw:.0f}%")
    plt.close()

# ===================== 3 PERIODOS =====================
jan = onoff[(onoff['month'] == 1) & (onoff['day'] >= 10) & (onoff['day'] <= 13)].copy()
print("=== 10-13 ENERO ===")
plot_onoff(jan, '10 al 13 de Enero', '%d Ene %Hh', '/workspaces/sinergym/onoff_jan10_13.png')

hot = onoff[(onoff['month'] == 2) & (onoff['day'] >= 7) & (onoff['day'] <= 9)].copy()
print("\n=== 7-9 FEBRERO (MÁS CALUROSOS) ===")
plot_onoff(hot, '3 Días Más Calurosos: 7-9 Feb', '%d Feb %Hh', '/workspaces/sinergym/onoff_hottest.png')

cold = onoff[(onoff['month'] == 6) & (onoff['day'] >= 27) & (onoff['day'] <= 29)].copy()
print("\n=== 27-29 JUNIO (MÁS FRÍOS) ===")
plot_onoff(cold, '3 Días Más Fríos: 27-29 Jun', '%d Jun %Hh', '/workspaces/sinergym/onoff_coldest.png')
