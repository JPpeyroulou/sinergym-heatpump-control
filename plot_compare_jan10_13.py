import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ===================== CARGAR DATOS SAC =====================
base_sac = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"
obs_sac = pd.read_csv(f"{base_sac}/observations.csv")
act_sac = pd.read_csv(f"{base_sac}/simulated_actions.csv")
ml = min(len(obs_sac), len(act_sac))
obs_sac = obs_sac.iloc[:ml].copy()
act_sac = act_sac.iloc[:ml].copy()
sac = pd.concat([obs_sac.reset_index(drop=True), act_sac.reset_index(drop=True)], axis=1)
sac['heat_pump_kW'] = sac['heat_pump_power'] / 1000.0

months_days = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
base_date = pd.Timestamp('2025-01-01')
offsets_list = []
for _, row in sac.iterrows():
    m, dd = int(row['month']), int(row['day_of_month'])
    offsets_list.append(sum(months_days[i] for i in range(1, m)) + dd - 1)
sac['datetime'] = [base_date + pd.Timedelta(days=d, hours=int(sac.iloc[i]['hour'])) for i, d in enumerate(offsets_list)]

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

def pmv(T, RH):
    return -7.4928 + 0.2882 * T - 0.0020 * RH + 0.0004 * T * RH
def T_from_pmv(target, RH):
    return (target + 7.4928 + 0.0020 * RH) / (0.2882 + 0.0004 * RH)
def pct_comfort(vals):
    return 100 * ((vals >= -0.5) & (vals <= 0.5)).sum() / len(vals)

onoff['pmv_east'] = pmv(onoff['east_temp'], onoff['east_rh'])
onoff['pmv_west'] = pmv(onoff['west_temp'], onoff['west_rh'])

# ===================== FUNCIÓN PLOT =====================
def plot_comparison(sac_slice, onoff_slice, title_suffix, date_fmt, fname):
    fig, axes = plt.subplots(5, 1, figsize=(20, 22), sharex=True,
                              gridspec_kw={'height_ratios': [1.8, 1.0, 2.5, 2.2, 1.8]})

    ax1 = axes[0]
    ax1.plot(sac_slice['datetime'], sac_slice['heat_pump_kW'], color='#2196F3', linewidth=1.2, label='SAC', alpha=0.9)
    ax1.fill_between(sac_slice['datetime'], 0, sac_slice['heat_pump_kW'], color='#2196F3', alpha=0.15)
    ax1.plot(onoff_slice['datetime'], onoff_slice['heat_pump_kW'], color='#FF5722', linewidth=1.2, label='ON/OFF', alpha=0.9)
    ax1.fill_between(onoff_slice['datetime'], 0, onoff_slice['heat_pump_kW'], color='#FF5722', alpha=0.10)
    ax1.set_ylabel('Potencia (kW)', fontsize=11)
    ax1.set_title(f'{title_suffix}\nSAC λ_T=25 vs Baseline ON/OFF', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    kwh_sac = sac_slice['heat_pump_kW'].sum()
    kwh_onoff = onoff_slice['heat_pump_kW'].sum()
    ax1.text(0.5, 0.92,
             f"SAC: {kwh_sac:,.0f} kWh (media {sac_slice['heat_pump_kW'].mean():.1f} kW)  |  "
             f"ON/OFF: {kwh_onoff:,.0f} kWh (media {onoff_slice['heat_pump_kW'].mean():.1f} kW)",
             transform=ax1.transAxes, fontsize=10, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    ax2 = axes[1]
    ax2.step(sac_slice['datetime'], sac_slice['bomba'], where='post', color='#E91E63', linewidth=1.4)
    ax2.fill_between(sac_slice['datetime'], 0, sac_slice['bomba'], step='post', color='#E91E63', alpha=0.1)
    ax2.set_ylabel('Setpoint (°C)', fontsize=11)
    ax2.set_title('Setpoint de la Bomba (SAC — ON/OFF usa setpoint fijo)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax3 = axes[2]
    rh_avg = sac_slice[['east_perimeter_air_humidity', 'west_perimeter_air_humidity']].mean(axis=1)
    T_lo = T_from_pmv(-0.5, rh_avg); T_hi = T_from_pmv(0.5, rh_avg)
    ax3.fill_between(sac_slice['datetime'], T_lo, T_hi, color='green', alpha=0.12, label='Confort PMV [-0.5, 0.5]')
    ax3.plot(sac_slice['datetime'], sac_slice['east_perimeter_air_temperature'],
             color='#9C27B0', linewidth=1.5, label='Este (SAC)', alpha=0.9)
    ax3.plot(sac_slice['datetime'], sac_slice['west_perimeter_air_temperature'],
             color='#00BCD4', linewidth=1.5, label='Oeste (SAC)', alpha=0.9)
    ax3.plot(onoff_slice['datetime'], onoff_slice['east_temp'],
             color='#9C27B0', linewidth=1.5, linestyle='--', label='Este (ON/OFF)', alpha=0.7)
    ax3.plot(onoff_slice['datetime'], onoff_slice['west_temp'],
             color='#00BCD4', linewidth=1.5, linestyle='--', label='Oeste (ON/OFF)', alpha=0.7)
    ax3.plot(sac_slice['datetime'], sac_slice['outdoor_temperature'],
             color='#F44336', linewidth=1.8, linestyle=':', label='Exterior', alpha=0.7)
    ax3.set_ylabel('Temperatura (°C)', fontsize=11)
    ax3.set_title('Temperaturas Zona Este y Oeste — SAC (sólida) vs ON/OFF (punteada)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    stats = (f"T media Este: SAC={sac_slice['east_perimeter_air_temperature'].mean():.1f}°C / ON/OFF={onoff_slice['east_temp'].mean():.1f}°C  |  "
             f"T media Oeste: SAC={sac_slice['west_perimeter_air_temperature'].mean():.1f}°C / ON/OFF={onoff_slice['west_temp'].mean():.1f}°C")
    ax3.text(0.5, 0.05, stats, transform=ax3.transAxes, fontsize=9, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    ax4 = axes[3]
    ax4.axhspan(-0.5, 0.5, color='green', alpha=0.15, label='Confort [-0.5, +0.5]')
    ax4.axhline(y=-0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
    ax4.axhline(y=0.5, color='green', linewidth=1.0, linestyle='--', alpha=0.6)
    ax4.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    sac_pmv_e = pmv(sac_slice['east_perimeter_air_temperature'], sac_slice['east_perimeter_air_humidity'])
    sac_pmv_w = pmv(sac_slice['west_perimeter_air_temperature'], sac_slice['west_perimeter_air_humidity'])
    ax4.plot(sac_slice['datetime'], sac_pmv_e, color='#9C27B0', linewidth=1.3, label='Este (SAC)', alpha=0.9)
    ax4.plot(sac_slice['datetime'], sac_pmv_w, color='#00BCD4', linewidth=1.3, label='Oeste (SAC)', alpha=0.9)
    ax4.plot(onoff_slice['datetime'], onoff_slice['pmv_east'], color='#9C27B0', linewidth=1.3,
             linestyle='--', label='Este (ON/OFF)', alpha=0.7)
    ax4.plot(onoff_slice['datetime'], onoff_slice['pmv_west'], color='#00BCD4', linewidth=1.3,
             linestyle='--', label='Oeste (ON/OFF)', alpha=0.7)
    ax4.set_ylabel('PMV', fontsize=11)
    ax4.set_title('PMV Zona Este y Oeste — SAC (sólida) vs ON/OFF (punteada)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    pe_s = pct_comfort(sac_pmv_e.values); pw_s = pct_comfort(sac_pmv_w.values)
    pe_o = pct_comfort(onoff_slice['pmv_east'].values); pw_o = pct_comfort(onoff_slice['pmv_west'].values)
    ax4.text(0.5, 0.05,
             f"% en confort → Este: SAC {pe_s:.0f}% / ON/OFF {pe_o:.0f}%  |  Oeste: SAC {pw_s:.0f}% / ON/OFF {pw_o:.0f}%",
             transform=ax4.transAxes, fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew', edgecolor='green', alpha=0.9), fontweight='bold')

    ax5 = axes[4]
    for col, label, color, off in [('electrovalve_east','Este','#9C27B0',1), ('electrovalve_west','Oeste','#00BCD4',0)]:
        y_vals = sac_slice[col] * 0.7 + off
        y_base = np.full(len(sac_slice), off)
        ax5.fill_between(sac_slice['datetime'], y_base, y_vals, step='post', color=color, alpha=0.6)
        ax5.step(sac_slice['datetime'], y_vals, where='post', color=color, linewidth=0.5, alpha=0.8)
        pct = 100 * sac_slice[col].mean()
        ax5.text(sac_slice['datetime'].iloc[-1] + pd.Timedelta(hours=1), off + 0.35,
                 f'{pct:.0f}%', fontsize=11, ha='left', va='center', color=color, fontweight='bold')
    ax5.set_ylabel('Electroválvulas SAC', fontsize=11)
    ax5.set_title('Electroválvulas SAC (ON/OFF usa todas abiertas por termostato)', fontsize=12, fontweight='bold')
    ax5.set_yticks([0.35, 1.35]); ax5.set_yticklabels(['Oeste', 'Este'])
    ax5.set_ylim(-0.2, 2.2); ax5.grid(True, alpha=0.2, linestyle='--', axis='x')
    ax5.set_xlabel('Fecha', fontsize=12)
    ax5.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax5.xaxis.set_minor_locator(mdates.HourLocator())
    plt.xticks(rotation=45, ha='right')
    for ax in axes: ax.set_xlim(sac_slice['datetime'].min(), sac_slice['datetime'].max())
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Guardado: {fname}")
    print(f"  Consumo: SAC={kwh_sac:.0f} kWh vs ON/OFF={kwh_onoff:.0f} kWh")
    print(f"  Confort Este:  SAC={pe_s:.0f}% vs ON/OFF={pe_o:.0f}%")
    print(f"  Confort Oeste: SAC={pw_s:.0f}% vs ON/OFF={pw_o:.0f}%")
    plt.close()

# ===================== 10-13 Enero =====================
sac_jan = sac[(sac['month'] == 1) & (sac['day_of_month'] >= 10) & (sac['day_of_month'] <= 13)].copy()
onoff_jan = onoff[(onoff['month'] == 1) & (onoff['day'] >= 10) & (onoff['day'] <= 13)].copy()
print("=== 10-13 ENERO ===")
plot_comparison(sac_jan, onoff_jan, '10 al 13 de Enero', '%d Ene %Hh',
                '/workspaces/sinergym/compare_jan10_13.png')

# ===================== 7-9 Feb (calurosos) =====================
sac_hot = sac[(sac['month'] == 2) & (sac['day_of_month'] >= 7) & (sac['day_of_month'] <= 9)].copy()
onoff_hot = onoff[(onoff['month'] == 2) & (onoff['day'] >= 7) & (onoff['day'] <= 9)].copy()
print("\n=== 3 DÍAS MÁS CALUROSOS (7-9 Feb) ===")
plot_comparison(sac_hot, onoff_hot, '3 Días Más Calurosos: 7-9 Feb (T ext media 25.2°C)',
                '%d Feb %Hh', '/workspaces/sinergym/compare_hottest_3days.png')

# ===================== 27-29 Jun (fríos) =====================
sac_cold = sac[(sac['month'] == 6) & (sac['day_of_month'] >= 27) & (sac['day_of_month'] <= 29)].copy()
onoff_cold = onoff[(onoff['month'] == 6) & (onoff['day'] >= 27) & (onoff['day'] <= 29)].copy()
print("\n=== 3 DÍAS MÁS FRÍOS (27-29 Jun) ===")
plot_comparison(sac_cold, onoff_cold, '3 Días Más Fríos: 27-29 Jun (T ext media 4.0°C)',
                '%d Jun %Hh', '/workspaces/sinergym/compare_coldest_3days.png')
