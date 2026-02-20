import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

base_sac = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"
obs_sac = pd.read_csv(f"{base_sac}/observations.csv")
act_sac = pd.read_csv(f"{base_sac}/simulated_actions.csv")
ml = min(len(obs_sac), len(act_sac))
obs_sac = obs_sac.iloc[:ml].copy(); act_sac = act_sac.iloc[:ml].copy()
sac = pd.concat([obs_sac.reset_index(drop=True), act_sac.reset_index(drop=True)], axis=1)
sac['heat_pump_kW'] = sac['heat_pump_power'] / 1000.0
md = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
bd = pd.Timestamp('2025-01-01')
sac['datetime'] = [bd + pd.Timedelta(days=sum(md[i] for i in range(1,int(r['month'])))+int(r['day_of_month'])-1, hours=int(r['hour'])) for _,r in sac.iterrows()]

onoff = pd.read_csv('/workspaces/sinergym/baseline_onoff/eplusout.csv')
onoff.columns = onoff.columns.str.strip()
dt_list = []
for dt_str in onoff['Date/Time']:
    dt_str = dt_str.strip(); parts = dt_str.split(); m_d = parts[0].split('/')
    month, day = int(m_d[0]), int(m_d[1]); hour = int(parts[1].split(':')[0]) if len(parts)>1 else 0
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
onoff['hr_east'] = onoff['LOZARADIANTE_ZONAEAST:Zone Radiant HVAC Heating Rate [W](Hourly)']
onoff['hr_west'] = onoff['LOZARADIANTE_ZONAWEST:Zone Radiant HVAC Heating Rate [W](Hourly)']
onoff['ev_east'] = (onoff['hr_east']>0).astype(int); onoff['ev_west'] = (onoff['hr_west']>0).astype(int)
onoff['month'] = onoff['datetime'].dt.month; onoff['day'] = onoff['datetime'].dt.day
def pmv(T,RH): return -7.4928+0.2882*T-0.0020*RH+0.0004*T*RH
def T_from_pmv(t,RH): return (t+7.4928+0.0020*RH)/(0.2882+0.0004*RH)
def pct_c(v): return 100*((v>=-0.5)&(v<=0.5)).sum()/len(v)
onoff['pmv_east']=pmv(onoff['east_temp'],onoff['east_rh']); onoff['pmv_west']=pmv(onoff['west_temp'],onoff['west_rh'])

sac_s = sac[(sac['month']==8)&(sac['day_of_month']>=4)&(sac['day_of_month']<=6)].copy()
onoff_s = onoff[(onoff['month']==8)&(onoff['day']>=4)&(onoff['day']<=6)].copy()

fig, axes = plt.subplots(5, 1, figsize=(20, 22), sharex=True, gridspec_kw={'height_ratios': [1.8,1.0,2.5,2.2,1.8]})

ax1 = axes[0]
ax1.plot(sac_s['datetime'], sac_s['heat_pump_kW'], color='#2196F3', lw=1.2, label='SAC', alpha=0.9)
ax1.fill_between(sac_s['datetime'], 0, sac_s['heat_pump_kW'], color='#2196F3', alpha=0.15)
ax1.plot(onoff_s['datetime'], onoff_s['heat_pump_kW'], color='#FF5722', lw=1.2, label='ON/OFF', alpha=0.9)
ax1.fill_between(onoff_s['datetime'], 0, onoff_s['heat_pump_kW'], color='#FF5722', alpha=0.10)
ax1.set_ylabel('Potencia (kW)', fontsize=11); ax1.legend(fontsize=11, loc='upper right')
ax1.set_title('4 al 6 de Agosto (T ext media 8.6°C, mín -1°C, máx 22°C)\nSAC λ_T=25 vs Baseline ON/OFF', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--'); ax1.set_ylim(bottom=0)
ks=sac_s['heat_pump_kW'].sum(); ko=onoff_s['heat_pump_kW'].sum()
ax1.text(0.5, 0.92, f"SAC: {ks:,.0f} kWh (media {sac_s['heat_pump_kW'].mean():.1f} kW)  |  ON/OFF: {ko:,.0f} kWh (media {onoff_s['heat_pump_kW'].mean():.1f} kW)",
         transform=ax1.transAxes, fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax2 = axes[1]
ax2.step(sac_s['datetime'], sac_s['bomba'], where='post', color='#E91E63', lw=1.4)
ax2.fill_between(sac_s['datetime'], 0, sac_s['bomba'], step='post', color='#E91E63', alpha=0.1)
ax2.set_ylabel('Setpoint (°C)', fontsize=11)
ax2.set_title('Setpoint de la Bomba (SAC — ON/OFF usa setpoint fijo)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

ax3 = axes[2]
rh_avg = sac_s[['east_perimeter_air_humidity','west_perimeter_air_humidity']].mean(axis=1)
Tlo=T_from_pmv(-0.5,rh_avg); Thi=T_from_pmv(0.5,rh_avg)
ax3.fill_between(sac_s['datetime'], Tlo, Thi, color='green', alpha=0.12, label='Confort PMV [-0.5, 0.5]')
ax3.plot(sac_s['datetime'], sac_s['east_perimeter_air_temperature'], color='#9C27B0', lw=1.5, label='Este (SAC)')
ax3.plot(sac_s['datetime'], sac_s['west_perimeter_air_temperature'], color='#00BCD4', lw=1.5, label='Oeste (SAC)')
ax3.plot(onoff_s['datetime'], onoff_s['east_temp'], color='#9C27B0', lw=1.5, ls='--', label='Este (ON/OFF)', alpha=0.7)
ax3.plot(onoff_s['datetime'], onoff_s['west_temp'], color='#00BCD4', lw=1.5, ls='--', label='Oeste (ON/OFF)', alpha=0.7)
ax3.plot(sac_s['datetime'], sac_s['outdoor_temperature'], color='#F44336', lw=1.8, ls=':', label='Exterior', alpha=0.7)
ax3.set_ylabel('Temperatura (°C)', fontsize=11)
ax3.set_title('Temperaturas Este y Oeste — SAC (sólida) vs ON/OFF (punteada)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9); ax3.grid(True, alpha=0.3, linestyle='--')
st = (f"T media Este: SAC={sac_s['east_perimeter_air_temperature'].mean():.1f}°C / ON/OFF={onoff_s['east_temp'].mean():.1f}°C  |  "
      f"T media Oeste: SAC={sac_s['west_perimeter_air_temperature'].mean():.1f}°C / ON/OFF={onoff_s['west_temp'].mean():.1f}°C")
ax3.text(0.5, 0.05, st, transform=ax3.transAxes, fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax4 = axes[3]
ax4.axhspan(-0.5, 0.5, color='green', alpha=0.15, label='Confort [-0.5, +0.5]')
ax4.axhline(y=-0.5, color='green', lw=1.0, ls='--', alpha=0.6); ax4.axhline(y=0.5, color='green', lw=1.0, ls='--', alpha=0.6)
ax4.axhline(y=0, color='gray', lw=0.5, alpha=0.3)
spe=pmv(sac_s['east_perimeter_air_temperature'],sac_s['east_perimeter_air_humidity'])
spw=pmv(sac_s['west_perimeter_air_temperature'],sac_s['west_perimeter_air_humidity'])
ax4.plot(sac_s['datetime'], spe, color='#9C27B0', lw=1.3, label='Este (SAC)')
ax4.plot(sac_s['datetime'], spw, color='#00BCD4', lw=1.3, label='Oeste (SAC)')
ax4.plot(onoff_s['datetime'], onoff_s['pmv_east'], color='#9C27B0', lw=1.3, ls='--', label='Este (ON/OFF)', alpha=0.7)
ax4.plot(onoff_s['datetime'], onoff_s['pmv_west'], color='#00BCD4', lw=1.3, ls='--', label='Oeste (ON/OFF)', alpha=0.7)
ax4.set_ylabel('PMV', fontsize=11)
ax4.set_title('PMV Este y Oeste — SAC (sólida) vs ON/OFF (punteada)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9); ax4.grid(True, alpha=0.3, linestyle='--')
pe_s=pct_c(spe.values); pw_s=pct_c(spw.values); pe_o=pct_c(onoff_s['pmv_east'].values); pw_o=pct_c(onoff_s['pmv_west'].values)
ax4.text(0.5, 0.05, f"% en confort → Este: SAC {pe_s:.0f}% / ON/OFF {pe_o:.0f}%  |  Oeste: SAC {pw_s:.0f}% / ON/OFF {pw_o:.0f}%",
         transform=ax4.transAxes, fontsize=10, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew', edgecolor='green', alpha=0.9), fontweight='bold')

ax5 = axes[4]
for col,label,color,off in [('electrovalve_east','E SAC','#9C27B0',3),('electrovalve_west','O SAC','#00BCD4',2)]:
    y=sac_s[col]*0.7+off; yb=np.full(len(sac_s),off)
    ax5.fill_between(sac_s['datetime'], yb, y, step='post', color=color, alpha=0.6)
    ax5.step(sac_s['datetime'], y, where='post', color=color, lw=0.5, alpha=0.8)
    ax5.text(sac_s['datetime'].iloc[-1]+pd.Timedelta(hours=1), off+0.35, f'{100*sac_s[col].mean():.0f}%', fontsize=10, ha='left', va='center', color=color, fontweight='bold')
for col,label,color,off in [('ev_east','E on/off','#9C27B0',1),('ev_west','O on/off','#00BCD4',0)]:
    y=onoff_s[col]*0.7+off; yb=np.full(len(onoff_s),off)
    ax5.fill_between(onoff_s['datetime'], yb, y, step='post', color=color, alpha=0.3, hatch='//')
    ax5.step(onoff_s['datetime'], y, where='post', color=color, lw=0.5, alpha=0.5, ls='--')
    ax5.text(onoff_s['datetime'].iloc[-1]+pd.Timedelta(hours=4), off+0.35, f'{100*onoff_s[col].mean():.0f}%', fontsize=10, ha='left', va='center', color=color, fontstyle='italic')
ax5.set_ylabel('Calefacción', fontsize=11)
ax5.set_title('Calefacción Activa: SAC (sólido) vs ON/OFF (rayado)', fontsize=12, fontweight='bold')
ax5.set_yticks([0.35,1.35,2.35,3.35]); ax5.set_yticklabels(['O on/off','E on/off','O SAC','E SAC'])
ax5.set_ylim(-0.2, 4.2); ax5.grid(True, alpha=0.2, linestyle='--', axis='x')
ax5.set_xlabel('Fecha', fontsize=12)
ax5.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18]))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d Ago %Hh'))
ax5.xaxis.set_minor_locator(mdates.HourLocator())
plt.xticks(rotation=45, ha='right')
for ax in axes: ax.set_xlim(sac_s['datetime'].min(), sac_s['datetime'].max())
plt.tight_layout()
plt.savefig('/workspaces/sinergym/compare_aug4_6.png', dpi=150, bbox_inches='tight')
print(f"Guardado: /workspaces/sinergym/compare_aug4_6.png")
print(f"  Consumo: SAC={ks:.0f} kWh vs ON/OFF={ko:.0f} kWh")
print(f"  Confort Este:  SAC={pe_s:.0f}% vs ON/OFF={pe_o:.0f}%")
print(f"  Confort Oeste: SAC={pw_s:.0f}% vs ON/OFF={pw_o:.0f}%")
