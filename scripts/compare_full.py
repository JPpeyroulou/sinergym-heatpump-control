"""Compare ON/OFF vs PPO vs SAC with comprehensive metrics and plots.

Usage:
    python scripts/compare_full.py
"""

import csv, json, os, statistics
from datetime import datetime

# ------------------------------------------------------------------ #
#                           CONFIG                                    #
# ------------------------------------------------------------------ #
ONOFF_CSV = 'baseline_onoff/eplusout.csv'
PPO_CSV   = 'Eplus-PPO-training-nuestroMultizona_2026-02-19_22-21_EVALUATION-res1/episode-20/monitor/observations.csv'
SAC_CSV   = 'Eplus-SAC-training-nuestroMultizona_2026-02-19_22-21_EVALUATION-res1/episode-20/monitor/observations.csv'

TARIFA_JSON = 'sinergym/data/tarifas/tarifas_ute.json'
W = 0.5
LAMBDA_E = 1.0
LAMBDA_T = 25.0
YEAR = 2000

WINTER = {5, 6, 7, 8, 9}
SUMMER = {11, 12, 1, 2, 3}

with open(TARIFA_JSON) as f:
    tj = json.load(f)
PRECIO_PUNTA = tj['precios']['punta']
PRECIO_FUERA = tj['precios']['fuera_de_punta']
PUNTA_INI = tj['horarios']['punta_inicio']
PUNTA_FIN = tj['horarios']['punta_fin']
_DAY_MAP = {'lunes':0,'martes':1,'miercoles':2,'jueves':3,'viernes':4,'sabado':5,'domingo':6}
DIAS_PUNTA = {_DAY_MAP[d] for d in tj['horarios']['dias_punta']}

def pmv_simple(tdb, rh):
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh

def price_at(month, day, hour):
    dt = datetime(YEAR, month, max(1, min(28, day)), hour)
    if dt.weekday() in DIAS_PUNTA and PUNTA_INI <= hour <= PUNTA_FIN:
        return PRECIO_PUNTA
    return PRECIO_FUERA

# ------------------------------------------------------------------ #
#                    PROCESS AGENT CSV (observations)                 #
# ------------------------------------------------------------------ #
def process_agent(path, label):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            mo = int(float(r['month']))
            if mo == 0:
                continue
            day = int(float(r['day_of_month']))
            hr  = int(float(r['hour']))
            pw  = float(r['heat_pump_power'])
            t_e = float(r['east_perimeter_air_temperature'])
            t_w = float(r['west_perimeter_air_temperature'])
            h_e = float(r['east_perimeter_air_humidity'])
            h_w = float(r['west_perimeter_air_humidity'])
            rows.append(dict(mo=mo, day=day, hr=hr, pw=pw,
                             t_e=t_e, t_w=t_w, h_e=h_e, h_w=h_w))
    return compute_metrics(rows, label)

# ------------------------------------------------------------------ #
#                    PROCESS ON/OFF CSV (eplusout)                    #
# ------------------------------------------------------------------ #
def process_onoff(path, label):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            dt_str = r['Date/Time'].strip().split()
            md = dt_str[0].split('/')
            mo = int(md[0]); day = int(md[1])
            hr_parts = dt_str[1].split(':')
            hr = int(hr_parts[0]) - 1
            if hr < 0: hr = 0
            pw = float(r['BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)'])
            t_e = float(r['EAST PERIMETER:Zone Air Temperature [C](Hourly)'])
            t_w = float(r['WEST PERIMETER:Zone Air Temperature [C](Hourly)'])
            h_e = float(r['EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)'])
            h_w = float(r['WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)'])
            rows.append(dict(mo=mo, day=day, hr=hr, pw=pw,
                             t_e=t_e, t_w=t_w, h_e=h_e, h_w=h_w))
    return compute_metrics(rows, label)

# ------------------------------------------------------------------ #
#                       COMPUTE ALL METRICS                          #
# ------------------------------------------------------------------ #
def compute_metrics(rows, label):
    total_kwh = 0.0
    total_cost = 0.0
    kwh_punta = 0.0
    rewards = []
    pmv_list = []
    cold_viol = 0; hot_viol = 0; total_h = 0
    cold_viol_w = 0; hot_viol_w = 0; total_h_w = 0
    pmv_dev_list = []
    kwh_by_month = {m: 0.0 for m in range(1, 13)}
    cold_by_month = {m: 0 for m in range(1, 13)}
    hours_by_month = {m: 0 for m in range(1, 13)}

    for r in rows:
        mo, day, hr, pw = r['mo'], r['day'], r['hr'], r['pw']
        t_e, t_w = r['t_e'], r['t_w']
        h_e, h_w = r['h_e'], r['h_w']

        kwh = pw / 1000.0
        pr = price_at(mo, day, hr)
        cost = kwh * pr
        total_kwh += kwh
        total_cost += cost
        kwh_by_month[mo] += kwh

        dt = datetime(YEAR, mo, max(1, min(28, day)), hr)
        if dt.weekday() in DIAS_PUNTA and PUNTA_INI <= hr <= PUNTA_FIN:
            kwh_punta += kwh

        pmv_e = pmv_simple(t_e, h_e)
        pmv_w = pmv_simple(t_w, h_w)
        avg_pmv = (pmv_e + pmv_w) / 2.0
        pmv_list.append(avg_pmv)

        d_cold_e = max(-0.5 - pmv_e, 0.0)
        d_hot_e  = max(pmv_e - 0.5, 0.0) if pw > 0 else 0.0
        d_e = d_cold_e + d_hot_e
        viol_e = d_e + d_e * d_e

        d_cold_w = max(-0.5 - pmv_w, 0.0)
        d_hot_w  = max(pmv_w - 0.5, 0.0) if pw > 0 else 0.0
        d_w = d_cold_w + d_hot_w
        viol_w = d_w + d_w * d_w

        total_viol = viol_e + viol_w
        energy_term = LAMBDA_E * W * (-(kwh) * pr)
        comfort_term = LAMBDA_T * (1 - W) * (-total_viol)
        reward = energy_term + comfort_term
        rewards.append(reward)

        total_h += 1
        hours_by_month[mo] += 1
        is_cold_e = pmv_e < -0.5
        is_cold_w = pmv_w < -0.5
        is_hot_e  = pmv_e > 0.5
        is_hot_w  = pmv_w > 0.5
        if is_cold_e or is_cold_w:
            cold_viol += 1
            cold_by_month[mo] += 1
        if is_hot_e or is_hot_w:
            hot_viol += 1

        dev = 0.0
        if avg_pmv < -0.5: dev = -0.5 - avg_pmv
        elif avg_pmv > 0.5: dev = avg_pmv - 0.5
        pmv_dev_list.append(dev)

        if mo in WINTER:
            total_h_w += 1
            if is_cold_e or is_cold_w: cold_viol_w += 1
            if is_hot_e or is_hot_w: hot_viol_w += 1

    mean_rew = statistics.mean(rewards) if rewards else 0
    pct_punta = (kwh_punta / total_kwh * 100) if total_kwh > 0 else 0
    cold_pct = cold_viol / total_h * 100 if total_h else 0
    hot_pct  = hot_viol / total_h * 100 if total_h else 0
    cold_w_pct = cold_viol_w / total_h_w * 100 if total_h_w else 0
    mean_dev = statistics.mean(pmv_dev_list) if pmv_dev_list else 0

    iecc_inv = 0
    if total_cost > 0 and cold_w_pct < 100:
        iecc_inv = (total_cost / 1000) / max(cold_w_pct, 0.1)

    summer_kwh = sum(kwh_by_month[m] for m in SUMMER)
    winter_kwh = sum(kwh_by_month[m] for m in WINTER)

    return {
        'label': label,
        'total_kwh': total_kwh,
        'total_cost': total_cost,
        'pct_punta': pct_punta,
        'mean_reward': mean_rew,
        'cold_pct': cold_pct,
        'hot_pct': hot_pct,
        'cold_w_pct': cold_w_pct,
        'mean_dev': mean_dev,
        'iecc_inv': iecc_inv,
        'summer_kwh': summer_kwh,
        'winter_kwh': winter_kwh,
        'kwh_by_month': kwh_by_month,
        'cold_by_month': cold_by_month,
        'hours_by_month': hours_by_month,
        'pmv_list': pmv_list,
        'rewards': rewards,
    }

# ------------------------------------------------------------------ #
#                           MAIN                                      #
# ------------------------------------------------------------------ #
onoff = process_onoff(ONOFF_CSV, 'ON/OFF')
ppo   = process_agent(PPO_CSV, 'PPO λ=25')
sac   = process_agent(SAC_CSV, 'SAC λ=25')

# ---- Print metrics table ----
print('\n' + '='*80)
print('  COMPARACIÓN COMPLETA: ON/OFF vs PPO vs SAC (reward condicional, λ_T=25)')
print('='*80)
fmt = '{:<25} {:>15} {:>15} {:>15}'
print(fmt.format('Métrica', 'ON/OFF', 'PPO λ=25', 'SAC λ=25'))
print('-'*80)

def row(name, key, fmt_val=',.0f'):
    vals = [onoff[key], ppo[key], sac[key]]
    strs = [f'{v:{fmt_val}}' for v in vals]
    print(f'{name:<25} {strs[0]:>15} {strs[1]:>15} {strs[2]:>15}')

row('kWh/año',          'total_kwh')
row('Costo anual ($)',   'total_cost', ',.0f')
row('% kWh en punta',   'pct_punta', '.1f')
row('Mean reward',       'mean_reward', '.1f')
row('Viol% FRÍO (anual)','cold_pct', '.1f')
row('Viol% CALOR',       'hot_pct', '.1f')
row('Viol% FRÍO INVIERNO','cold_w_pct', '.1f')
row('Mean PMV deviation', 'mean_dev', '.3f')
row('kWh VERANO',         'summer_kwh', ',.0f')
row('kWh INVIERNO',       'winter_kwh', ',.0f')
row('IECC_inv',           'iecc_inv', '.3f')
print('-'*80)

best_iecc = max([onoff, ppo, sac], key=lambda x: x['iecc_inv'])
print(f'\n→ Mejor IECC_inv (mayor = mejor comfort/costo en invierno): {best_iecc["label"]}')

# ---- Consumption by month table ----
print('\n' + '='*80)
print('  CONSUMO MENSUAL (kWh)')
print('='*80)
print(f'{"Mes":>5} | {"ON/OFF":>10} | {"PPO":>10} | {"SAC":>10} | {"Estación":>10}')
print('-'*60)
for mo in range(1, 13):
    season = 'VERANO' if mo in SUMMER else ('INVIERNO' if mo in WINTER else 'Trans.')
    print(f'{mo:>5} | {onoff["kwh_by_month"][mo]:>10,.0f} | {ppo["kwh_by_month"][mo]:>10,.0f} | {sac["kwh_by_month"][mo]:>10,.0f} | {season:>10}')

# ---- Cold violation by month table ----
print('\n' + '='*80)
print('  VIOLACIÓN FRÍO POR MES (%)')
print('='*80)
print(f'{"Mes":>5} | {"ON/OFF":>10} | {"PPO":>10} | {"SAC":>10} | {"Estación":>10}')
print('-'*60)
for mo in range(1, 13):
    season = 'VERANO' if mo in SUMMER else ('INVIERNO' if mo in WINTER else 'Trans.')
    oo = onoff['cold_by_month'][mo] / max(onoff['hours_by_month'][mo], 1) * 100
    pp = ppo['cold_by_month'][mo] / max(ppo['hours_by_month'][mo], 1) * 100
    ss = sac['cold_by_month'][mo] / max(sac['hours_by_month'][mo], 1) * 100
    print(f'{mo:>5} | {oo:>9.1f}% | {pp:>9.1f}% | {ss:>9.1f}% | {season:>10}')


# ================================================================== #
#                            PLOTS                                    #
# ================================================================== #
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

months = list(range(1, 13))
month_labels = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Comparación ON/OFF vs PPO vs SAC (reward condicional, λ_T=25)', fontsize=16, fontweight='bold')

colors = {'ON/OFF': '#888888', 'PPO λ=25': '#2196F3', 'SAC λ=25': '#FF5722'}

# ---- 1. Consumo mensual (barras) ----
ax = axes[0, 0]
x = np.arange(12)
w = 0.25
for i, d in enumerate([onoff, ppo, sac]):
    vals = [d['kwh_by_month'][m] for m in months]
    ax.bar(x + i*w, vals, w, label=d['label'], color=colors[d['label']], alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(month_labels)
ax.set_ylabel('kWh')
ax.set_title('Consumo mensual (kWh)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ---- 2. Violación frío mensual (líneas) ----
ax = axes[0, 1]
for d in [onoff, ppo, sac]:
    vals = [d['cold_by_month'][m] / max(d['hours_by_month'][m], 1) * 100 for m in months]
    ax.plot(month_labels, vals, 'o-', label=d['label'], color=colors[d['label']], linewidth=2)
ax.set_ylabel('% horas con PMV < -0.5')
ax.set_title('Violación frío por mes')
ax.legend()
ax.grid(alpha=0.3)
ax.axhspan(0, 0, color='green', alpha=0.1)

# ---- 3. Resumen métricas clave (barras horizontales) ----
ax = axes[0, 2]
metrics_names = ['Costo ($k)', 'Viol% Frío Inv', 'kWh Verano', 'Mean PMV dev']
onoff_vals = [onoff['total_cost']/1000, onoff['cold_w_pct'], onoff['summer_kwh']/100, onoff['mean_dev']*100]
ppo_vals   = [ppo['total_cost']/1000, ppo['cold_w_pct'], ppo['summer_kwh']/100, ppo['mean_dev']*100]
sac_vals   = [sac['total_cost']/1000, sac['cold_w_pct'], sac['summer_kwh']/100, sac['mean_dev']*100]

y_pos = np.arange(len(metrics_names))
h = 0.25
ax.barh(y_pos - h, onoff_vals, h, label='ON/OFF', color=colors['ON/OFF'], alpha=0.85)
ax.barh(y_pos,     ppo_vals,   h, label='PPO',    color=colors['PPO λ=25'], alpha=0.85)
ax.barh(y_pos + h, sac_vals,   h, label='SAC',    color=colors['SAC λ=25'], alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_names)
ax.set_title('Métricas clave (escaladas)')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# ---- 4. PMV promedio a lo largo del año ----
ax = axes[1, 0]
window = 24 * 7  # media movil semanal
for d in [onoff, ppo, sac]:
    pmvs = d['pmv_list']
    if len(pmvs) > window:
        kernel = np.ones(window) / window
        smooth = np.convolve(pmvs, kernel, mode='valid')
        ax.plot(smooth, label=d['label'], color=colors[d['label']], linewidth=1, alpha=0.8)
ax.axhline(-0.5, color='blue', linestyle='--', alpha=0.5, label='Comfort ±0.5')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axhspan(-0.5, 0.5, color='green', alpha=0.05)
ax.set_ylabel('PMV (media semanal)')
ax.set_xlabel('Hora del año')
ax.set_title('PMV promedio durante el año')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---- 5. Reward acumulada ----
ax = axes[1, 1]
for d in [onoff, ppo, sac]:
    cumrew = np.cumsum(d['rewards'])
    ax.plot(cumrew, label=d['label'], color=colors[d['label']], linewidth=1.5)
ax.set_ylabel('Reward acumulada')
ax.set_xlabel('Hora del año')
ax.set_title('Reward acumulada')
ax.legend()
ax.grid(alpha=0.3)

# ---- 6. Radar / tabla visual IECC ----
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Métrica', 'ON/OFF', 'PPO', 'SAC'],
    ['Costo ($)', f'{onoff["total_cost"]:,.0f}', f'{ppo["total_cost"]:,.0f}', f'{sac["total_cost"]:,.0f}'],
    ['kWh/año', f'{onoff["total_kwh"]:,.0f}', f'{ppo["total_kwh"]:,.0f}', f'{sac["total_kwh"]:,.0f}'],
    ['Viol% Frío Inv', f'{onoff["cold_w_pct"]:.1f}%', f'{ppo["cold_w_pct"]:.1f}%', f'{sac["cold_w_pct"]:.1f}%'],
    ['Viol% Calor', f'{onoff["hot_pct"]:.1f}%', f'{ppo["hot_pct"]:.1f}%', f'{sac["hot_pct"]:.1f}%'],
    ['kWh Verano', f'{onoff["summer_kwh"]:,.0f}', f'{ppo["summer_kwh"]:,.0f}', f'{sac["summer_kwh"]:,.0f}'],
    ['Mean reward', f'{onoff["mean_reward"]:.1f}', f'{ppo["mean_reward"]:.1f}', f'{sac["mean_reward"]:.1f}'],
    ['IECC_inv', f'{onoff["iecc_inv"]:.3f}', f'{ppo["iecc_inv"]:.3f}', f'{sac["iecc_inv"]:.3f}'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='bold')
ax.set_title('Resumen de métricas', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('comparacion_onoff_ppo_sac.png', dpi=150, bbox_inches='tight')
print(f'\n✓ Gráfica guardada: comparacion_onoff_ppo_sac.png')

# ---- Extra: Consumo en verano detallado ----
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Detalle: Consumo en verano y violación de comfort', fontsize=14, fontweight='bold')

ax = axes2[0]
summer_months = [11, 12, 1, 2, 3]
summer_labels = ['Nov','Dic','Ene','Feb','Mar']
x = np.arange(len(summer_months))
w = 0.25
for i, d in enumerate([onoff, ppo, sac]):
    vals = [d['kwh_by_month'][m] for m in summer_months]
    ax.bar(x + i*w, vals, w, label=d['label'], color=colors[d['label']], alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(summer_labels)
ax.set_ylabel('kWh')
ax.set_title('Consumo en meses de verano')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes2[1]
winter_months = [5, 6, 7, 8, 9]
winter_labels = ['May','Jun','Jul','Ago','Sep']
x = np.arange(len(winter_months))
for i, d in enumerate([onoff, ppo, sac]):
    vals = [d['cold_by_month'][m] / max(d['hours_by_month'][m], 1) * 100 for m in winter_months]
    ax.bar(x + i*w, vals, w, label=d['label'], color=colors[d['label']], alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(winter_labels)
ax.set_ylabel('% horas con PMV < -0.5')
ax.set_title('Violación frío en invierno')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('detalle_verano_invierno.png', dpi=150, bbox_inches='tight')
print(f'✓ Gráfica guardada: detalle_verano_invierno.png')
