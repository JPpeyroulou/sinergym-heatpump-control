"""Compare ON/OFF vs PPO using comprehensive metrics.

Usage:
    python scripts/compare_metrics.py <ppo_training_dir>
    
Example:
    python scripts/compare_metrics.py Eplus-PPO-training-nuestroMultizona_2026-02-17_23-57-res1
"""

import csv, json, os, sys, statistics
from datetime import datetime

TARIFA_JSON = '/workspaces/sinergym/sinergym/data/tarifas/tarifas_ute.json'
BASELINE_CSV = '/workspaces/sinergym/baseline_onoff/eplusout.csv'

with open(TARIFA_JSON) as f:
    tarifa = json.load(f)

precio_punta = tarifa['precios']['punta']
precio_fuera_punta = tarifa['precios']['fuera_de_punta']
punta_inicio = tarifa['horarios']['punta_inicio']
punta_fin = tarifa['horarios']['punta_fin']
dias_map = {'lunes':0,'martes':1,'miercoles':2,'jueves':3,
            'viernes':4,'sabado':5,'domingo':6}
dias_punta = [dias_map[d] for d in tarifa['horarios']['dias_punta']]

WINTER = [5, 6, 7, 8, 9]
SUMMER = [11, 12, 1, 2, 3]


def pmv_simple(tdb, rh):
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh


def analyze_csv(path, hp_col, zones):
    """Analyze an EnergyPlus output CSV.
    
    Args:
        path: path to eplusout.csv
        hp_col: column name for heat pump power [W]
        zones: list of (temp_col, humidity_col) tuples
    """
    m = dict(cost=0, kwh=0, kwh_peak=0, kwh_offpeak=0,
             viol_cold=0, viol_hot=0, zones=0, pmv_dev_sum=0,
             viol_winter=0, zones_winter=0,
             viol_summer=0, zones_summer=0)

    with open(path) as f:
        for row in csv.DictReader(f):
            dt = row['Date/Time'].strip().split()
            md = dt[0].split('/')
            mo, dy = int(md[0]), int(md[1])
            hr = int(dt[1].split(':')[0]) if len(dt) > 1 else 0
            if hr == 24:
                hr = 0
            try:
                cdt = datetime(2025, mo, dy, hr)
            except ValueError:
                continue

            wd = cdt.weekday()
            is_pk = wd in dias_punta and punta_inicio <= hr <= punta_fin
            pk = precio_punta if is_pk else precio_fuera_punta

            pw = float(row[hp_col])
            kwh = pw / 1000.0
            m['kwh'] += kwh
            m['cost'] += kwh * pk
            if is_pk:
                m['kwh_peak'] += kwh
            else:
                m['kwh_offpeak'] += kwh

            for t_col, h_col in zones:
                tdb = float(row[t_col])
                rh = float(row[h_col])
                pmv = pmv_simple(tdb, rh)
                d = max(-0.5 - pmv, 0) + max(pmv - 0.5, 0)
                m['pmv_dev_sum'] += d
                m['zones'] += 1

                if pmv < -0.5:
                    m['viol_cold'] += 1
                elif pmv > 0.5:
                    m['viol_hot'] += 1

                if mo in WINTER:
                    m['zones_winter'] += 1
                    if abs(pmv) > 0.5:
                        m['viol_winter'] += 1
                if mo in SUMMER:
                    m['zones_summer'] += 1
                    if abs(pmv) > 0.5:
                        m['viol_summer'] += 1

    return m


def pct(n, t):
    return n / t * 100 if t > 0 else 0


def find_ppo_eplusout(ppo_dir):
    eval_dir = os.path.join(ppo_dir, 'evaluation')
    eps = sorted(
        [d for d in os.listdir(eval_dir) if d.startswith('episode-')],
        key=lambda x: int(x.split('-')[1])
    )
    for ep in reversed(eps):
        candidate = os.path.join(eval_dir, ep, 'output', 'eplusout.csv')
        if os.path.exists(candidate):
            return candidate, ep
    return None, None


def find_columns(path):
    with open(path) as f:
        cols = next(csv.reader(f))
    hp = [c for c in cols if 'Heat Pump' in c and 'Rate' in c and '[W]' in c]
    et = [c for c in cols if 'EAST' in c and 'Temperature' in c and 'Zone Air' in c]
    eh = [c for c in cols if 'EAST' in c and 'Humidity' in c and 'Zone Air' in c]
    wt = [c for c in cols if 'WEST' in c and 'Temperature' in c and 'Zone Air' in c]
    wh = [c for c in cols if 'WEST' in c and 'Humidity' in c and 'Zone Air' in c]
    return hp[0], [(et[0], eh[0]), (wt[0], wh[0])]


# ---- Main ----
ppo_dir = sys.argv[1] if len(sys.argv) > 1 else 'Eplus-PPO-training-nuestroMultizona_2026-02-17_23-57-res1'

# ON/OFF
onoff_hp = 'BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)'
onoff_zones = [
    ('EAST PERIMETER:Zone Air Temperature [C](Hourly)',
     'EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)'),
    ('WEST PERIMETER:Zone Air Temperature [C](Hourly)',
     'WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)')
]
onoff = analyze_csv(BASELINE_CSV, onoff_hp, onoff_zones)

# PPO
ppo_csv, ppo_ep = find_ppo_eplusout(ppo_dir)
if ppo_csv is None:
    print("ERROR: No se encontro eplusout.csv en evaluacion del PPO")
    sys.exit(1)

ppo_hp, ppo_zones = find_columns(ppo_csv)
ppo = analyze_csv(ppo_csv, ppo_hp, ppo_zones)

# ---- Print ----
def row(name, v1, v2, fmt, lower_better=True):
    s1 = fmt.format(v1)
    s2 = fmt.format(v2)
    if lower_better:
        ver = 'PPO MEJOR' if v2 < v1 - 0.01 else ('EMPATE' if abs(v2 - v1) < 0.01 else 'ON/OFF mejor')
    else:
        ver = 'PPO MEJOR' if v2 > v1 + 0.01 else ('EMPATE' if abs(v2 - v1) < 0.01 else 'ON/OFF mejor')
    print(f'{name:>30} | {s1:>12} | {s2:>12} | {ver:>14}')


print('=' * 75)
print('METRICAS COMPLETAS: ON/OFF vs PPO')
print(f'PPO dir: {ppo_dir}')
print(f'PPO eval episode: {ppo_ep}')
print('=' * 75)
print()
print(f'{"METRICA":>30} | {"ON/OFF":>12} | {"PPO":>12} | {"Veredicto":>14}')
print('-' * 75)

print('--- ENERGIA Y COSTO ---')
row('Costo anual ($)', onoff['cost'], ppo['cost'], '${:,.0f}', True)
row('kWh/ano', onoff['kwh'], ppo['kwh'], '{:,.0f}', True)
row('% kWh en punta', pct(onoff['kwh_peak'], onoff['kwh']),
    pct(ppo['kwh_peak'], ppo['kwh']), '{:.1f}%', True)

print()
print('--- CONFORT (lo que importa) ---')
row('Viol% por FRIO', pct(onoff['viol_cold'], onoff['zones']),
    pct(ppo['viol_cold'], ppo['zones']), '{:.1f}%', True)
row('Viol% INVIERNO (May-Sep)', pct(onoff['viol_winter'], onoff['zones_winter']),
    pct(ppo['viol_winter'], ppo['zones_winter']), '{:.1f}%', True)
row('Mean PMV deviation', onoff['pmv_dev_sum'] / onoff['zones'],
    ppo['pmv_dev_sum'] / ppo['zones'], '{:.3f}', True)

print()
print('--- CONTEXTO (no accionable) ---')
row('Viol% por CALOR', pct(onoff['viol_hot'], onoff['zones']),
    pct(ppo['viol_hot'], ppo['zones']), '{:.1f}%', True)
row('Viol% VERANO (Nov-Mar)', pct(onoff['viol_summer'], onoff['zones_summer']),
    pct(ppo['viol_summer'], ppo['zones_summer']), '{:.1f}%', True)
row('Viol% TOTAL', pct(onoff['viol_cold'] + onoff['viol_hot'], onoff['zones']),
    pct(ppo['viol_cold'] + ppo['viol_hot'], ppo['zones']), '{:.1f}%', True)
