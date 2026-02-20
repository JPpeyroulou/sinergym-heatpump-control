"""Run ON/OFF EnergyPlus simulations with different thermostat setpoints.

Modifies the IDF file's Setpoint_Constante_22 schedule value, runs EnergyPlus,
and collects results for comparison.

Usage:
    python scripts/run_onoff_setpoints.py
"""

import os, shutil, subprocess, csv, json, statistics, re
from datetime import datetime

BASE_IDF = 'sinergym/data/buildings/20260211/idf_multiplesZonas_termostato.idf'
WEATHER  = 'sinergym/data/weather/URY_Montevideo.epw'
OUT_ROOT = 'baseline_setpoints'
SETPOINTS = [20, 21, 22, 23, 25, 26]

TARIFA_JSON = 'sinergym/data/tarifas/tarifas_ute.json'
with open(TARIFA_JSON) as f:
    tj = json.load(f)
PRECIO_PUNTA = tj['precios']['punta']
PRECIO_FUERA = tj['precios']['fuera_de_punta']
PUNTA_INI = tj['horarios']['punta_inicio']
PUNTA_FIN = tj['horarios']['punta_fin']
_DAY_MAP = {'lunes':0,'martes':1,'miercoles':2,'jueves':3,'viernes':4,'sabado':5,'domingo':6}
DIAS_PUNTA = {_DAY_MAP[d] for d in tj['horarios']['dias_punta']}

YEAR = 2000
WINTER = {5, 6, 7, 8, 9}

def pmv_simple(tdb, rh):
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh

def price_at(month, day, hour):
    dt = datetime(YEAR, month, max(1, min(28, day)), hour)
    if dt.weekday() in DIAS_PUNTA and PUNTA_INI <= hour <= PUNTA_FIN:
        return PRECIO_PUNTA
    return PRECIO_FUERA

def modify_idf(src_path, dst_path, setpoint_value):
    """Replace the setpoint value in Setpoint_Constante_22 schedule."""
    with open(src_path) as f:
        content = f.read()

    pattern = r'(Setpoint_Constante_22,\s*!- Name\s*Temperature,\s*!- Schedule Type Limits Name\s*Through: 12/31,\s*!- Field 1\s*For: AllDays,\s*!- Field 2\s*Until: 24:00,\s*!- Field 3\s*)\s*[\d.]+(\s*;\s*!- Field 4)'
    replacement = rf'\g<1> {setpoint_value}\2'
    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        raise ValueError(f'Could not find Setpoint_Constante_22 pattern in {src_path}')

    with open(dst_path, 'w') as f:
        f.write(new_content)
    print(f'  Setpoint modified to {setpoint_value}°C in {dst_path}')

def run_eplus(idf_path, weather_path, output_dir):
    """Run EnergyPlus simulation."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ['energyplus', '-w', weather_path, '-d', output_dir, idf_path]
    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f'  ERROR: EnergyPlus failed with code {result.returncode}')
        print(result.stderr[-500:] if result.stderr else 'No stderr')
        return False
    print(f'  EnergyPlus completed successfully')
    return True

def analyze_eplusout(csv_path, setpoint):
    """Analyze eplusout.csv and return metrics dict."""
    total_kwh = 0.0
    total_cost = 0.0
    kwh_punta = 0.0
    cold_viol = 0
    hot_viol = 0
    total_h = 0
    cold_viol_w = 0
    total_h_w = 0
    pmv_devs = []
    kwh_by_month = {m: 0.0 for m in range(1, 13)}

    hp_col = None
    te_col = None
    tw_col = None
    he_col = None
    hw_col = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            cl = col.lower()
            if 'bombacalor' in cl and 'electricity rate' in cl.lower():
                hp_col = col
            elif 'east perimeter' in cl and 'temperature' in cl.lower() and 'zone air' in cl.lower():
                te_col = col
            elif 'west perimeter' in cl and 'temperature' in cl.lower() and 'zone air' in cl.lower():
                tw_col = col
            elif 'east perimeter' in cl and 'humidity' in cl.lower() and 'zone air' in cl.lower():
                he_col = col
            elif 'west perimeter' in cl and 'humidity' in cl.lower() and 'zone air' in cl.lower():
                hw_col = col

        if not all([hp_col, te_col, tw_col, he_col, hw_col]):
            print(f'  WARNING: Missing columns. Found: hp={hp_col}, te={te_col}, tw={tw_col}, he={he_col}, hw={hw_col}')
            print(f'  Available: {reader.fieldnames[:5]}...')
            return None

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            dt_str = row['Date/Time'].strip().split()
            md = dt_str[0].split('/')
            mo = int(md[0])
            day = int(md[1])
            hr_parts = dt_str[1].split(':')
            hr = int(hr_parts[0]) - 1
            if hr < 0:
                hr = 0

            pw = float(row[hp_col])
            t_e = float(row[te_col])
            t_w = float(row[tw_col])
            h_e = float(row[he_col])
            h_w = float(row[hw_col])

            kwh = pw / 1000.0
            pr = price_at(mo, day, hr)
            total_kwh += kwh
            total_cost += kwh * pr
            kwh_by_month[mo] += kwh

            dt = datetime(YEAR, mo, max(1, min(28, day)), hr)
            if dt.weekday() in DIAS_PUNTA and PUNTA_INI <= hr <= PUNTA_FIN:
                kwh_punta += kwh

            pmv_e = pmv_simple(t_e, h_e)
            pmv_w = pmv_simple(t_w, h_w)
            avg_pmv = (pmv_e + pmv_w) / 2.0

            total_h += 1
            is_cold = pmv_e < -0.5 or pmv_w < -0.5
            is_hot = pmv_e > 0.5 or pmv_w > 0.5
            if is_cold:
                cold_viol += 1
            if is_hot:
                hot_viol += 1

            dev = 0.0
            if avg_pmv < -0.5:
                dev = -0.5 - avg_pmv
            elif avg_pmv > 0.5:
                dev = avg_pmv - 0.5
            pmv_devs.append(dev)

            if mo in WINTER:
                total_h_w += 1
                if is_cold:
                    cold_viol_w += 1

    return {
        'setpoint': setpoint,
        'total_kwh': total_kwh,
        'total_cost': total_cost,
        'pct_punta': (kwh_punta / total_kwh * 100) if total_kwh > 0 else 0,
        'cold_pct': cold_viol / total_h * 100 if total_h else 0,
        'hot_pct': hot_viol / total_h * 100 if total_h else 0,
        'cold_w_pct': cold_viol_w / total_h_w * 100 if total_h_w else 0,
        'mean_dev': statistics.mean(pmv_devs) if pmv_devs else 0,
        'summer_kwh': sum(kwh_by_month[m] for m in [11, 12, 1, 2, 3]),
        'winter_kwh': sum(kwh_by_month[m] for m in WINTER),
    }


# ---- MAIN ----
os.makedirs(OUT_ROOT, exist_ok=True)

results = []
for sp in SETPOINTS:
    print(f'\n{"="*60}')
    print(f'  SETPOINT = {sp}°C')
    print(f'{"="*60}')

    sim_dir = os.path.join(OUT_ROOT, f'sp_{sp}')
    idf_path = os.path.join(sim_dir, 'model.idf')
    os.makedirs(sim_dir, exist_ok=True)

    modify_idf(BASE_IDF, idf_path, sp)

    if not run_eplus(idf_path, WEATHER, sim_dir):
        print(f'  SKIPPING setpoint {sp} due to EnergyPlus error')
        continue

    # Convert .eso to .csv using ReadVarsESO (must run from output dir, no args)
    eso_file = os.path.join(sim_dir, 'eplusout.eso')
    if os.path.exists(eso_file):
        subprocess.run(['ReadVarsESO'], cwd=sim_dir,
                       capture_output=True, timeout=60)
        print(f'  ReadVarsESO completed')

    eplusout = os.path.join(sim_dir, 'eplusout.csv')
    if not os.path.exists(eplusout):
        print(f'  WARNING: {eplusout} not found, skipping')
        continue

    metrics = analyze_eplusout(eplusout, sp)
    if metrics:
        results.append(metrics)
        print(f'  kWh={metrics["total_kwh"]:,.0f}  Cost=${metrics["total_cost"]:,.0f}  '
              f'Cold_inv={metrics["cold_w_pct"]:.1f}%  Hot={metrics["hot_pct"]:.1f}%')

# ---- RESULTS TABLE ----
if results:
    print(f'\n\n{"="*100}')
    print(f'  COMPARACIÓN ON/OFF POR SETPOINT DE TERMOSTATO')
    print(f'{"="*100}')
    hdr = f'{"SP°C":>6} | {"kWh/año":>10} | {"Costo($)":>10} | {"%Punta":>7} | {"Frío%":>7} | {"Calor%":>7} | {"Frío_Inv%":>10} | {"MeanDev":>8} | {"kWh_Ver":>10} | {"kWh_Inv":>10}'
    print(hdr)
    print('-' * 100)
    for r in results:
        print(f'{r["setpoint"]:>6} | {r["total_kwh"]:>10,.0f} | {r["total_cost"]:>10,.0f} | '
              f'{r["pct_punta"]:>6.1f}% | {r["cold_pct"]:>6.1f}% | {r["hot_pct"]:>6.1f}% | '
              f'{r["cold_w_pct"]:>9.1f}% | {r["mean_dev"]:>8.3f} | '
              f'{r["summer_kwh"]:>10,.0f} | {r["winter_kwh"]:>10,.0f}')

    print(f'\nResultados guardados en {OUT_ROOT}/')
