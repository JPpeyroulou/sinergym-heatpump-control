"""Calcula la reward del baseline ON/OFF usando la misma función de reward
que los agentes SAC/PPO, para poder comparar mean_reward directamente.

Usa los datos hora a hora de /workspaces/sinergym/baseline_onoff/eplusout.csv
y aplica NuestroRewardMultizona con los parámetros actuales.

Ejecutar:
    python scripts/eval_baseline_onoff.py
"""

import csv
import json
import statistics
from datetime import datetime

# --------------------------------------------------------------------------- #
#                              PARAMETROS REWARD                              #
# --------------------------------------------------------------------------- #
W = 0.5
LAMBDA_E = 1.0
LAMBDA_T = 40.0

TARIFA_JSON = '/workspaces/sinergym/sinergym/data/tarifas/tarifas_ute.json'
BASELINE_CSV = '/workspaces/sinergym/baseline_onoff/eplusout.csv'

# --------------------------------------------------------------------------- #
#                           FUNCIONES (mismas que rewards.py)                  #
# --------------------------------------------------------------------------- #

def pmv_simple(tdb, rh):
    """PMV simplificado (clo=0.57, met=1.2, vr=0.1)."""
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh


def calc_violation(tdb, rh):
    """Penalización bilateral d + d² (misma que _calculate_pmv_violation)."""
    pmv = pmv_simple(tdb, rh)
    d = max(-0.5 - pmv, 0.0) + max(pmv - 0.5, 0.0)
    violation = d + d * d
    return pmv, violation


# --------------------------------------------------------------------------- #
#                              CARGAR TARIFAS                                 #
# --------------------------------------------------------------------------- #

with open(TARIFA_JSON) as f:
    tarifa = json.load(f)

precio_punta = tarifa['precios']['punta']
precio_fuera_punta = tarifa['precios']['fuera_de_punta']
punta_inicio = tarifa['horarios']['punta_inicio']
punta_fin = tarifa['horarios']['punta_fin']
dias_map = {
    'lunes': 0, 'martes': 1, 'miercoles': 2,
    'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6
}
dias_punta = [dias_map[d] for d in tarifa['horarios']['dias_punta']]

# --------------------------------------------------------------------------- #
#                         PROCESAR DATOS ON/OFF                               #
# --------------------------------------------------------------------------- #

rewards = []
energy_terms = []
comfort_terms = []
violations_east = 0
violations_west = 0
total_steps = 0
total_kwh = 0.0

with open(BASELINE_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # --- Parsear fecha/hora (formato EnergyPlus: ' 07/21  01:00:00') ---
        dt_str = row['Date/Time'].strip()
        parts = dt_str.split()
        month_day = parts[0].split('/')
        month, day = int(month_day[0]), int(month_day[1])
        hour = int(parts[1].split(':')[0]) if len(parts) > 1 else 0
        if hour == 24:
            hour = 0
        try:
            current_dt = datetime(2025, month, day, hour)
        except ValueError:
            continue

        weekday = current_dt.weekday()

        # --- Precio de energía ---
        price_kwh = precio_fuera_punta
        if weekday in dias_punta and punta_inicio <= hour <= punta_fin:
            price_kwh = precio_punta

        # --- Energía ---
        power = float(row['BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)'])
        energy_penalty = -(power / 1000.0) * price_kwh
        total_kwh += power / 1000.0

        # --- PMV y violación (zonas East y West, igual que la AI) ---
        tdb_e = float(row['EAST PERIMETER:Zone Air Temperature [C](Hourly)'])
        rh_e = float(row['EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)'])
        tdb_w = float(row['WEST PERIMETER:Zone Air Temperature [C](Hourly)'])
        rh_w = float(row['WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)'])

        pmv_e, viol_e = calc_violation(tdb_e, rh_e)
        pmv_w, viol_w = calc_violation(tdb_w, rh_w)

        # --- Reward (misma ecuación que NuestroRewardMultizona.__call__) ---
        energy_term = LAMBDA_E * W * energy_penalty
        comfort_term = LAMBDA_T * (1 - W) * (-(viol_e + viol_w))
        reward = energy_term + comfort_term

        rewards.append(reward)
        energy_terms.append(energy_term)
        comfort_terms.append(comfort_term)

        if abs(pmv_e) > 0.5:
            violations_east += 1
        if abs(pmv_w) > 0.5:
            violations_west += 1
        total_steps += 1

# --------------------------------------------------------------------------- #
#                              RESULTADOS                                     #
# --------------------------------------------------------------------------- #

mean_r = statistics.mean(rewards)
std_r = statistics.stdev(rewards)
mean_et = statistics.mean(energy_terms)
mean_ct = statistics.mean(comfort_terms)
viol_east_pct = violations_east / total_steps * 100
viol_west_pct = violations_west / total_steps * 100
viol_total_pct = (violations_east + violations_west) / (total_steps * 2) * 100

print('=' * 65)
print('BASELINE ON/OFF - REWARD CALCULADA')
print(f'  Parámetros: lambda_E={LAMBDA_E}, lambda_T={LAMBDA_T}, W={W}')
print(f'  Penalización: bilateral d + d² (PMV fuera de [-0.5, +0.5])')
print('=' * 65)
print(f'  Timesteps:         {total_steps}')
print(f'  mean_reward:       {mean_r:.2f} +/- {std_r:.2f}')
print(f'  energy_term:       {mean_et:.2f}')
print(f'  comfort_term:      {mean_ct:.2f}')
print(f'  comfort_viol East: {viol_east_pct:.1f}%')
print(f'  comfort_viol West: {viol_west_pct:.1f}%')
print(f'  comfort_viol Total:{viol_total_pct:.1f}%')
print(f'  kWh/año:           {total_kwh:.0f}')
print()
print('TARGET para SAC/PPO: superar mean_reward = {:.2f}'.format(mean_r))
