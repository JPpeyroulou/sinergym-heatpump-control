"""Plot zone temperatures + outdoor + thermostat setpoint across the year for ON/OFF."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import numpy as np
from datetime import datetime

SETPOINT = 22
CSV_PATH = f'baseline_setpoints_40/sp_{SETPOINT}/eplusout.csv'

outdoor, core, east, north, south, west = [], [], [], [], [], []
hours = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    outdoor_col = [c for c in cols if 'Outdoor' in c and 'Temperature' in c][0]
    core_col = [c for c in cols if 'CORE' in c and 'Zone Air Temperature' in c][0]
    east_col = [c for c in cols if 'EAST PERIMETER' in c and 'Zone Air Temperature' in c][0]
    north_col = [c for c in cols if 'NORTH PERIMETER' in c and 'Zone Air Temperature' in c][0]
    south_col = [c for c in cols if 'SOUTH PERIMETER' in c and 'Zone Air Temperature' in c][0]
    west_col = [c for c in cols if 'WEST PERIMETER' in c and 'Zone Air Temperature' in c][0]

    found_jan1 = False
    for row in reader:
        dt_str = row['Date/Time'].strip().split()
        md = dt_str[0].split('/')
        mo = int(md[0])
        day = int(md[1])
        hr_parts = dt_str[1].split(':')
        hr = int(hr_parts[0]) - 1
        if hr < 0:
            hr = 0

        if not found_jan1:
            if mo == 1 and day == 1:
                found_jan1 = True
            else:
                continue

        if mo != 1:
            continue

        try:
            dt = datetime(2000, mo, max(1, min(28, day)), hr)
        except:
            continue
        hours.append(dt)
        outdoor.append(float(row[outdoor_col]))
        core.append(float(row[core_col]))
        east.append(float(row[east_col]))
        north.append(float(row[north_col]))
        south.append(float(row[south_col]))
        west.append(float(row[west_col]))

hours = np.array(hours)
t = hours

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(t, outdoor, color='#888888', linewidth=0.8, alpha=0.5, label='Exterior')
ax1.plot(t, north, color='#1565C0', linewidth=1.0, alpha=0.8, label='North')
ax1.plot(t, south, color='#C62828', linewidth=1.0, alpha=0.8, label='South')
ax1.plot(t, east, color='#2E7D32', linewidth=1.0, alpha=0.8, label='East')
ax1.plot(t, west, color='#E65100', linewidth=1.0, alpha=0.8, label='West')
ax1.plot(t, core, color='#6A1B9A', linewidth=1.0, alpha=0.8, label='Core')

ax1.axhline(SETPOINT, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.8,
            label=f'Setpoint = {SETPOINT}°C')

ax1.set_ylabel('Temperatura (°C)', fontsize=13)
ax1.set_title(f'Temperatura de zonas - Enero - ON/OFF SP={SETPOINT}°C, Bomba=40°C (horario)',
              fontsize=15, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right', ncol=4)
ax1.grid(alpha=0.3)

# Bottom: heat pump power (daily avg)
hp_col_name = 'BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)'
hp_power = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    found_jan1_hp = False
    for row in reader:
        dt_str = row['Date/Time'].strip().split()
        md = dt_str[0].split('/')
        mo_hp = int(md[0])
        day_hp = int(md[1])
        if not found_jan1_hp:
            if mo_hp == 1 and day_hp == 1:
                found_jan1_hp = True
            else:
                continue
        if mo_hp != 1:
            continue
        hp_power.append(float(row[hp_col_name]))
hp_arr = np.array(hp_power[:len(t)])

ax2.fill_between(t, 0, hp_arr / 1000.0, color='#FF7043', alpha=0.6)
ax2.plot(t, hp_arr / 1000.0, color='#BF360C', linewidth=0.6)
ax2.set_ylabel('Bomba calor (kW)', fontsize=13)
ax2.set_xlabel('Día de Enero', fontsize=13)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.tight_layout()
plt.savefig('temperaturas_zonas_enero.png', dpi=150, bbox_inches='tight')
print('Gráfica guardada: temperaturas_zonas_enero.png')
