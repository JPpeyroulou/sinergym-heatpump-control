"""Plot ON/OFF setpoint curve: Cost vs PMV deviation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from simulations: (setpoint, cost, mean_dev, cold_w_pct, hot_pct, kwh)
onoff = [
    (20, 39593,  0.446, 99.6,  5.9,  6593),
    (21, 53755,  0.279, 99.1,  8.3,  8935),
    (22, 69503,  0.128, 94.0, 11.3, 11650),
    (23, 87148,  0.047, 66.7, 15.1, 14655),
    (25, 125689, 0.066,  3.2, 37.4, 21082),
    (26, 143002, 0.140,  0.9, 58.4, 24206),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('ON/OFF Baseline: Efecto del setpoint del termostato', fontsize=16, fontweight='bold')

sps    = [s[0] for s in onoff]
costs  = [s[1]/1000 for s in onoff]
devs   = [s[2] for s in onoff]
colds  = [s[3] for s in onoff]
hots   = [s[4] for s in onoff]
kwhs   = [s[5]/1000 for s in onoff]
sp_lbl = [str(s) for s in sps]

# ---- 1. Costo vs Mean PMV Deviation ----
ax = axes[0, 0]
ax.plot(devs, costs, 'o-', color='#333333', linewidth=2.5, markersize=10, zorder=3)
for i, sp in enumerate(sps):
    ax.annotate(f'{sp}°C', (devs[i], costs[i]),
                textcoords='offset points', xytext=(10, 5),
                fontsize=11, fontweight='bold', color='#333333')
# Highlight optimal zone (SP=23)
ax.plot(devs[3], costs[3], 'o', color='#4CAF50', markersize=16, zorder=4, markeredgewidth=2, markerfacecolor='none')
ax.annotate('Óptimo\n(SP=23°C)', (devs[3], costs[3]),
            textcoords='offset points', xytext=(-60, 15),
            fontsize=10, color='#4CAF50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.set_xlabel('Mean PMV Deviation (menor = mejor comfort)', fontsize=12)
ax.set_ylabel('Costo anual ($k)', fontsize=12)
ax.set_title('Costo vs Comfort', fontsize=13)
ax.grid(alpha=0.3)

# ---- 2. Setpoint vs Costo y kWh ----
ax = axes[0, 1]
ax2 = ax.twinx()
bars = ax.bar(sp_lbl, costs, color='#2196F3', alpha=0.7, label='Costo ($k)')
line = ax2.plot(sp_lbl, kwhs, 'o-', color='#FF5722', linewidth=2, markersize=8, label='kWh (miles)')
ax.set_xlabel('Setpoint (°C)', fontsize=12)
ax.set_ylabel('Costo anual ($k)', fontsize=12, color='#2196F3')
ax2.set_ylabel('kWh/año (miles)', fontsize=12, color='#FF5722')
ax.set_title('Consumo y costo por setpoint', fontsize=13)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# ---- 3. Violación frío vs calor ----
ax = axes[1, 0]
x = np.arange(len(sps))
w = 0.35
ax.bar(x - w/2, colds, w, color='#2196F3', alpha=0.8, label='Frío inv. (%)')
ax.bar(x + w/2, hots, w, color='#FF5722', alpha=0.8, label='Calor (%)')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}°C' for s in sps])
ax.set_xlabel('Setpoint', fontsize=12)
ax.set_ylabel('% horas violación', fontsize=12)
ax.set_title('Violación comfort: Frío invierno vs Calor', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
# Highlight tradeoff
ax.axhline(20, color='gray', linestyle='--', alpha=0.4)
ax.text(4.5, 22, '20%', fontsize=9, color='gray')

# ---- 4. Tabla resumen ----
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['SP (°C)', 'kWh/año', 'Costo ($)', 'Frío inv%', 'Calor%', 'PMV dev'],
]
for s in onoff:
    table_data.append([
        f'{s[0]}°C',
        f'{s[5]:,}',
        f'${s[1]:,}',
        f'{s[3]:.1f}%',
        f'{s[4]:.1f}%',
        f'{s[2]:.3f}',
    ])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='bold')
    # Highlight SP=23 row
    if row == 4 and col > 0:
        cell.set_facecolor('#E8F5E9')
ax.set_title('Resumen de métricas', fontsize=13, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('curva_onoff_setpoints.png', dpi=150, bbox_inches='tight')
print('Gráfica guardada: curva_onoff_setpoints.png')
