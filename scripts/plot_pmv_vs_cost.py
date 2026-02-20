"""Plot PMV deviation vs Cost for ON/OFF setpoints + PPO + SAC."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from simulations
# ON/OFF setpoints: (setpoint, cost, mean_dev, cold_w_pct)
onoff = [
    (20, 39593,  0.446, 99.6),
    (21, 53755,  0.279, 99.1),
    (22, 69503,  0.128, 94.0),
    (23, 87148,  0.047, 66.7),
    (25, 125689, 0.066,  3.2),
    (26, 143002, 0.140,  0.9),
]

# PPO and SAC from compare_full.py results
ppo = {'label': 'PPO λ=25', 'cost': 127884, 'mean_dev': 0.024, 'cold_w': 21.4}
sac = {'label': 'SAC λ=25', 'cost': 150904, 'mean_dev': 0.083, 'cold_w': 3.8}

fig, ax = plt.subplots(figsize=(12, 7))

# ON/OFF curve
sp_labels = [str(s[0]) for s in onoff]
x_onoff = [s[2] for s in onoff]  # mean PMV deviation
y_onoff = [s[1] / 1000 for s in onoff]  # cost in $k

ax.plot(x_onoff, y_onoff, 'o-', color='#888888', linewidth=2.5,
        markersize=10, label='ON/OFF (por setpoint)', zorder=3)

for i, sp in enumerate(onoff):
    ax.annotate(f'SP={sp[0]}°C\n({sp[3]:.0f}% frío inv)',
                (x_onoff[i], y_onoff[i]),
                textcoords='offset points',
                xytext=(12, -5 if i % 2 == 0 else 10),
                fontsize=8, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=0.8))

# PPO
ax.plot(ppo['mean_dev'], ppo['cost']/1000, 's', color='#2196F3',
        markersize=14, label=f'PPO λ=25 ({ppo["cold_w"]:.0f}% frío inv)', zorder=4)
ax.annotate('PPO', (ppo['mean_dev'], ppo['cost']/1000),
            textcoords='offset points', xytext=(10, 5),
            fontsize=11, fontweight='bold', color='#2196F3')

# SAC
ax.plot(sac['mean_dev'], sac['cost']/1000, 'D', color='#FF5722',
        markersize=14, label=f'SAC λ=25 ({sac["cold_w"]:.0f}% frío inv)', zorder=4)
ax.annotate('SAC', (sac['mean_dev'], sac['cost']/1000),
            textcoords='offset points', xytext=(10, 5),
            fontsize=11, fontweight='bold', color='#FF5722')

# Ideal zone (low dev, low cost)
ax.axvspan(0, 0.05, color='green', alpha=0.05)
ax.axhspan(0, 80, color='green', alpha=0.05)
ax.text(0.015, 35, 'Zona ideal\n(bajo costo + bajo PMV dev)',
        fontsize=9, color='green', alpha=0.6, style='italic')

ax.set_xlabel('Mean PMV Deviation (menor = mejor comfort)', fontsize=13)
ax.set_ylabel('Costo anual ($k)', fontsize=13)
ax.set_title('Frontera Costo vs Comfort: ON/OFF setpoints vs RL agents', fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-0.01, 0.50)
ax.set_ylim(30, 165)

plt.tight_layout()
plt.savefig('pmv_vs_costo.png', dpi=150, bbox_inches='tight')
print('Gráfica guardada: pmv_vs_costo.png')
