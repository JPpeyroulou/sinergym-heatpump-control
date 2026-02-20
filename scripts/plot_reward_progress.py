#!/usr/bin/env python3
"""Plot reward evolution across training episodes for a PPO run."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

CSV_PATH = '/workspaces/sinergym/Eplus-PPO-training-nuestroMultizona_2026-02-20_00-28-res1/progress.csv'
OUT_PATH = '/workspaces/sinergym/reward_progress_ppo.png'

df = pd.read_csv(CSV_PATH)
# Drop incomplete last episode if much shorter
df = df[df['length(timesteps)'] > 5000].copy()

episodes = df['episode_num']

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Total reward with std band
ax1 = axes[0]
ax1.plot(episodes, df['mean_reward'], 'b-o', linewidth=2, markersize=4, label='Reward total (media)')
ax1.fill_between(episodes,
                 df['mean_reward'] - df['std_reward'],
                 df['mean_reward'] + df['std_reward'],
                 alpha=0.15, color='blue')
ax1.set_ylabel('Reward (media por timestep)')
ax1.set_title('PPO 20 ep (02-20_00-28) — Evolución de la Reward por Episodio', fontsize=13)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Panel 2: Energy term vs Comfort term
ax2 = axes[1]
ax2.plot(episodes, df['mean_reward_energy_term'], 'r-s', linewidth=2, markersize=4, label='Término energía')
ax2.plot(episodes, df['mean_reward_comfort_term'], 'g-^', linewidth=2, markersize=4, label='Término confort')
ax2.fill_between(episodes,
                 df['mean_reward_energy_term'] - df['std_reward_energy_term'],
                 df['mean_reward_energy_term'] + df['std_reward_energy_term'],
                 alpha=0.1, color='red')
ax2.fill_between(episodes,
                 df['mean_reward_comfort_term'] - df['std_reward_comfort_term'],
                 df['mean_reward_comfort_term'] + df['std_reward_comfort_term'],
                 alpha=0.1, color='green')
ax2.set_ylabel('Componente de Reward')
ax2.set_title('Descomposición: Energía vs Confort')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Panel 3: Comfort violation % and mean power demand
ax3 = axes[2]
color_viol = 'tab:orange'
ax3.plot(episodes, df['comfort_violation_time(%)'], color=color_viol, marker='D',
         linewidth=2, markersize=4, label='Violación confort (%)')
ax3.set_ylabel('Violación confort (%)', color=color_viol)
ax3.tick_params(axis='y', labelcolor=color_viol)
ax3.set_xlabel('Episodio de entrenamiento')
ax3.set_title('Violación de confort y Potencia media')
ax3.grid(True, alpha=0.3)

ax3b = ax3.twinx()
color_pow = 'tab:purple'
ax3b.plot(episodes, df['mean_power_demand'], color=color_pow, marker='v',
          linewidth=2, markersize=4, label='Potencia media (W)')
ax3b.set_ylabel('Potencia media (W)', color=color_pow)
ax3b.tick_params(axis='y', labelcolor=color_pow)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Guardado en {OUT_PATH}')

# Print summary
print(f'\nResumen:')
print(f'  Ep 1:  reward={df.iloc[0]["mean_reward"]:.1f}, energy={df.iloc[0]["mean_reward_energy_term"]:.1f}, comfort={df.iloc[0]["mean_reward_comfort_term"]:.1f}, violation={df.iloc[0]["comfort_violation_time(%)"]:.1f}%')
print(f'  Ep {int(df.iloc[-1]["episode_num"])}:  reward={df.iloc[-1]["mean_reward"]:.1f}, energy={df.iloc[-1]["mean_reward_energy_term"]:.1f}, comfort={df.iloc[-1]["mean_reward_comfort_term"]:.1f}, violation={df.iloc[-1]["comfort_violation_time(%)"]:.1f}%')
print(f'  Mejora reward: {df.iloc[-1]["mean_reward"] - df.iloc[0]["mean_reward"]:.1f} ({(1 - df.iloc[-1]["mean_reward"]/df.iloc[0]["mean_reward"])*100:.0f}%)')
