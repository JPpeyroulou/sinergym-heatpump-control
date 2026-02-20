import pandas as pd
import numpy as np

base = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"
obs = pd.read_csv(f"{base}/observations.csv")

obs['day_idx'] = obs['month'].astype(int) * 100 + obs['day_of_month'].astype(int)
daily_mean = obs.groupby('day_idx')['outdoor_temperature'].mean()

best_sum = 999
best_start = None
days_list = daily_mean.index.tolist()
vals_list = daily_mean.values.tolist()

for i in range(len(days_list) - 2):
    s = vals_list[i] + vals_list[i+1] + vals_list[i+2]
    if s < best_sum:
        best_sum = s
        best_start = i

d1, d2, d3 = days_list[best_start], days_list[best_start+1], days_list[best_start+2]
m1, day1 = d1 // 100, d1 % 100
m2, day2 = d2 // 100, d2 % 100
m3, day3 = d3 // 100, d3 % 100

print(f"3 días consecutivos más fríos (T exterior media):")
print(f"  Día 1: mes {m1}, día {day1} -> T media ext = {vals_list[best_start]:.1f}°C")
print(f"  Día 2: mes {m2}, día {day2} -> T media ext = {vals_list[best_start+1]:.1f}°C")
print(f"  Día 3: mes {m3}, día {day3} -> T media ext = {vals_list[best_start+2]:.1f}°C")
print(f"  Promedio 3 días: {best_sum/3:.1f}°C")
