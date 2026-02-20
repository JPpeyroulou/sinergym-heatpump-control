import pandas as pd
import numpy as np

base = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"
obs = pd.read_csv(f"{base}/observations.csv")

obs['day_idx'] = obs['month'].astype(int) * 100 + obs['day_of_month'].astype(int)
daily_mean = obs.groupby('day_idx')['outdoor_temperature'].mean()

days_list = daily_mean.index.tolist()
vals_list = daily_mean.values.tolist()

results = []
for i in range(len(days_list) - 2):
    s = vals_list[i] + vals_list[i+1] + vals_list[i+2]
    m1, d1 = days_list[i] // 100, days_list[i] % 100
    results.append((s/3, i, m1, d1))

results.sort(key=lambda x: x[0])

print("Top 10 períodos de 3 días más fríos:")
for rank, (avg, idx, m, d) in enumerate(results[:10]):
    d1 = days_list[idx]; d2 = days_list[idx+1]; d3 = days_list[idx+2]
    m1,dy1 = d1//100, d1%100
    m2,dy2 = d2//100, d2%100
    m3,dy3 = d3//100, d3%100
    overlap = any(d1 <= 629 and d3 >= 627 for _ in [0])  # check overlap with jun 27-29
    flag = " <-- YA USADO" if (m1==6 and dy1==27) else ""
    print(f"  #{rank+1}: {dy1}/{m1} - {dy3}/{m3}  T media={avg:.1f}°C{flag}")
