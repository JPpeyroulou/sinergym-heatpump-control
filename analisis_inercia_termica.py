import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

base = "/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/monitor"
obs = pd.read_csv(f"{base}/observations.csv")
act = pd.read_csv(f"{base}/simulated_actions.csv")

min_len = min(len(obs), len(act))
obs = obs.iloc[:min_len].copy()
act = act.iloc[:min_len].copy()
df = pd.concat([obs.reset_index(drop=True), act.reset_index(drop=True)], axis=1)

# =========================================================
# 1. MODELO: Propiedades térmicas del edificio
# =========================================================
with open('/workspaces/sinergym/Eplus-SAC-training-nuestroMultizona_2026-02-18_00-27_EVALUATION-res1/episode-20/idf_multiplesZonas_Bomba2.epJSON') as f:
    model = json.load(f)

zones_info = {}
for name, z in model.get('Zone', {}).items():
    if name == 'PLENUM':
        continue
    vol = z.get('volume', 0)
    h = z.get('ceiling_height', 2.44)
    area_floor = vol / h if h > 0 else 0
    zones_info[name] = {
        'volume': vol,
        'floor_area': area_floor,
        'height': h,
    }

rho_air = 1.2  # kg/m³
cp_air = 1005  # J/(kg·K)

materials = model.get('Material', {})
mat_nomass = model.get('Material:NoMass', {})

print("=" * 70)
print("ANÁLISIS DE INERCIA TÉRMICA POR ZONA")
print("=" * 70)

print("\n1. PROPIEDADES DESDE EL MODELO DEL EDIFICIO")
print("-" * 50)

for zname in ['NORTH PERIMETER', 'SOUTH PERIMETER', 'EAST PERIMETER', 'WEST PERIMETER', 'CORE']:
    zi = zones_info[zname]
    C_air = rho_air * zi['volume'] * cp_air  # J/K
    C_air_kJ = C_air / 1000
    C_air_Wh = C_air / 3600

    surfaces = model.get('BuildingSurface:Detailed', {})
    n_ext_walls = sum(1 for s in surfaces.values()
                      if s.get('zone_name') == zname and
                      s.get('outside_boundary_condition') == 'Outdoors')
    has_windows = any(
        s.get('building_surface_name', '') in
        [sn for sn, sv in surfaces.items() if sv.get('zone_name') == zname]
        for s in model.get('FenestrationSurface:Detailed', {}).values()
    )

    print(f"\n  {zname}:")
    print(f"    Volumen: {zi['volume']:.1f} m³")
    print(f"    Área planta: {zi['floor_area']:.1f} m²")
    print(f"    Capacidad térmica aire: {C_air_Wh:.0f} Wh/K ({C_air_kJ:.0f} kJ/K)")
    print(f"    Paredes exteriores: {n_ext_walls}")
    print(f"    Ventanas: {'Sí' if has_windows else 'No'}")

# =========================================================
# 2. DATOS: Estimación de constante de tiempo térmica
# =========================================================
print("\n\n2. ESTIMACIÓN DE CONSTANTE DE TIEMPO DESDE DATOS")
print("-" * 50)
print("   (τ = tiempo que tarda la zona en perder 63% de ΔT respecto al exterior)")
print("   (Mayor τ → mayor inercia térmica)")

zone_cols = {
    'NORTH PERIMETER': ('north_perimeter_air_temperature', 'electrovalve_north'),
    'SOUTH PERIMETER': ('south_perimeter_air_temperature', 'electrovalve_south'),
    'EAST PERIMETER':  ('east_perimeter_air_temperature', 'electrovalve_east'),
    'WEST PERIMETER':  ('west_perimeter_air_temperature', 'electrovalve_west'),
}

def exp_decay(t, T_ext, dT0, tau):
    return T_ext + dT0 * np.exp(-t / tau)

results = {}

for zname, (tcol, evcol) in zone_cols.items():
    df['heating_on'] = (df[evcol] == 1).astype(int)
    df['heat_off_group'] = (df['heating_on'].diff() != 0).cumsum()

    taus = []
    segments = []
    for gid, group in df[df['heating_on'] == 0].groupby('heat_off_group'):
        if len(group) < 6:
            continue
        T_zone = group[tcol].values
        T_ext = group['outdoor_temperature'].values
        dT = T_zone - T_ext

        if dT[0] < 1.0:
            continue
        if np.all(np.diff(dT) >= 0):
            continue

        t = np.arange(len(group), dtype=float)
        try:
            popt, _ = curve_fit(exp_decay, t, T_zone,
                                p0=[T_ext.mean(), dT[0], 10],
                                bounds=([T_ext.min()-5, 0, 1], [T_ext.max()+5, dT[0]+10, 200]),
                                maxfev=5000)
            tau = popt[2]
            if 1 < tau < 150:
                taus.append(tau)
                segments.append((group.index[0], len(group), tau, T_zone[0], T_ext.mean()))
        except:
            pass

    if taus:
        tau_med = np.median(taus)
        tau_mean = np.mean(taus)
        tau_std = np.std(taus)
        results[zname] = {
            'tau_median': tau_med,
            'tau_mean': tau_mean,
            'tau_std': tau_std,
            'n_segments': len(taus),
            'segments': segments
        }
        print(f"\n  {zname}:")
        print(f"    τ mediana: {tau_med:.1f} horas")
        print(f"    τ media:   {tau_mean:.1f} ± {tau_std:.1f} horas")
        print(f"    Segmentos analizados: {len(taus)}")
    else:
        print(f"\n  {zname}: sin suficientes datos de enfriamiento libre")
        results[zname] = None

# Para CORE (no tiene electroválvula), analizar períodos nocturnos
print(f"\n  CORE:")
core_temps = df['north_perimeter_air_temperature'].values  # proxy: core no se monitorea directamente
print(f"    (CORE no tiene electroválvula ni sensor directo - se infiere del modelo)")
C_core = rho_air * zones_info['CORE']['volume'] * cp_air / 3600
print(f"    Capacidad térmica del aire: {C_core:.0f} Wh/K")
print(f"    Al no tener paredes exteriores, su inercia térmica es la mayor de todas")

# =========================================================
# 3. GRÁFICO COMPARATIVO
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel 1: Constantes de tiempo
ax1 = axes[0, 0]
zone_names_short = []
tau_vals = []
tau_errs = []
colors_bars = []
color_map = {
    'NORTH PERIMETER': '#4CAF50',
    'SOUTH PERIMETER': '#FF9800',
    'EAST PERIMETER': '#9C27B0',
    'WEST PERIMETER': '#00BCD4',
}
for zname in ['NORTH PERIMETER', 'SOUTH PERIMETER', 'EAST PERIMETER', 'WEST PERIMETER']:
    r = results.get(zname)
    if r:
        zone_names_short.append(zname.replace(' PERIMETER', ''))
        tau_vals.append(r['tau_median'])
        tau_errs.append(r['tau_std'])
        colors_bars.append(color_map[zname])

bars = ax1.bar(zone_names_short, tau_vals, color=colors_bars, alpha=0.8, edgecolor='gray')
ax1.errorbar(zone_names_short, tau_vals, yerr=tau_errs, fmt='none', ecolor='black', capsize=5)
for bar, val in zip(bars, tau_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}h', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.set_ylabel('Constante de tiempo τ (horas)', fontsize=12)
ax1.set_title('Constante de Tiempo Térmica por Zona\n(mayor = más inercia)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Volúmenes
ax2 = axes[0, 1]
vol_names = []
vol_vals = []
vol_colors = []
for zname in ['NORTH PERIMETER', 'SOUTH PERIMETER', 'EAST PERIMETER', 'WEST PERIMETER', 'CORE']:
    vol_names.append(zname.replace(' PERIMETER', ''))
    vol_vals.append(zones_info[zname]['volume'])
    vol_colors.append(color_map.get(zname, '#607D8B'))
bars2 = ax2.bar(vol_names, vol_vals, color=vol_colors, alpha=0.8, edgecolor='gray')
for bar, val in zip(bars2, vol_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.0f} m³', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax2.set_ylabel('Volumen (m³)', fontsize=12)
ax2.set_title('Volumen por Zona', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Ejemplo de curva de enfriamiento - mejor segmento para cada zona
ax3 = axes[1, 0]
for zname in ['NORTH PERIMETER', 'SOUTH PERIMETER', 'EAST PERIMETER', 'WEST PERIMETER']:
    r = results.get(zname)
    if not r or not r['segments']:
        continue
    tcol = zone_cols[zname][0]
    best = sorted(r['segments'], key=lambda x: x[1], reverse=True)[0]
    idx_start = best[0]
    n_pts = best[1]
    seg = df.iloc[idx_start:idx_start+n_pts]
    t_hours = np.arange(n_pts)
    T_zone = seg[tcol].values
    T_ext_mean = seg['outdoor_temperature'].mean()

    color = color_map[zname]
    label_short = zname.replace(' PERIMETER', '')
    ax3.plot(t_hours, T_zone - T_ext_mean, color=color, linewidth=2, label=f'{label_short} (τ≈{best[2]:.0f}h)')

ax3.set_xlabel('Tiempo (horas)', fontsize=12)
ax3.set_ylabel('ΔT zona - T_ext (°C)', fontsize=12)
ax3.set_title('Ejemplo de Curva de Enfriamiento Libre\n(sin calefacción activa)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Panel 4: Resumen tabla
ax4 = axes[1, 1]
ax4.axis('off')
table_data = [['Zona', 'Vol (m³)', 'C_aire\n(Wh/K)', 'τ mediana\n(horas)', 'Paredes\next.', 'Ventanas']]

ext_walls = {'NORTH PERIMETER': 1, 'SOUTH PERIMETER': 1, 'EAST PERIMETER': 1, 'WEST PERIMETER': 1, 'CORE': 0}
has_win = {'NORTH PERIMETER': 'Sí (2)', 'SOUTH PERIMETER': 'Sí (2)', 'EAST PERIMETER': 'Sí (1)', 'WEST PERIMETER': 'Sí (1)', 'CORE': 'No'}

for zname in ['NORTH PERIMETER', 'SOUTH PERIMETER', 'EAST PERIMETER', 'WEST PERIMETER', 'CORE']:
    zi = zones_info[zname]
    C = rho_air * zi['volume'] * cp_air / 3600
    r = results.get(zname)
    tau_str = f"{r['tau_median']:.1f}" if r else "N/A (interior)"
    label = zname.replace(' PERIMETER', '')
    table_data.append([label, f"{zi['volume']:.0f}", f"{C:.0f}", tau_str,
                       str(ext_walls[zname]), has_win[zname]])

table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
for i in range(len(table_data[0])):
    table[0, i].set_facecolor('#E3F2FD')
    table[0, i].set_text_props(fontweight='bold')

ax4.set_title('Resumen de Propiedades Térmicas por Zona', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/workspaces/sinergym/inercia_termica.png', dpi=150, bbox_inches='tight')
print("\n\nGráfico guardado en /workspaces/sinergym/inercia_termica.png")
