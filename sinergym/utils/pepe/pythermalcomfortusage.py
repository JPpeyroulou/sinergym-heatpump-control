
# import os
# import tempfile
# import numba

# # Crear un directorio temporal con permisos adecuados
# temp_dir = os.path.join(tempfile.gettempdir(), 'numba_cache')
# os.makedirs(temp_dir, exist_ok=True)

# # Configurar Numba para usar este directorio
# numba.config.CACHE_DIR = temp_dir
# print(f"Usando directorio de caché alternativo: {temp_dir}")

# Ahora ejecuta tu código original
from pythermalcomfort.models import pmv_ppd_ashrae
# ... resto de tu código ...
import sys, json

def calcular_confort(tdb, rh):
    resultado = pmv_ppd_ashrae(
        tdb=tdb, tr=tdb, vr=0.1, rh=rh, met=1.2, clo=0.57, wme=0
    )
    return resultado['pmv'], resultado['ppd']

if __name__ == "__main__":
    tdb = float(sys.argv[1])
    rh = float(sys.argv[2])
    pmv, ppd = calcular_confort(tdb, rh)
    print(json.dumps({"pmv": pmv, "ppd": ppd}))


# # Parámetros de entrada comunes
# tdb = 25.0    # Temperatura de bulbo seco [°C]
# tr = 25.0     # Temperatura radiante media [°C]
# vr = 0.1      # Velocidad relativa del aire [m/s]
# rh = 50       # Humedad relativa [%]
# met = 1.2     # Tasa metabólica [met]
# clo = 0.5     # Resistencia térmica de la ropa [clo]

# # Cálculo PMV según ASHRAE 55
# resultado_ashrae = pmv_ppd_ashrae(
#     tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, wme=0
# )

# # Cálculo PMV según ISO 7730
# resultado_iso = pmv_ppd_iso(
#     tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, wme=0
# )

# # Mostrar resultados
# print("======== RESULTADOS PMV ========")
# print(f"Parámetros:")
# print(f"  Temperatura: {tdb}°C")
# print(f"  Humedad: {rh}%")
# print(f"  Velocidad aire: {vr} m/s")
# print(f"  Metabolismo: {met} met")
# print(f"  Ropa: {clo} clo")
# print()

# print("ASHRAE 55:")
# print(f"  PMV: {resultado_ashrae['pmv']:.3f}")
# print(f"  PPD: {resultado_ashrae['ppd']:.1f}%")

# print()

# print("ISO 7730:")
# print(f"  PMV: {resultado_iso['pmv']:.3f}")
# print(f"  PPD: {resultado_iso['ppd']:.1f}%")
