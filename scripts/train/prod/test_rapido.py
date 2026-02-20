#!/usr/bin/env python3
"""
Script rápido de prueba - versión simplificada
"""
import requests

HA_URL = "http://localhost:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0Mzg5YzYxZWJkZmU0NDA4YjdhNTljMzYwZTQzNjkyYSIsImlhdCI6MTc2OTU1NjQ3OCwiZXhwIjoyMDg0OTE2NDc4fQ.Z_KjATkzbD8CwSzXhzDTKl_-EwDRTAU58DBjv-uglyk"

headers = {"Authorization": f"Bearer {HA_TOKEN}"}

print("=" * 60)
print("PRUEBA RÁPIDA - HOME ASSISTANT")
print("=" * 60)
print()

# 1. Conexión
print("1. Probando conexión...", end=" ")
try:
    r = requests.get(f"{HA_URL}/api/", headers=headers, timeout=5)
    if r.status_code == 200:
        print("✅ OK")
    else:
        print(f"❌ Error {r.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# 2. Sensor
print("2. Probando sensor...", end=" ")
try:
    r = requests.get(f"{HA_URL}/api/states/sensor.north_heating_rate_sensor", headers=headers, timeout=5)
    if r.status_code == 200:
        print(f"✅ OK (valor: {r.json()['state']})")
    else:
        print(f"❌ Error {r.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Actuador (lectura)
print("3. Probando actuador (lectura)...", end=" ")
try:
    r = requests.get(f"{HA_URL}/api/states/input_number.zona_north", headers=headers, timeout=5)
    if r.status_code == 200:
        print(f"✅ OK (valor: {r.json()['state']})")
    else:
        print(f"❌ Error {r.status_code} - ¿Reiniciaste Home Assistant?")
except Exception as e:
    print(f"❌ Error: {e}")

# 4. Actuador (escritura)
print("4. Probando actuador (escritura)...", end=" ")
try:
    r = requests.post(
        f"{HA_URL}/api/services/input_number/set_value",
        headers=headers,
        json={"entity_id": "input_number.zona_north", "value": 50.0},
        timeout=5
    )
    if r.status_code in [200, 201]:
        # Verificar
        r2 = requests.get(f"{HA_URL}/api/states/input_number.zona_north", headers=headers, timeout=5)
        if r2.status_code == 200:
            print(f"✅ OK (valor actualizado: {r2.json()['state']})")
        else:
            print("⚠️  Escrito pero no verificado")
    else:
        print(f"❌ Error {r.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("Prueba completada")
print("=" * 60)
