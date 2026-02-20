#!/usr/bin/env python3
"""
Script para encontrar la URL correcta de Home Assistant
Prueba diferentes URLs comunes cuando se ejecuta desde Docker
"""
import requests
import sys

HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0Mzg5YzYxZWJkZmU0NDA4YjdhNTljMzYwZTQzNjkyYSIsImlhdCI6MTc2OTU1NjQ3OCwiZXhwIjoyMDg0OTE2NDc4fQ.Z_KjATkzbD8CwSzXhzDTKl_-EwDRTAU58DBjv-uglyk"

# URLs comunes a probar
URLS_TO_TRY = [
    "http://localhost:8123",
    "http://127.0.0.1:8123",
    "http://host.docker.internal:8123",  # Docker Desktop
    "http://172.17.0.1:8123",  # Docker bridge network
    "http://homeassistant:8123",  # Si está en la misma red Docker
    "http://home-assistant:8123",
    "http://ha:8123",
]

def test_url(url):
    """Prueba si Home Assistant responde en esta URL"""
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    try:
        response = requests.get(f"{url}/api/", headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return True, data.get('message', 'OK')
        return False, f"Status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

print("=" * 70)
print("BUSCANDO HOME ASSISTANT")
print("=" * 70)
print(f"Token: {'*' * 20}...{HA_TOKEN[-4:]}")
print()
print("Probando diferentes URLs...")
print()

found = False
for url in URLS_TO_TRY:
    print(f"Probando {url:40s}...", end=" ")
    success, message = test_url(url)
    if success:
        print(f"✅ ENCONTRADO!")
        print(f"   Mensaje: {message}")
        print()
        print("=" * 70)
        print("URL CORRECTA ENCONTRADA")
        print("=" * 70)
        print(f"Usa esta URL en el script de prueba:")
        print(f"  --url {url}")
        print()
        print("O ejecuta directamente:")
        print(f"  python3 test_homeassistant_integration.py --url {url} --token TU_TOKEN")
        found = True
        break
    else:
        print(f"❌ {message}")

print()
if not found:
    print("=" * 70)
    print("NO SE ENCONTRÓ HOME ASSISTANT")
    print("=" * 70)
    print()
    print("Opciones:")
    print("1. Verifica que Home Assistant esté corriendo")
    print("2. Si Home Assistant está en el host (no en Docker):")
    print("   - En Linux: usa la IP del host (ej: http://192.168.1.100:8123)")
    print("   - En Docker Desktop: usa http://host.docker.internal:8123")
    print("3. Si Home Assistant está en otro contenedor:")
    print("   - Usa el nombre del contenedor (ej: http://homeassistant:8123)")
    print("   - O la IP del contenedor")
    print()
    print("Para encontrar la IP del host en Linux:")
    print("  ip addr show | grep 'inet ' | grep -v '127.0.0.1'")
    print()
    print("Para encontrar contenedores Docker:")
    print("  docker ps | grep homeassistant")
    sys.exit(1)

sys.exit(0)
