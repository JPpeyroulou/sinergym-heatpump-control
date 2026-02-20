#!/usr/bin/env python3
"""
Script para crear input_number en Home Assistant usando la API REST
Útil cuando los packages no funcionan o no quieres reiniciar HA

USO:
    python3 create_input_numbers.py --url http://localhost:8123 --token YOUR_TOKEN
"""

import argparse
import requests
import json
import sys

# Definición de los 5 actuadores según tus especificaciones
ACTUATORS = {
    "zona_north": {
        "name": "Zona North Control",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit_of_measurement": "%",
        "initial": 0,
        "icon": "mdi:radiator"
    },
    "zona_south": {
        "name": "Zona South Control",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit_of_measurement": "%",
        "initial": 0,
        "icon": "mdi:radiator"
    },
    "zona_east": {
        "name": "Zona East Control",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit_of_measurement": "%",
        "initial": 0,
        "icon": "mdi:radiator"
    },
    "zona_west": {
        "name": "Zona West Control",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit_of_measurement": "%",
        "initial": 0,
        "icon": "mdi:radiator"
    },
    "temperatura_calefaccion": {
        "name": "Temperatura Calefacción",
        "min": 15,
        "max": 30,
        "step": 0.5,
        "unit_of_measurement": "°C",
        "initial": 22.5,
        "icon": "mdi:thermometer"
    }
}


def create_input_number(ha_url, token, entity_id, config):
    """Crea un input_number usando la API de Home Assistant."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Home Assistant no tiene un endpoint directo para crear input_number
    # Necesitamos usar el servicio input_number.set_value después de que exista
    # O mejor: usar el servicio input_number.reload y luego crear manualmente
    
    # Alternativa: Usar el servicio de configuración (si está disponible)
    # O mejor aún: usar el servicio helper.create
    
    # NOTA: Los input_number NO se pueden crear dinámicamente sin reinicio
    # Este script solo puede verificar si existen y configurarlos
    
    # Verificar si existe
    response = requests.get(
        f"{ha_url}/api/states/input_number.{entity_id}",
        headers=headers,
        timeout=5
    )
    
    if response.status_code == 200:
        print(f"✅ {entity_id} ya existe")
        return True
    elif response.status_code == 404:
        print(f"❌ {entity_id} NO existe - requiere reinicio de HA")
        print(f"   Configuración necesaria:")
        print(f"   input_number:")
        print(f"     {entity_id}:")
        for key, value in config.items():
            print(f"       {key}: {value}")
        return False
    else:
        print(f"⚠️  Error verificando {entity_id}: {response.status_code}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verifica o crea input_number en Home Assistant"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8123",
        help="URL de Home Assistant"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Token de acceso de Home Assistant"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Genera archivo de configuración YAML"
    )
    
    args = parser.parse_args()
    
    ha_url = args.url.rstrip('/')
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json"
    }
    
    print("=" * 60)
    print("VERIFICACIÓN DE INPUT_NUMBER EN HOME ASSISTANT")
    print("=" * 60)
    print(f"URL: {ha_url}")
    print()
    
    # Verificar conexión
    try:
        response = requests.get(
            f"{ha_url}/api/",
            headers=headers,
            timeout=5
        )
        if response.status_code != 200:
            print(f"❌ Error conectando a Home Assistant: {response.status_code}")
            sys.exit(1)
        print("✅ Conexión con Home Assistant establecida")
        print()
    except Exception as e:
        print(f"❌ Error conectando a Home Assistant: {e}")
        sys.exit(1)
    
    # Verificar cada actuador
    results = {}
    for entity_id, config in ACTUATORS.items():
        results[entity_id] = create_input_number(ha_url, args.token, entity_id, config)
        print()
    
    # Resumen
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    existing = sum(1 for v in results.values() if v)
    missing = len(results) - existing
    
    print(f"✅ Existentes: {existing}/{len(results)}")
    print(f"❌ Faltantes: {missing}/{len(results)}")
    print()
    
    if missing > 0:
        print("⚠️  SOLUCIÓN:")
        print("   Los input_number NO se pueden crear sin reiniciar Home Assistant.")
        print("   Opciones:")
        print()
        print("   1. REINICIAR HOME ASSISTANT (recomendado):")
        print("      - Agrega la configuración a /config/packages/sinergym.yaml")
        print("      - Reinicia Home Assistant")
        print("      - Verifica que las entidades existan")
        print()
        print("   2. USAR ARCHIVO DE CONFIGURACIÓN GENERADO:")
        if args.create_config:
            config_file = "sinergym_actuators.yaml"
            with open(config_file, 'w') as f:
                f.write("input_number:\n")
                for entity_id, config in ACTUATORS.items():
                    f.write(f"  {entity_id}:\n")
                    for key, value in config.items():
                        if isinstance(value, str) and ' ' in value:
                            f.write(f"    {key}: \"{value}\"\n")
                        else:
                            f.write(f"    {key}: {value}\n")
            print(f"      ✅ Archivo generado: {config_file}")
        else:
            print("      Ejecuta con --create-config para generar el archivo")
        print()
        print("   3. VERIFICAR ESTRUCTURA DEL PACKAGE:")
        print("      - El archivo debe estar en /config/packages/sinergym.yaml")
        print("      - configuration.yaml debe tener: homeassistant: packages: !include_dir_named packages/")
        print("      - La indentación YAML debe ser correcta (2 espacios)")
    else:
        print("✅ Todos los input_number existen correctamente!")
    
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
