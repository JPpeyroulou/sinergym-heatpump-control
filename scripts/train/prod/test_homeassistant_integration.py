#!/usr/bin/env python3
"""
Script de prueba completo para la integración con Home Assistant
Verifica conexión, sensores, actuadores y funcionalidad completa

USO:
    python3 test_homeassistant_integration.py --url http://localhost:8123 --token YOUR_TOKEN
"""

import argparse
import requests
import time
import sys
from typing import Dict, List, Tuple, Optional

# Configuración esperada según production_config.yaml
EXPECTED_SENSORS = {
    "lozaradiante_zonanorth_heating_rate": "sensor.north_heating_rate_sensor",
    "lozaradiante_zonasouth_heating_rate": "sensor.south_heating_rate_sensor",
    "lozaradiante_zonaeast_heating_rate": "sensor.east_heating_rate_sensor",
    "lozaradiante_zonawest_heating_rate": "sensor.west_heating_rate_sensor",
    "outdoor_temperature": "sensor.outdoor_temperature_sensor",
    "outdoor_humidity": "sensor.outdoor_humidity_sensor",
    "north_perimeter_air_temperature": "sensor.north_air_temperature_sensor",
    "south_perimeter_air_temperature": "sensor.south_air_temperature_sensor",
    "east_perimeter_air_temperature": "sensor.east_air_temperature_sensor",
    "west_perimeter_air_temperature": "sensor.west_air_temperature_sensor",
    "north_perimeter_air_humidity": "sensor.north_air_humidity_sensor",
    "south_perimeter_air_humidity": "sensor.south_air_humidity_sensor",
    "east_perimeter_air_humidity": "sensor.east_air_humidity_sensor",
    "west_perimeter_air_humidity": "sensor.west_air_humidity_sensor",
    "heat_pump_power": "sensor.heat_pump_power_sensor",
    "total_electricity_HVAC": "sensor.total_electricity_hvac_sensor",
}

EXPECTED_ACTUATORS = {
    "ZonaNorth": "input_number.zona_north",
    "ZonaSouth": "input_number.zona_south",
    "ZonaEast": "input_number.zona_east",
    "ZonaWest": "input_number.zona_west",
    "Temperatura_Calefaccion_Schedule": "input_number.temperatura_calefaccion",
}


class Colors:
    """Colores para la salida en terminal"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


def test_connection(ha_url: str, token: str) -> bool:
    """Prueba la conexión con Home Assistant"""
    print_header("1. PRUEBA DE CONEXIÓN")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"{ha_url}/api/",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Conexión establecida con Home Assistant")
            print_info(f"Versión: {data.get('message', 'N/A')}")
            return True
        else:
            print_error(f"Error de conexión: Status {response.status_code}")
            print_info(f"Respuesta: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error(f"No se pudo conectar a {ha_url}")
        print_info("Verifica que Home Assistant esté corriendo y la URL sea correcta")
        return False
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        return False


def test_sensors(ha_url: str, token: str) -> Tuple[Dict[str, bool], int]:
    """Prueba la lectura de sensores"""
    print_header("2. PRUEBA DE SENSORES (LECTURA)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    results = {}
    found = 0
    
    for var_name, entity_id in EXPECTED_SENSORS.items():
        try:
            response = requests.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                state = data.get('state', 'unknown')
                results[var_name] = True
                found += 1
                print_success(f"{entity_id:50s} = {state}")
            elif response.status_code == 404:
                results[var_name] = False
                print_error(f"{entity_id:50s} NO ENCONTRADO (404)")
            else:
                results[var_name] = False
                print_error(f"{entity_id:50s} Error: {response.status_code}")
                
        except Exception as e:
            results[var_name] = False
            print_error(f"{entity_id:50s} Excepción: {e}")
    
    print(f"\n{Colors.BOLD}Resumen sensores: {found}/{len(EXPECTED_SENSORS)} encontrados{Colors.RESET}")
    return results, found


def test_actuators_read(ha_url: str, token: str) -> Tuple[Dict[str, bool], int]:
    """Prueba la lectura de actuadores"""
    print_header("3. PRUEBA DE ACTUADORES (LECTURA)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    results = {}
    found = 0
    
    for action_name, entity_id in EXPECTED_ACTUATORS.items():
        try:
            response = requests.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                state = data.get('state', 'unknown')
                results[action_name] = True
                found += 1
                print_success(f"{entity_id:50s} = {state}")
            elif response.status_code == 404:
                results[action_name] = False
                print_error(f"{entity_id:50s} NO ENCONTRADO (404)")
                print_warning("  ⚠️  Este actuador requiere reinicio de Home Assistant")
            else:
                results[action_name] = False
                print_error(f"{entity_id:50s} Error: {response.status_code}")
                
        except Exception as e:
            results[action_name] = False
            print_error(f"{entity_id:50s} Excepción: {e}")
    
    print(f"\n{Colors.BOLD}Resumen actuadores: {found}/{len(EXPECTED_ACTUATORS)} encontrados{Colors.RESET}")
    return results, found


def test_actuators_write(ha_url: str, token: str, actuator_results: Dict[str, bool]) -> int:
    """Prueba la escritura en actuadores"""
    print_header("4. PRUEBA DE ACTUADORES (ESCRITURA)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Valores de prueba
    test_values = {
        "ZonaNorth": 50.0,      # 50%
        "ZonaSouth": 75.0,      # 75%
        "ZonaEast": 25.0,       # 25%
        "ZonaWest": 0.0,        # 0%
        "Temperatura_Calefaccion_Schedule": 22.5,  # 22.5°C
    }
    
    success_count = 0
    
    for action_name, entity_id in EXPECTED_ACTUATORS.items():
        if not actuator_results.get(action_name, False):
            print_warning(f"{entity_id:50s} Omitido (no existe)")
            continue
        
        test_value = test_values.get(action_name, 0.0)
        
        try:
            # Determinar el dominio
            domain = entity_id.split('.')[0]
            
            # Escribir valor
            response = requests.post(
                f"{ha_url}/api/services/{domain}/set_value",
                headers=headers,
                json={
                    "entity_id": entity_id,
                    "value": test_value
                },
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                # Verificar que se escribió correctamente
                time.sleep(0.5)  # Esperar un poco
                verify_response = requests.get(
                    f"{ha_url}/api/states/{entity_id}",
                    headers=headers,
                    timeout=5
                )
                
                if verify_response.status_code == 200:
                    actual_value = float(verify_response.json().get('state', 0))
                    if abs(actual_value - test_value) < 0.1:
                        print_success(f"{entity_id:50s} = {test_value} ✅")
                        success_count += 1
                    else:
                        print_warning(f"{entity_id:50s} = {actual_value} (esperado {test_value})")
                else:
                    print_warning(f"{entity_id:50s} Escrito pero no verificado")
            else:
                print_error(f"{entity_id:50s} Error al escribir: {response.status_code}")
                if response.text:
                    print_info(f"  Respuesta: {response.text[:100]}")
                    
        except Exception as e:
            print_error(f"{entity_id:50s} Excepción: {e}")
    
    print(f"\n{Colors.BOLD}Resumen escritura: {success_count}/{len(EXPECTED_ACTUATORS)} exitosos{Colors.RESET}")
    return success_count


def test_full_cycle(ha_url: str, token: str) -> bool:
    """Prueba un ciclo completo: leer sensores, escribir actuadores, leer de nuevo"""
    print_header("5. PRUEBA DE CICLO COMPLETO")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print_info("Simulando un paso completo del sistema RL...")
    print()
    
    # 1. Leer sensores iniciales
    print("📥 Leyendo sensores iniciales...")
    sensor_values = {}
    for var_name, entity_id in list(EXPECTED_SENSORS.items())[:5]:  # Solo primeros 5 para no saturar
        try:
            response = requests.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                value = float(response.json().get('state', 0))
                sensor_values[var_name] = value
                print(f"   {var_name:40s} = {value}")
        except:
            pass
    
    print()
    
    # 2. Escribir acciones
    print("📤 Escribiendo acciones...")
    actions = {
        "input_number.zona_north": 60.0,
        "input_number.zona_south": 40.0,
        "input_number.zona_east": 80.0,
        "input_number.zona_west": 20.0,
        "input_number.temperatura_calefaccion": 24.0,
    }
    
    for entity_id, value in actions.items():
        try:
            domain = entity_id.split('.')[0]
            response = requests.post(
                f"{ha_url}/api/services/{domain}/set_value",
                headers=headers,
                json={"entity_id": entity_id, "value": value},
                timeout=5
            )
            if response.status_code in [200, 201]:
                print(f"   {entity_id:40s} = {value} ✅")
            else:
                print(f"   {entity_id:40s} Error: {response.status_code}")
        except Exception as e:
            print(f"   {entity_id:40s} Excepción: {e}")
    
    print()
    print_info("Esperando 2 segundos para que Home Assistant procese...")
    time.sleep(2)
    print()
    
    # 3. Leer sensores de nuevo
    print("📥 Leyendo sensores después de acciones...")
    for var_name, entity_id in list(EXPECTED_SENSORS.items())[:5]:
        try:
            response = requests.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                value = float(response.json().get('state', 0))
                old_value = sensor_values.get(var_name, 0)
                change = "📈" if value != old_value else "➡️"
                print(f"   {var_name:40s} = {value} {change}")
        except:
            pass
    
    print()
    print_success("Ciclo completo ejecutado")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prueba completa de integración con Home Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Prueba básica
  python3 test_homeassistant_integration.py --url http://localhost:8123 --token YOUR_TOKEN
  
  # Prueba con URL personalizada
  python3 test_homeassistant_integration.py --url http://192.168.1.100:8123 --token YOUR_TOKEN
  
  # Solo verificar conexión y lectura
  python3 test_homeassistant_integration.py --url http://localhost:8123 --token YOUR_TOKEN --skip-write
        """
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8123",
        help="URL de Home Assistant (default: http://localhost:8123)"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Token de acceso de Home Assistant (Long-Lived Access Token)"
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Saltar pruebas de escritura (solo lectura)"
    )
    parser.add_argument(
        "--skip-cycle",
        action="store_true",
        help="Saltar prueba de ciclo completo"
    )
    
    args = parser.parse_args()
    
    ha_url = args.url.rstrip('/')
    
    print_header("PRUEBA DE INTEGRACIÓN HOME ASSISTANT - SINERGYM")
    print_info(f"URL: {ha_url}")
    print_info(f"Token: {'*' * 20}...{args.token[-4:]}")
    print()
    
    # 1. Prueba de conexión
    if not test_connection(ha_url, args.token):
        print_error("No se pudo establecer conexión. Abortando.")
        sys.exit(1)
    
    # 2. Prueba de sensores
    sensor_results, sensor_found = test_sensors(ha_url, args.token)
    
    # 3. Prueba de actuadores (lectura)
    actuator_results, actuator_found = test_actuators_read(ha_url, args.token)
    
    # 4. Prueba de actuadores (escritura)
    if not args.skip_write:
        write_success = test_actuators_write(ha_url, args.token, actuator_results)
    else:
        print_warning("Prueba de escritura omitida (--skip-write)")
        write_success = 0
    
    # 5. Prueba de ciclo completo
    if not args.skip_cycle and actuator_found > 0:
        test_full_cycle(ha_url, args.token)
    
    # Resumen final
    print_header("RESUMEN FINAL")
    
    total_sensors = len(EXPECTED_SENSORS)
    total_actuators = len(EXPECTED_ACTUATORS)
    
    print(f"{Colors.BOLD}Sensores:{Colors.RESET}")
    print(f"  ✅ Encontrados: {sensor_found}/{total_sensors}")
    if sensor_found < total_sensors:
        print_warning(f"  ⚠️  Faltantes: {total_sensors - sensor_found}")
    
    print(f"\n{Colors.BOLD}Actuadores:{Colors.RESET}")
    print(f"  ✅ Encontrados: {actuator_found}/{total_actuators}")
    if actuator_found < total_actuators:
        print_error(f"  ❌ Faltantes: {total_actuators - actuator_found}")
        print_warning("  ⚠️  Los actuadores faltantes requieren reinicio de Home Assistant")
    
    if not args.skip_write:
        print(f"\n{Colors.BOLD}Escritura:{Colors.RESET}")
        print(f"  ✅ Exitosos: {write_success}/{actuator_found}")
    
    # Estado general
    print()
    if sensor_found == total_sensors and actuator_found == total_actuators:
        print_success("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print_info("El sistema está listo para usar con Sinergym")
        return 0
    elif actuator_found == 0:
        print_error("❌ NINGÚN ACTUADOR ENCONTRADO")
        print_warning("⚠️  Necesitas reiniciar Home Assistant después de agregar el package")
        print_info("📝 Verifica:")
        print_info("   1. El archivo /config/packages/sinergym.yaml existe")
        print_info("   2. configuration.yaml incluye: packages: !include_dir_named packages/")
        print_info("   3. Home Assistant se reinició después de agregar el package")
        return 1
    else:
        print_warning("⚠️  ALGUNAS PRUEBAS FALLARON")
        print_info("Revisa los errores arriba y corrige la configuración")
        return 1


if __name__ == "__main__":
    sys.exit(main())
