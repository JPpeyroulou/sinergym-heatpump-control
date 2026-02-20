#!/usr/bin/env python3
"""
Versión mock del script de prueba para verificar funcionalidad sin Home Assistant real
Simula las respuestas de Home Assistant para probar la lógica del script
"""

import sys
import json
from unittest.mock import Mock, patch
from test_homeassistant_integration import (
    test_connection, test_sensors, test_actuators_read, 
    test_actuators_write, test_full_cycle, print_header, Colors
)

# Simular respuestas de Home Assistant
MOCK_RESPONSES = {
    # API root
    "/api/": {
        "status": 200,
        "json": {"message": "API running."}
    },
    # Sensores (todos existen)
    "/api/states/sensor.north_heating_rate_sensor": {
        "status": 200,
        "json": {"state": "0.5", "entity_id": "sensor.north_heating_rate_sensor"}
    },
    "/api/states/sensor.south_heating_rate_sensor": {
        "status": 200,
        "json": {"state": "0.3", "entity_id": "sensor.south_heating_rate_sensor"}
    },
    "/api/states/sensor.outdoor_temperature_sensor": {
        "status": 200,
        "json": {"state": "22.5", "entity_id": "sensor.outdoor_temperature_sensor"}
    },
    # Actuadores (todos existen)
    "/api/states/input_number.zona_north": {
        "status": 200,
        "json": {"state": "0", "entity_id": "input_number.zona_north"}
    },
    "/api/states/input_number.zona_south": {
        "status": 200,
        "json": {"state": "0", "entity_id": "input_number.zona_south"}
    },
    "/api/states/input_number.zona_east": {
        "status": 200,
        "json": {"state": "0", "entity_id": "input_number.zona_east"}
    },
    "/api/states/input_number.zona_west": {
        "status": 200,
        "json": {"state": "0", "entity_id": "input_number.zona_west"}
    },
    "/api/states/input_number.temperatura_calefaccion": {
        "status": 200,
        "json": {"state": "22.5", "entity_id": "input_number.temperatura_calefaccion"}
    },
    # Servicios (escritura exitosa)
    "/api/services/input_number/set_value": {
        "status": 200,
        "json": [{"id": "mock_id", "type": "result", "success": True}]
    },
}

def mock_get(url, headers=None, timeout=None, **kwargs):
    """Mock de requests.get"""
    response = Mock()
    path = url.split("?")[0].split("//")[-1].split("/", 1)[-1] if "//" in url else url
    
    # Buscar respuesta mock
    for mock_path, mock_data in MOCK_RESPONSES.items():
        if mock_path in path or path.endswith(mock_path.split("/")[-1]):
            response.status_code = mock_data["status"]
            response.json.return_value = mock_data["json"]
            response.text = json.dumps(mock_data["json"])
            return response
    
    # Si no se encuentra, devolver 404
    response.status_code = 404
    response.json.return_value = {"message": "Not found"}
    response.text = "Not found"
    return response

def mock_post(url, headers=None, json_data=None, timeout=None, **kwargs):
    """Mock de requests.post"""
    response = Mock()
    path = url.split("?")[0].split("//")[-1].split("/", 1)[-1] if "//" in url else url
    
    # Buscar respuesta mock
    for mock_path, mock_data in MOCK_RESPONSES.items():
        if mock_path in path:
            response.status_code = mock_data["status"]
            response.json.return_value = mock_data["json"]
            response.text = json.dumps(mock_data["json"])
            return response
    
    # Si no se encuentra, devolver 200 (asumir éxito)
    response.status_code = 200
    response.json.return_value = [{"id": "mock_id", "type": "result", "success": True}]
    response.text = json.dumps(response.json.return_value)
    return response

def main():
    print_header("PRUEBA MOCK - SIMULACIÓN DE HOME ASSISTANT")
    print(f"{Colors.BLUE}ℹ️  Esta es una prueba simulada sin Home Assistant real{Colors.RESET}")
    print(f"{Colors.BLUE}ℹ️  Todas las respuestas están mockeadas{Colors.RESET}\n")
    
    ha_url = "http://localhost:8123"
    token = "mock_token_for_testing"
    
    # Mockear requests
    with patch('requests.get', side_effect=mock_get), \
         patch('requests.post', side_effect=mock_post):
        
        # Ejecutar todas las pruebas
        print_header("1. PRUEBA DE CONEXIÓN (MOCK)")
        test_connection(ha_url, token)
        
        print_header("2. PRUEBA DE SENSORES (MOCK)")
        sensor_results, sensor_found = test_sensors(ha_url, token)
        
        print_header("3. PRUEBA DE ACTUADORES - LECTURA (MOCK)")
        actuator_results, actuator_found = test_actuators_read(ha_url, token)
        
        print_header("4. PRUEBA DE ACTUADORES - ESCRITURA (MOCK)")
        write_success = test_actuators_write(ha_url, token, actuator_results)
        
        print_header("5. PRUEBA DE CICLO COMPLETO (MOCK)")
        test_full_cycle(ha_url, token)
        
        # Resumen
        print_header("RESUMEN FINAL (MOCK)")
        print(f"{Colors.GREEN}✅ Prueba mock completada exitosamente{Colors.RESET}")
        print(f"{Colors.BLUE}ℹ️  Para probar con Home Assistant real, ejecuta:{Colors.RESET}")
        print(f"   python3 test_homeassistant_integration.py --url http://localhost:8123 --token TU_TOKEN")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
