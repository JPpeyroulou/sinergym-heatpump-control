# Prompt para Cursor - Agregar Sensores Faltantes en Home Assistant

## Problema

El sistema de control RL está funcionando correctamente con Home Assistant, pero faltan 12 sensores de los 16 esperados. Los actuadores (5/5) funcionan perfectamente, pero los sensores faltantes hacen que el sistema use valores por defecto (0.0) en lugar de datos reales.

## Estado Actual

### ✅ Funcionando (4 sensores):
- `sensor.north_heating_rate_sensor` ✅
- `sensor.outdoor_temperature_sensor` ✅
- `sensor.north_air_temperature_sensor` ✅
- `sensor.heat_pump_power_sensor` ✅

### ❌ Faltantes (12 sensores):
1. `sensor.south_heating_rate_sensor`
2. `sensor.east_heating_rate_sensor`
3. `sensor.west_heating_rate_sensor`
4. `sensor.outdoor_humidity_sensor`
5. `sensor.south_air_temperature_sensor`
6. `sensor.east_air_temperature_sensor`
7. `sensor.west_air_temperature_sensor`
8. `sensor.north_air_humidity_sensor`
9. `sensor.south_air_humidity_sensor`
10. `sensor.east_air_humidity_sensor`
11. `sensor.west_air_humidity_sensor`
12. `sensor.total_electricity_hvac_sensor`

## Lo que Necesito

Agregar los 12 sensores template faltantes al archivo `/config/packages/sinergym.yaml` (o donde esté tu configuración de Home Assistant).

## Configuración Requerida

Agrega estos sensores template a tu archivo de configuración de Home Assistant. Deben estar en la sección `template:` dentro de `sensor:`.

### Sensores a Agregar:

```yaml
template:
  - sensor:
      # ===== SENSORES FALTANTES =====
      
      # South Heating Rate Sensor
      - name: "South Heating Rate Sensor"
        unique_id: "south_heating_rate_sensor"
        state: "{{ states('input_number.south_heating_rate') | float }}"
        unit_of_measurement: ""
      
      # East Heating Rate Sensor
      - name: "East Heating Rate Sensor"
        unique_id: "east_heating_rate_sensor"
        state: "{{ states('input_number.east_heating_rate') | float }}"
        unit_of_measurement: ""
      
      # West Heating Rate Sensor
      - name: "West Heating Rate Sensor"
        unique_id: "west_heating_rate_sensor"
        state: "{{ states('input_number.west_heating_rate') | float }}"
        unit_of_measurement: ""
      
      # Outdoor Humidity Sensor
      - name: "Outdoor Humidity Sensor"
        unique_id: "outdoor_humidity_sensor"
        state: "{{ states('input_number.outdoor_humidity') | float }}"
        unit_of_measurement: "%"
      
      # South Air Temperature Sensor
      - name: "South Air Temperature Sensor"
        unique_id: "south_air_temperature_sensor"
        state: "{{ states('input_number.south_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      # East Air Temperature Sensor
      - name: "East Air Temperature Sensor"
        unique_id: "east_air_temperature_sensor"
        state: "{{ states('input_number.east_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      # West Air Temperature Sensor
      - name: "West Air Temperature Sensor"
        unique_id: "west_air_temperature_sensor"
        state: "{{ states('input_number.west_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      # North Air Humidity Sensor
      - name: "North Air Humidity Sensor"
        unique_id: "north_air_humidity_sensor"
        state: "{{ states('input_number.north_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      # South Air Humidity Sensor
      - name: "South Air Humidity Sensor"
        unique_id: "south_air_humidity_sensor"
        state: "{{ states('input_number.south_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      # East Air Humidity Sensor
      - name: "East Air Humidity Sensor"
        unique_id: "east_air_humidity_sensor"
        state: "{{ states('input_number.east_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      # West Air Humidity Sensor
      - name: "West Air Humidity Sensor"
        unique_id: "west_air_humidity_sensor"
        state: "{{ states('input_number.west_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      # Total Electricity HVAC Sensor
      - name: "Total Electricity HVAC Sensor"
        unique_id: "total_electricity_hvac_sensor"
        state: "{{ states('input_number.total_electricity_hvac') | float }}"
        unit_of_measurement: "Wh"
```

## Verificación de Input Numbers Base

Asegúrate de que estos `input_number` existan en tu configuración (deben estar en la sección `input_number:`):

```yaml
input_number:
  # ... otros input_number existentes ...
  
  # Verificar que estos existan:
  south_heating_rate:
    name: "South Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
    initial: 0
  
  east_heating_rate:
    name: "East Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
    initial: 0
  
  west_heating_rate:
    name: "West Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
    initial: 0
  
  outdoor_humidity:
    name: "Outdoor Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  south_air_temperature:
    name: "South Air Temperature"
    min: 10
    max: 35
    step: 0.1
    unit_of_measurement: "°C"
    initial: 22
  
  east_air_temperature:
    name: "East Air Temperature"
    min: 10
    max: 35
    step: 0.1
    unit_of_measurement: "°C"
    initial: 22
  
  west_air_temperature:
    name: "West Air Temperature"
    min: 10
    max: 35
    step: 0.1
    unit_of_measurement: "°C"
    initial: 22
  
  north_air_humidity:
    name: "North Air Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  south_air_humidity:
    name: "South Air Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  east_air_humidity:
    name: "East Air Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  west_air_humidity:
    name: "West Air Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  total_electricity_hvac:
    name: "Total Electricity HVAC"
    min: 0
    max: 100000
    step: 1
    unit_of_measurement: "Wh"
    initial: 0
```

## Instrucciones

1. **Abre** el archivo `/config/packages/sinergym.yaml` (o donde tengas la configuración)

2. **Verifica** que la sección `template:` exista. Si no existe, créala.

3. **Agrega** los 12 sensores template faltantes a la lista de sensores dentro de `template: - sensor:`

4. **Verifica** que los `input_number` base existan. Si faltan, agrégalos también.

5. **Valida** la sintaxis YAML:
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('/config/packages/sinergym.yaml'))"
   ```

6. **Reinicia** Home Assistant:
   - Desde la interfaz: Configuración > Sistema > Reiniciar
   - O desde terminal: `docker restart homeassistant`

7. **Verifica** que los sensores se hayan creado:
   - Ve a Configuración > Dispositivos y servicios > Entidades
   - Busca los sensores por nombre o `unique_id`

## Estructura Esperada del Archivo

Tu archivo `/config/packages/sinergym.yaml` debería tener esta estructura:

```yaml
# Actuadores (ya existen y funcionan)
input_number:
  zona_north: ...
  zona_south: ...
  zona_east: ...
  zona_west: ...
  temperatura_calefaccion: ...
  
  # Input numbers para sensores (verificar que existan)
  north_heating_rate: ...
  south_heating_rate: ...  # ← Verificar
  east_heating_rate: ...   # ← Verificar
  west_heating_rate: ...   # ← Verificar
  outdoor_temperature: ...
  outdoor_humidity: ...    # ← Verificar
  north_air_temperature: ...
  south_air_temperature: ... # ← Verificar
  east_air_temperature: ...  # ← Verificar
  west_air_temperature: ...  # ← Verificar
  north_air_humidity: ...    # ← Verificar
  south_air_humidity: ...    # ← Verificar
  east_air_humidity: ...     # ← Verificar
  west_air_humidity: ...     # ← Verificar
  heat_pump_power: ...
  total_electricity_hvac: ... # ← Verificar

# Sensores template
template:
  - sensor:
      # Sensores existentes (4)
      - name: "North Heating Rate Sensor"
        unique_id: "north_heating_rate_sensor"
        ...
      
      # AGREGAR AQUÍ LOS 12 SENSORES FALTANTES
      - name: "South Heating Rate Sensor"
        unique_id: "south_heating_rate_sensor"
        ...
      # ... etc
```

## Notas Importantes

1. **Los `unique_id` deben ser exactos** - El sistema RL los busca por estos IDs
2. **Los `input_number` base deben existir** - Los sensores template leen de ellos
3. **La indentación YAML debe ser correcta** - 2 espacios, no tabs
4. **Reinicio necesario** - Los sensores template se crean al reiniciar Home Assistant

## Verificación Post-Cambio

Después de agregar los sensores y reiniciar, puedes verificar con:

```python
import requests

ha_url = "http://localhost:8123"
token = "TU_TOKEN"

headers = {"Authorization": f"Bearer {token}"}

sensores_faltantes = [
    "sensor.south_heating_rate_sensor",
    "sensor.east_heating_rate_sensor",
    "sensor.west_heating_rate_sensor",
    "sensor.outdoor_humidity_sensor",
    "sensor.south_air_temperature_sensor",
    "sensor.east_air_temperature_sensor",
    "sensor.west_air_temperature_sensor",
    "sensor.north_air_humidity_sensor",
    "sensor.south_air_humidity_sensor",
    "sensor.east_air_humidity_sensor",
    "sensor.west_air_humidity_sensor",
    "sensor.total_electricity_hvac_sensor",
]

for sensor in sensores_faltantes:
    r = requests.get(f"{ha_url}/api/states/{sensor}", headers=headers)
    if r.status_code == 200:
        print(f"✅ {sensor}")
    else:
        print(f"❌ {sensor} - {r.status_code}")
```

## Resultado Esperado

Después de completar estos cambios:
- ✅ 16/16 sensores encontrados
- ✅ 5/5 actuadores funcionando
- ✅ Sistema RL funcionando con datos completos

---

**Por favor, agrega los 12 sensores template faltantes al archivo de configuración de Home Assistant y reinicia el servicio.**
