# Guía de Configuración de Home Assistant para Sinergym

Esta guía te ayudará a configurar Home Assistant para integrarlo con el sistema de control RL.

## 📋 Requisitos Previos

1. Home Assistant corriendo en Docker
2. Acceso a la interfaz web de Home Assistant (normalmente `http://localhost:8123`)
3. Permisos para crear entidades y tokens de acceso

---

## 🔑 Paso 1: Crear Token de Acceso

1. Abre Home Assistant en tu navegador
2. Ve a **Perfil** (icono de usuario en la esquina inferior izquierda)
3. Desplázate hasta **Tokens de acceso** (Long-Lived Access Tokens)
4. Haz clic en **CREAR TOKEN**
5. Dale un nombre descriptivo: `Sinergym RL Control`
6. **Copia el token** (solo se muestra una vez)
7. Pégalo en `production_config.yaml` en la sección `api_config.homeassistant.token`

---

## 📊 Paso 2: Crear Sensores (Variables de Observación)

Necesitas crear sensores en Home Assistant que correspondan a las 15 variables + 1 meter del modelo.

### Opción A: Usar `input_number` como sensores (para pruebas)

Agrega esto a tu `configuration.yaml` de Home Assistant:

```yaml
# Sensores para variables de observación
input_number:
  # Heating rate variables (4)
  north_heating_rate:
    name: "North Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
  
  south_heating_rate:
    name: "South Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
  
  east_heating_rate:
    name: "East Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
  
  west_heating_rate:
    name: "West Heating Rate"
    min: 0
    max: 1
    step: 0.01
    unit_of_measurement: ""
  
  # Outdoor variables (2)
  outdoor_temperature:
    name: "Outdoor Temperature"
    min: -10
    max: 50
    step: 0.1
    unit_of_measurement: "°C"
    initial: 20
  
  outdoor_humidity:
    name: "Outdoor Humidity"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 50
  
  # Air temperature variables (4)
  north_air_temperature:
    name: "North Air Temperature"
    min: 10
    max: 35
    step: 0.1
    unit_of_measurement: "°C"
    initial: 22
  
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
  
  # Air humidity variables (4)
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
  
  # Heat pump power (1)
  heat_pump_power:
    name: "Heat Pump Power"
    min: 0
    max: 20000
    step: 10
    unit_of_measurement: "W"
    initial: 1000
  
  # Meter (1)
  total_electricity_hvac:
    name: "Total Electricity HVAC"
    min: 0
    max: 100000
    step: 1
    unit_of_measurement: "Wh"
    initial: 0

# Crear sensores template que expongan estos valores
template:
  - sensor:
      - name: "North Heating Rate Sensor"
        state: "{{ states('input_number.north_heating_rate') | float }}"
        unit_of_measurement: ""
      
      - name: "South Heating Rate Sensor"
        state: "{{ states('input_number.south_heating_rate') | float }}"
        unit_of_measurement: ""
      
      - name: "East Heating Rate Sensor"
        state: "{{ states('input_number.east_heating_rate') | float }}"
        unit_of_measurement: ""
      
      - name: "West Heating Rate Sensor"
        state: "{{ states('input_number.west_heating_rate') | float }}"
        unit_of_measurement: ""
      
      - name: "Outdoor Temperature Sensor"
        state: "{{ states('input_number.outdoor_temperature') | float }}"
        unit_of_measurement: "°C"
      
      - name: "Outdoor Humidity Sensor"
        state: "{{ states('input_number.outdoor_humidity') | float }}"
        unit_of_measurement: "%"
      
      - name: "North Air Temperature Sensor"
        state: "{{ states('input_number.north_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      - name: "South Air Temperature Sensor"
        state: "{{ states('input_number.south_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      - name: "East Air Temperature Sensor"
        state: "{{ states('input_number.east_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      - name: "West Air Temperature Sensor"
        state: "{{ states('input_number.west_air_temperature') | float }}"
        unit_of_measurement: "°C"
      
      - name: "North Air Humidity Sensor"
        state: "{{ states('input_number.north_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      - name: "South Air Humidity Sensor"
        state: "{{ states('input_number.south_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      - name: "East Air Humidity Sensor"
        state: "{{ states('input_number.east_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      - name: "West Air Humidity Sensor"
        state: "{{ states('input_number.west_air_humidity') | float }}"
        unit_of_measurement: "%"
      
      - name: "Heat Pump Power Sensor"
        state: "{{ states('input_number.heat_pump_power') | float }}"
        unit_of_measurement: "W"
      
      - name: "Total Electricity HVAC Sensor"
        state: "{{ states('input_number.total_electricity_hvac') | float }}"
        unit_of_measurement: "Wh"
```

**Nota:** Si prefieres usar sensores reales (por ejemplo, sensores de temperatura/humedad físicos), simplemente reemplaza los `entity_id` en `production_config.yaml` con los IDs de tus sensores reales.

---

## 🎛️ Paso 3: Crear Actuadores (Entidades de Control)

Necesitas crear 5 actuadores que correspondan a las acciones del modelo:

```yaml
# Actuadores para control (5 actuadores)
input_number:
  # Zonas (4 actuadores - valores 0 o 1)
  zona_north:
    name: "Zona North Control"
    min: 0
    max: 1
    step: 1
    unit_of_measurement: ""
    initial: 0
  
  zona_south:
    name: "Zona South Control"
    min: 0
    max: 1
    step: 1
    unit_of_measurement: ""
    initial: 0
  
  zona_east:
    name: "Zona East Control"
    min: 0
    max: 1
    step: 1
    unit_of_measurement: ""
    initial: 0
  
  zona_west:
    name: "Zona West Control"
    min: 0
    max: 1
    step: 1
    unit_of_measurement: ""
    initial: 0
  
  # Temperatura de calefacción (1 actuador - rango 15-45°C)
  temperatura_calefaccion:
    name: "Temperatura Calefacción"
    min: 15
    max: 45
    step: 0.5
    unit_of_measurement: "°C"
    initial: 25
```

---

## 🔄 Paso 4: Automatización (Opcional pero Recomendado)

Puedes crear automatizaciones que actualicen los sensores basándose en los actuadores, o conectar los actuadores a dispositivos reales.

### Ejemplo: Actualizar sensores basándose en actuadores

```yaml
automation:
  - alias: "Update Heating Rates from Controls"
    trigger:
      - platform: state
        entity_id:
          - input_number.zona_north
          - input_number.zona_south
          - input_number.zona_east
          - input_number.zona_west
    action:
      - service: input_number.set_value
        data:
          entity_id: input_number.north_heating_rate
          value: "{{ states('input_number.zona_north') | float }}"
      - service: input_number.set_value
        data:
          entity_id: input_number.south_heating_rate
          value: "{{ states('input_number.zona_south') | float }}"
      - service: input_number.set_value
        data:
          entity_id: input_number.east_heating_rate
          value: "{{ states('input_number.zona_east') | float }}"
      - service: input_number.set_value
        data:
          entity_id: input_number.west_heating_rate
          value: "{{ states('input_number.zona_west') | float }}"
```

---

## ✅ Paso 5: Verificar Configuración

1. **Reinicia Home Assistant** después de agregar la configuración:
   ```bash
   # En el contenedor Docker
   docker restart homeassistant
   # O desde la interfaz: Configuración > Sistema > Reiniciar
   ```

2. **Verifica que las entidades existan:**
   - Ve a **Configuración > Dispositivos y servicios > Entidades**
   - Busca las entidades creadas (deberías ver `input_number.zona_north`, `sensor.north_heating_rate_sensor`, etc.)

3. **Verifica los entity_id exactos:**
   - Haz clic en cada entidad y copia el **Entity ID** exacto
   - Asegúrate de que coincidan con los de `production_config.yaml`

---

## 🔧 Paso 6: Mapeo de Entity IDs

Asegúrate de que los `entity_id` en `production_config.yaml` coincidan con los creados en Home Assistant.

**Ejemplo de mapeo correcto:**

```yaml
# En production_config.yaml
sensor_entities:
  lozaradiante_zonanorth_heating_rate: "sensor.north_heating_rate_sensor"
  lozaradiante_zonasouth_heating_rate: "sensor.south_heating_rate_sensor"
  # ... etc

actuator_entities:
  ZonaNorth: "input_number.zona_north"
  ZonaSouth: "input_number.zona_south"
  # ... etc
```

**Nota:** Los entity_id son case-sensitive y deben coincidir exactamente.

---

## 🧪 Paso 7: Probar la Conexión

Puedes probar la conexión desde Python:

```python
import requests

ha_url = "http://localhost:8123"
ha_token = "TU_TOKEN_AQUI"

headers = {
    "Authorization": f"Bearer {ha_token}",
    "Content-Type": "application/json"
}

# Leer un sensor
response = requests.get(
    f"{ha_url}/api/states/sensor.north_heating_rate_sensor",
    headers=headers
)
print("Sensor:", response.json())

# Escribir un actuador
response = requests.post(
    f"{ha_url}/api/services/input_number/set_value",
    headers=headers,
    json={
        "entity_id": "input_number.zona_north",
        "value": 1.0
    }
)
print("Actuador:", response.status_code)
```

---

## 📝 Resumen de Entity IDs Necesarios

### Sensores (16 entidades):
1. `sensor.north_heating_rate_sensor`
2. `sensor.south_heating_rate_sensor`
3. `sensor.east_heating_rate_sensor`
4. `sensor.west_heating_rate_sensor`
5. `sensor.outdoor_temperature_sensor`
6. `sensor.outdoor_humidity_sensor`
7. `sensor.north_air_temperature_sensor`
8. `sensor.south_air_temperature_sensor`
9. `sensor.east_air_temperature_sensor`
10. `sensor.west_air_temperature_sensor`
11. `sensor.north_air_humidity_sensor`
12. `sensor.south_air_humidity_sensor`
13. `sensor.east_air_humidity_sensor`
14. `sensor.west_air_humidity_sensor`
15. `sensor.heat_pump_power_sensor`
16. `sensor.total_electricity_hvac_sensor`

### Actuadores (5 entidades):
1. `input_number.zona_north`
2. `input_number.zona_south`
3. `input_number.zona_east`
4. `input_number.zona_west`
5. `input_number.temperatura_calefaccion`

---

## 🚀 Siguiente Paso

Una vez configurado Home Assistant, actualiza `production_config.yaml` con:
- El token de acceso
- Los entity_id correctos (si difieren de los del ejemplo)
- La URL correcta (si Home Assistant no está en `localhost:8123`)

Luego ejecuta:
```bash
python ./train_online_production.py --config ./production_config.yaml --model ./model.zip
```

---

## 💡 Tips

1. **Para desarrollo/pruebas:** Usa `input_number` como en el ejemplo
2. **Para producción:** Conecta sensores reales y reemplaza los entity_id
3. **Monitoreo:** Crea un dashboard en Home Assistant para ver los valores en tiempo real
4. **Logs:** Revisa los logs de Home Assistant si hay problemas de conexión
