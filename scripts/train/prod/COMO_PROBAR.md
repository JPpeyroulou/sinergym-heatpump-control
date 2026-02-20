# 🧪 Cómo Probar la Integración con Home Assistant

## 📋 Prerequisitos

1. Home Assistant corriendo y accesible
2. Token de acceso creado (Long-Lived Access Token)
3. Python 3 con `requests` instalado

---

## 🚀 Método 1: Script de Prueba Completo (Recomendado)

### Paso 1: Instalar dependencias (si es necesario)

```bash
pip install requests
```

### Paso 2: Ejecutar el script de prueba

```bash
cd /workspaces/sinergym/scripts/train/prod
python3 test_homeassistant_integration.py --url http://localhost:8123 --token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0Mzg5YzYxZWJkZmU0NDA4YjdhNTljMzYwZTQzNjkyYSIsImlhdCI6MTc2OTU1NjQ3OCwiZXhwIjoyMDg0OTE2NDc4fQ.Z_KjATkzbD8CwSzXhzDTKl_-EwDRTAU58DBjv-uglyk
```

### Paso 3: Interpretar los resultados

El script ejecuta 5 pruebas:

1. **Prueba de Conexión**: Verifica que puedes conectarte a Home Assistant
2. **Prueba de Sensores (Lectura)**: Verifica que todos los sensores existen y se pueden leer
3. **Prueba de Actuadores (Lectura)**: Verifica que todos los actuadores existen
4. **Prueba de Actuadores (Escritura)**: Prueba escribir valores en los actuadores
5. **Prueba de Ciclo Completo**: Simula un paso completo del sistema RL

**Salida esperada:**
```
✅ Conexión establecida
✅ Todos los sensores encontrados (16/16)
✅ Todos los actuadores encontrados (5/5)
✅ Escritura exitosa (5/5)
✅ Ciclo completo ejecutado
```

---

## 🔍 Método 2: Prueba Manual con Python

### Script rápido de prueba

Crea un archivo `test_quick.py`:

```python
import requests

ha_url = "http://localhost:8123"
token = "TU_TOKEN_AQUI"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# 1. Probar conexión
print("1. Probando conexión...")
response = requests.get(f"{ha_url}/api/", headers=headers)
print(f"   Status: {response.status_code}")
print(f"   Mensaje: {response.json().get('message', 'N/A')}")

# 2. Leer un sensor
print("\n2. Leyendo sensor...")
response = requests.get(
    f"{ha_url}/api/states/sensor.north_heating_rate_sensor",
    headers=headers
)
if response.status_code == 200:
    print(f"   ✅ Sensor encontrado: {response.json()['state']}")
else:
    print(f"   ❌ Error: {response.status_code}")

# 3. Leer un actuador
print("\n3. Leyendo actuador...")
response = requests.get(
    f"{ha_url}/api/states/input_number.zona_north",
    headers=headers
)
if response.status_code == 200:
    print(f"   ✅ Actuador encontrado: {response.json()['state']}")
else:
    print(f"   ❌ Error: {response.status_code}")

# 4. Escribir en actuador
print("\n4. Escribiendo en actuador...")
response = requests.post(
    f"{ha_url}/api/services/input_number/set_value",
    headers=headers,
    json={
        "entity_id": "input_number.zona_north",
        "value": 50.0
    }
)
if response.status_code in [200, 201]:
    print(f"   ✅ Escritura exitosa")
    
    # Verificar que se escribió
    response = requests.get(
        f"{ha_url}/api/states/input_number.zona_north",
        headers=headers
    )
    if response.status_code == 200:
        print(f"   ✅ Valor actual: {response.json()['state']}")
else:
    print(f"   ❌ Error: {response.status_code}")
```

Ejecuta:
```bash
python3 test_quick.py
```

---

## 🌐 Método 3: Prueba desde la Interfaz Web de Home Assistant

### Verificar que las entidades existen:

1. Abre Home Assistant en tu navegador
2. Ve a **Configuración > Dispositivos y servicios > Entidades**
3. Busca:
   - `input_number.zona_north`
   - `input_number.zona_south`
   - `input_number.zona_east`
   - `input_number.zona_west`
   - `input_number.temperatura_calefaccion`
   - `sensor.north_heating_rate_sensor`
   - etc.

### Probar escritura manualmente:

1. Ve a **Configuración > Dispositivos y servicios > Helpers**
2. Busca "Zona North Control"
3. Haz clic y cambia el valor
4. Verifica que el cambio se refleje

---

## 🔧 Método 4: Usar el Bridge Directamente

### Prueba con el bridge de Home Assistant

Crea un script `test_bridge.py`:

```python
import sys
sys.path.append('/workspaces/sinergym/scripts/train/prod')

from homeassistant_bridge import HomeAssistantBridge
import numpy as np

# Configuración
ha_url = "http://localhost:8123"
ha_token = "TU_TOKEN_AQUI"

sensor_entities = {
    "lozaradiante_zonanorth_heating_rate": "sensor.north_heating_rate_sensor",
    "outdoor_temperature": "sensor.outdoor_temperature_sensor",
    # ... agrega todos los sensores
}

actuator_entities = {
    "ZonaNorth": "input_number.zona_north",
    "ZonaSouth": "input_number.zona_south",
    "ZonaEast": "input_number.zona_east",
    "ZonaWest": "input_number.zona_west",
    "Temperatura_Calefaccion_Schedule": "input_number.temperatura_calefaccion",
}

# Crear bridge
bridge = HomeAssistantBridge(
    ha_url=ha_url,
    ha_token=ha_token,
    sensor_entities=sensor_entities,
    actuator_entities=actuator_entities,
    action_space_low=np.array([0, 0, 0, 0, 15]),
    action_space_high=np.array([100, 100, 100, 100, 30]),
    time_variables=['month', 'day_of_month', 'hour']
)

# Probar reset
print("Probando reset...")
obs, info = bridge.reset()
print(f"Observación: {obs}")
print(f"Info: {info}")

# Probar step
print("\nProbando step...")
action = np.array([50.0, 30.0, 70.0, 20.0, 22.5])  # Acción de prueba
obs, info = bridge.step(action)
print(f"Observación: {obs}")
print(f"Info: {info}")

bridge.close()
```

Ejecuta:
```bash
python3 test_bridge.py
```

---

## 📊 Interpretación de Resultados

### ✅ Todo funciona correctamente:
- Conexión exitosa
- Todos los sensores encontrados (16/16)
- Todos los actuadores encontrados (5/5)
- Escritura exitosa
- **→ El sistema está listo para usar**

### ⚠️ Sensores faltantes:
- Algunos sensores no se encuentran
- **Solución**: Verifica que los sensores template estén creados correctamente
- Los sensores template se pueden crear dinámicamente vía API si es necesario

### ❌ Actuadores faltantes:
- Los `input_number` no se encuentran (404)
- **Solución**: 
  1. Verifica que el package esté en `/config/packages/sinergym.yaml`
  2. Verifica que `configuration.yaml` incluya `packages: !include_dir_named packages/`
  3. **Reinicia Home Assistant**

### ❌ Error de conexión:
- No se puede conectar a Home Assistant
- **Solución**: 
  1. Verifica que Home Assistant esté corriendo
  2. Verifica la URL (debe ser `http://localhost:8123` o la IP correcta)
  3. Verifica que el token sea válido

---

## 🐛 Troubleshooting

### Error: "401 Unauthorized"
- El token es inválido o expiró
- **Solución**: Crea un nuevo token en Home Assistant

### Error: "404 Not Found" para actuadores
- Los `input_number` no existen
- **Solución**: Reinicia Home Assistant después de agregar el package

### Error: "Connection refused"
- Home Assistant no está corriendo o la URL es incorrecta
- **Solución**: Verifica que Home Assistant esté activo y la URL sea correcta

### Los sensores existen pero devuelven "unknown"
- Los sensores template no tienen valores iniciales
- **Solución**: Los `input_number` base deben tener valores iniciales

---

## ✅ Checklist de Prueba

Antes de ejecutar el sistema completo, verifica:

- [ ] Conexión con Home Assistant funciona
- [ ] Todos los sensores existen y devuelven valores
- [ ] Todos los actuadores existen
- [ ] Se puede escribir en los actuadores
- [ ] Los valores escritos se reflejan correctamente
- [ ] El bridge puede hacer reset() sin errores
- [ ] El bridge puede hacer step() sin errores

---

## 🚀 Próximo Paso

Una vez que todas las pruebas pasen, puedes ejecutar el sistema completo:

```bash
python ./train_online_production.py --config ./production_config.yaml --model ./model.zip
```

El sistema usará automáticamente el bridge de Home Assistant para leer sensores y escribir actuadores.
