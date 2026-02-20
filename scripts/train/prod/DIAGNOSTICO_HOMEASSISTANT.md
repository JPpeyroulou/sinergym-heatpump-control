# 🔍 Diagnóstico y Solución: Input_Number no se cargan en Home Assistant

## ❌ Problema Identificado

Los `input_number` no se están cargando desde el package aunque la configuración YAML parece correcta.

## 🔎 Causas Comunes

### 1. **Los `input_number` requieren reinicio**
   - ⚠️ **IMPORTANTE**: Los `input_number` NO se pueden crear dinámicamente sin reiniciar Home Assistant
   - A diferencia de los sensores template que se pueden crear vía API, los `input_number` son helpers que requieren reinicio

### 2. **Estructura incorrecta del package**
   - El package debe tener la estructura correcta
   - La indentación YAML debe ser exacta (2 espacios, no tabs)

### 3. **Problemas con `configuration.yaml`**
   - Debe incluir: `homeassistant: packages: !include_dir_named packages/`
   - El directorio `packages/` debe existir

### 4. **Errores de sintaxis YAML**
   - Comas faltantes
   - Indentación incorrecta
   - Valores sin comillas cuando son necesarias

---

## ✅ Solución 1: Verificar y Corregir el Package

### Paso 1: Verificar estructura del package

El archivo `/config/packages/sinergym.yaml` debe tener esta estructura:

```yaml
input_number:
  zona_north:
    name: "Zona North Control"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 0
  
  zona_south:
    name: "Zona South Control"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 0
  
  zona_east:
    name: "Zona East Control"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 0
  
  zona_west:
    name: "Zona West Control"
    min: 0
    max: 100
    step: 1
    unit_of_measurement: "%"
    initial: 0
  
  temperatura_calefaccion:
    name: "Temperatura Calefacción"
    min: 15
    max: 30
    step: 0.5
    unit_of_measurement: "°C"
    initial: 22.5
```

### Paso 2: Verificar `configuration.yaml`

Debe incluir:

```yaml
homeassistant:
  packages: !include_dir_named packages/
```

**NOTA**: Si ya tienes otras configuraciones en `homeassistant:`, agrega `packages:` dentro de esa sección:

```yaml
homeassistant:
  # ... otras configuraciones ...
  packages: !include_dir_named packages/
```

### Paso 3: Verificar sintaxis YAML

```bash
# En el contenedor Docker o sistema donde corre HA
python3 -c "import yaml; yaml.safe_load(open('/config/packages/sinergym.yaml'))"
```

Si no hay errores, la sintaxis es correcta.

### Paso 4: **REINICIAR Home Assistant**

```bash
# Opción 1: Desde la interfaz web
# Configuración > Sistema > Reiniciar

# Opción 2: Desde Docker
docker restart homeassistant

# Opción 3: Desde el sistema
systemctl restart homeassistant
```

### Paso 5: Verificar logs

Después del reinicio, revisa los logs:

```bash
# Ver logs de Home Assistant
docker logs homeassistant | grep -i "input_number\|sinergym\|error"
```

Busca mensajes como:
- ✅ `Loaded input_number from packages`
- ❌ `Error loading package sinergym`
- ❌ `Invalid config for input_number`

---

## ✅ Solución 2: Usar Script de Diagnóstico

Ejecuta el script de diagnóstico:

```bash
cd /workspaces/sinergym/scripts/train/prod
python3 create_input_numbers.py --url http://localhost:8123 --token TU_TOKEN --create-config
```

Este script:
1. Verifica qué `input_number` existen
2. Muestra cuáles faltan
3. Genera un archivo de configuración correcto

---

## ✅ Solución 3: Crear Manualmente (Alternativa Temporal)

Si no puedes reiniciar Home Assistant ahora, puedes usar la interfaz web:

1. Ve a **Configuración > Dispositivos y servicios > Helpers**
2. Haz clic en **+ CREAR HELPER**
3. Selecciona **Número**
4. Configura cada uno:
   - **Nombre**: "Zona North Control"
   - **Mínimo**: 0
   - **Máximo**: 100
   - **Unidad**: "%"
   - **ID de entidad**: `zona_north` (importante que coincida exactamente)

Repite para los 5 actuadores.

---

## 🔍 Verificación Post-Reinicio

### Verificar que las entidades existen:

```python
import requests

ha_url = "http://localhost:8123"
token = "TU_TOKEN"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

actuators = [
    "input_number.zona_north",
    "input_number.zona_south",
    "input_number.zona_east",
    "input_number.zona_west",
    "input_number.temperatura_calefaccion"
]

for entity_id in actuators:
    response = requests.get(
        f"{ha_url}/api/states/{entity_id}",
        headers=headers
    )
    if response.status_code == 200:
        print(f"✅ {entity_id}: {response.json()['state']}")
    else:
        print(f"❌ {entity_id}: No encontrado ({response.status_code})")
```

### Probar escritura:

```python
# Probar escribir un valor
response = requests.post(
    f"{ha_url}/api/services/input_number/set_value",
    headers=headers,
    json={
        "entity_id": "input_number.zona_north",
        "value": 50
    }
)
print(f"Resultado: {response.status_code}")
```

---

## 🐛 Troubleshooting

### Error: "Referenced entities input_number.zona_north are missing"

**Causa**: El `input_number` no existe o no se cargó correctamente.

**Solución**:
1. Verifica que el package esté en `/config/packages/sinergym.yaml`
2. Verifica que `configuration.yaml` incluya `packages: !include_dir_named packages/`
3. **Reinicia Home Assistant**
4. Verifica los logs para errores de sintaxis

### Error: "Invalid config for input_number"

**Causa**: Error de sintaxis en el YAML.

**Solución**:
1. Verifica la indentación (2 espacios, no tabs)
2. Verifica que todos los valores estén correctamente formateados
3. Usa un validador YAML online

### Los sensores template funcionan pero los input_number no

**Causa**: Los sensores template se pueden crear dinámicamente, los `input_number` no.

**Solución**: **Reinicia Home Assistant** después de agregar los `input_number` al package.

---

## 📝 Checklist Final

- [ ] El archivo `/config/packages/sinergym.yaml` existe
- [ ] El archivo tiene sintaxis YAML válida
- [ ] `configuration.yaml` incluye `packages: !include_dir_named packages/`
- [ ] Home Assistant se reinició después de agregar el package
- [ ] Los logs no muestran errores relacionados con `input_number`
- [ ] Las entidades aparecen en **Configuración > Dispositivos y servicios > Entidades**
- [ ] El script de diagnóstico confirma que existen

---

## 💡 Nota Importante

**Los `input_number` NO se pueden crear sin reiniciar Home Assistant**. Esto es una limitación de Home Assistant, no un bug. Si necesitas crear entidades dinámicamente, considera usar:

- **Sensores template** (se pueden crear dinámicamente)
- **Integraciones personalizadas** (más complejo)
- **API de configuración** (requiere reinicio de todas formas)

Para este caso, **el reinicio es necesario y es la solución correcta**.
