# Para qué sirven los wrappers del YAML

Los **wrappers** son capas que se aplican al entorno: transforman la **observación** que ve el modelo (y a veces la recompensa) antes de usarla. En `nuestroMultrizona.yaml` tienes:

```yaml
wrappers:
  - DatetimeWrapper
  - NormalizeObservation
  - ForecastWrapper: { forecast_csv: ..., horizon: 24 }
  - EnergyCostWrapper
```

El orden importa: se aplican de arriba a abajo (entorno base → Datetime → Normalize → Forecast → EnergyCost → modelo).

---

## 1. **DatetimeWrapper**

**Qué hace:** Cambia cómo se codifica el **tiempo** en la observación.

| Antes (env base) | Después (lo que ve el modelo) |
|------------------|-------------------------------|
| `month` (1–12) | `month_cos`, `month_sin` (codificación circular) |
| `day_of_month` (1–31) | `is_weekend` (0 o 1) |
| `hour` (0–23) | `hour_cos`, `hour_sin` (codificación circular) |

**Por qué:**  
- Hora y mes son **cíclicos** (23h y 0h están cerca). Con seno/coseno el modelo entiende mejor que 23 y 0 son vecinos.  
- `is_weekend` resume si es fin de semana, útil para patrones de uso.

**Efecto en dimensiones:** La observación base tiene 3 variables de tiempo; después del wrapper sigues teniendo 3 “slots”, pero con 2 valores para month y 2 para hour → en total **+2 dimensiones** (de 19 pasas a 21 antes de los otros wrappers).

---

## 2. **NormalizeObservation**

**Qué hace:** **Normaliza** la observación (resta media, divide por desviación típica) para que las magnitudes sean parecidas y el entrenamiento sea más estable.

- Puede **calibrar** media y varianza automáticamente mientras interactúas con el entorno.  
- Opcionalmente puede cargar/salvar `mean.txt`, `var.txt`, `count.txt` por episodio.

**Por qué:** Temperaturas (15–25), humedades (30–70), potencias (0–5000) tienen rangos muy distintos; normalizar ayuda a que la red no se vaya a unas variables y descuide otras.

**Efecto en dimensiones:** No cambia el tamaño del vector, solo los valores.

---

## 3. **ForecastWrapper**

**Qué hace:** Añade a la observación un **vector de predicción meteorológica** (próximas `horizon` filas de un CSV: temp, humedad, viento, radiación, etc.).

- En tu YAML: `horizon: 24` y un CSV con 6 columnas → **24×6 = 144** números extra.

**Por qué:** El modelo puede usar “cómo va a estar el tiempo en las próximas X horas” para decidir mejor (precalentar, ahorrar, etc.).

**Efecto en dimensiones:** Añade `horizon × número_de_columnas_del_CSV` (en tu caso 144).

---

## 4. **EnergyCostWrapper**

**Qué hace:**  
- Añade a la observación **una variable más**: el **precio/costo de la energía** en el instante actual (leyendo un CSV de precios por fecha/hora).  
- Sustituye la recompensa por **EnergyCostLinearReward**, que combina confort, consumo y **costo** de la energía.

**Por qué:** Así el modelo puede considerar no solo “cuánta energía uso” sino “a qué precio” (horarios caros vs baratos).

**Efecto en dimensiones:** Añade **1** dimensión (`energy_cost`).  
**Requisito:** Necesita `energy_cost_data_path` (CSV con columnas tipo datetime y valor de precio). Si en el YAML solo pones `- EnergyCostWrapper` sin parámetros, **no hay path por defecto**: el wrapper exige ese argumento y fallará al crear el entorno si no lo indicás.

### Dónde se configura el path (de dónde agarra el archivo)

El path **no está hardcodeado en el código**: lo tiene que recibir el wrapper al crearse.

| Dónde | Cómo se usa el path |
|-------|----------------------|
| **YAML del entorno** (ej. `nuestroMultrizona.yaml`) | Si el entorno se crea con `create_environment(..., wrappers=wrappers)` y los wrappers se leen de ese YAML, el path tiene que estar en la entrada de `EnergyCostWrapper`, p. ej. `EnergyCostWrapper: { energy_cost_data_path: "ruta/al/archivo.csv" }`. Si solo ponés `- EnergyCostWrapper` sin parámetros, **el path no se define** y el wrapper fallará. |
| **Config de producción** (`production_config.yaml`) | Los wrappers se definen en `wrappers:` como lista con `name` y `params`. Si usás EnergyCostWrapper ahí, tenés que añadir en `params` algo como `energy_cost_data_path: "/ruta/al/archivo.csv"`. Hoy ese config **no** incluye EnergyCostWrapper. |
| **Código** (sin YAML) | Al instanciar: `EnergyCostWrapper(env, energy_cost_data_path="/ruta/al/archivo.csv", ...)`. |

**Ruta típica del CSV en el repo:**  
`sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv`  
(o la ruta absoluta según donde corras: `/workspaces/sinergym/sinergym/data/energy_cost/...`).

---

## Resumen rápido

| Wrapper             | Para qué sirve                         | Cambio en la observación                          |
|---------------------|----------------------------------------|---------------------------------------------------|
| **DatetimeWrapper** | Codificar hora/mes de forma cíclica y fin de semana | month/day/hour → is_weekend, hour_sin/cos, month_sin/cos (+2 dims) |
| **NormalizeObservation** | Estabilizar entrenamiento            | Misma dimensión; valores normalizados             |
| **ForecastWrapper** | Dar predicción meteorológica al modelo | + horizon×columnas (ej. +144)                     |
| **EnergyCostWrapper** | Incluir precio de la energía         | +1 (energy_cost) y recompensa con costo           |

Si entrenaste con este YAML, el **modelo** espera la observación **ya transformada** por todos estos wrappers (en el mismo orden). Para correr solo `model.zip` fuera de Sinergym tendrías que replicar tú: datetime → normalizar → añadir forecast → añadir energy_cost (o usar el mismo pipeline de wrappers si tienes el env disponible).

---

## Cómo se usa EnergyCostWrapper

### 1. Formato del CSV de precios

El wrapper espera un CSV con **separador `;`** (punto y coma) y al menos estas columnas:

- **`datetime`**: fecha y hora (se usa para Month, Day, Hour).
- **`value`**: precio de la energía en ese instante (ej. €/MWh o céntimos/kWh).

Ejemplo (como `sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv`):

```csv
id;name;geoid;geoname;value;datetime
1001;...;8741;Península;41.45;2023-01-01T00:00:00+01:00
1001;...;8741;Península;43.01;2023-01-01T01:00:00+01:00
...
```

El código lee el CSV, deriva `Month`, `Day`, `Hour` de `datetime` y en cada paso busca la fila que coincida con `info['month']`, `info['day']`, `info['hour']` y añade esa fila’s `value` como dimensión extra en la observación.

### 2. Cómo configurarlo en el YAML

**EnergyCostWrapper requiere** el argumento `energy_cost_data_path`. Si en el YAML solo pones `- EnergyCostWrapper` sin parámetros, fallará al crear el entorno.

Tienes que pasarlo como un wrapper con argumentos. Formato típico cuando los wrappers vienen de la config (p. ej. `train_agent_local_conf` o `create_environment`):

**Opción A – Lista de wrappers (cada uno es un dict de nombre → params):**

```yaml
wrappers:
  - DatetimeWrapper: {}
  - NormalizeObservation: {}
  - ForecastWrapper:
      forecast_csv: /workspaces/sinergym/sinergym/data/weather/forecast/forecast_24h.csv
      horizon: 24
  - EnergyCostWrapper:
      energy_cost_data_path: /workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv
      reward_kwargs:
        temperature_variables: [air_temperature]
        energy_variables: [HVAC_electricity_demand_rate]
        energy_cost_variables: [energy_cost]
        range_comfort_winter: [20.0, 23.5]
        range_comfort_summer: [23.0, 26.0]
        temperature_weight: 0.4
        energy_weight: 0.4
        lambda_energy: 1.0e-4
        lambda_temperature: 1.0
        lambda_energy_cost: 1.0
```

**Opción B – Solo EnergyCostWrapper con ruta mínima:**

```yaml
  - EnergyCostWrapper:
      energy_cost_data_path: /workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv
```

El resto de `reward_kwargs` usan valores por defecto si no los pones.

### 3. Uso en código (sin YAML)

Si creas el entorno a mano y aplicas wrappers tú mismo:

```python
from sinergym.utils.wrappers import EnergyCostWrapper

env = gym.make("Eplus-...")
env = EnergyCostWrapper(
    env,
    energy_cost_data_path="/ruta/al/archivo_precios.csv",
    reward_kwargs={
        "temperature_variables": ["air_temperature"],
        "energy_variables": ["HVAC_electricity_demand_rate"],
        "energy_cost_variables": ["energy_cost"],
        "range_comfort_winter": [20.0, 23.5],
        "range_comfort_summer": [23.0, 26.0],
        "temperature_weight": 0.4,
        "energy_weight": 0.4,
        "lambda_energy": 1e-4,
        "lambda_temperature": 1.0,
        "lambda_energy_cost": 1.0,
    },
)
```

### 4. Resumen

| Qué | Cómo |
|-----|------|
| **CSV** | Separador `;`, columnas `datetime` y `value`. |
| **Ruta** | Obligatoria: `energy_cost_data_path` en el YAML o en el constructor. |
| **Observación** | Se añade 1 número por paso: el precio correspondiente a (month, day, hour). |
| **Recompensa** | Pasa a usar `EnergyCostLinearReward` (confort + energía + costo). |

Si en `nuestroMultrizona.yaml` tienes solo `- EnergyCostWrapper` sin parámetros, hay que **añadir al menos** `energy_cost_data_path` (y opcionalmente `reward_kwargs`) como en los ejemplos de arriba para que funcione.

---

## Cómo toma los valores de precio ahora (dentro del entrenamiento)

1. **Al iniciar** (reset): el wrapper lee el CSV (`energy_cost_data_path`), parsea la columna `datetime`, le suma 1 hora (`DateOffset(hours=1)`), y guarda en memoria una tabla con columnas `Month`, `Day`, `Hour`, `value`.
2. **En cada paso**: usa `info['month']`, `info['day']`, `info['hour']` del entorno, busca la fila de esa tabla que coincida con ese (month, day, hour) y toma el `value` de esa fila.
3. Ese **único número** se concatena al final del vector de observación: `obs = [..., value]`.

Es decir: el precio lo toma **solo del CSV**, en función del tiempo simulado (month, day, hour) que viene en `info`.

---

## Cómo ingresar el precio si ejecutás el modelo por fuera del entrenamiento

Si corrés solo el modelo (por ejemplo con `run_model_with_forecast.py` o tu propio script) **sin** el entorno de Sinergym, la observación que le pasás a `model.predict(obs)` debe tener **las mismas dimensiones y el mismo orden** que en entrenamiento. Si entrenaste con EnergyCostWrapper, la última dimensión es el **energy_cost**.

Tenés que:

1. **Calcular (month, day, hour)** del paso actual (igual que en simulación: puede ser tiempo real o un contador de pasos convertido a fecha/hora).
2. **Obtener el valor de precio** para esa (month, day, hour) desde el **mismo CSV** (o una tabla/API equivalente).
3. **Añadir ese valor al final** del vector de observación antes de llamar a `model.predict(obs)`.

Ejemplo usando la utilidad `energy_cost_utils` (en `scripts/train/prod/`):

```python
import numpy as np
from stable_baselines3 import SAC

# Ajustar rutas
MODEL_PATH = "model.zip"
ENERGY_COST_CSV = "/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv"

# Cargar utilidad (mismo directorio que este script o en path)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_cost_utils import get_energy_cost_value  # o load_energy_cost_table + get_energy_cost_from_table

model = SAC.load(MODEL_PATH, device="cpu")

# Observación que ya tenés (base + forecast, etc.) — debe tener 163 dims si usaste nuestroMultrizona con Forecast 24 y sin EnergyCost; con EnergyCost serían 164
obs_sin_precio = np.zeros(163, dtype=np.float32)  # rellenar con tus sensores + forecast

month, day, hour = 1, 15, 10  # ejemplo: 15 enero, 10:00
energy_cost = get_energy_cost_value(month, day, hour, ENERGY_COST_CSV)

obs_completa = np.concatenate([obs_sin_precio, [energy_cost]], axis=0)  # 164 dims
action, _ = model.predict(obs_completa, deterministic=True)
```

Si no usás CSV y tenés el precio por otro medio (API, otro archivo):

- Calculá **un solo float** por paso (el precio en esa hora).
- Hacé `obs_completa = np.concatenate([obs_sin_precio, [precio_hora]], axis=0)` y pasá `obs_completa` al modelo.

Resumen: **dentro del entrenamiento** los valores los toma el wrapper desde el CSV usando `info['month'/'day'/'hour']`. **Por fuera**, tenés que obtener vos ese valor (mismo CSV o misma lógica) y **añadirlo como última dimensión** de la observación.
