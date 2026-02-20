# Comparación de archivos: Sinergym (este repo) vs sinergym:arm64 (RPi)

## 1. Estructura y archivos que solo existen en uno

### Solo en este repo (workspace)

| Ruta | Descripción |
|------|-------------|
| `scripts/train/prod/` | **Carpeta completa**: homeassistant_bridge.py, pyenv_production.py, train_online_production.py, production_config.yaml, register_production_envs.py, model.zip, etc. |
| `sinergym/data/default_configuration/nuestroMultrizona.yaml` | Configuración multizona (tu edificio) |
| `sinergym/data/default_configuration/nuestro.yaml` | Configuración nuestro |
| `sinergym/data/default_configuration/2ZoneDataCenterHVAC_wEconomizer.yaml` | Un solo YAML (en RPi hay dos: _DX y _CW) |
| `sinergym/version.txt` | Versión 3.9.1 (en RPi la versión está solo en pyproject.toml) |
| `pepe2/`, `sinergym/utils/common_prod.py` | Código adicional del proyecto |

### Solo en la imagen sinergym:arm64 (RPi)

| Ruta | Descripción |
|------|-------------|
| `sinergym/data/default_configuration/2ZoneDataCenterHVAC_wEconomizer_DX.yaml` | Variante DX |
| `sinergym/data/default_configuration/2ZoneDataCenterHVAC_wEconomizer_CW.yaml` | Variante CW |
| (no existe `scripts/train/prod/`) | No hay scripts de producción ni Home Assistant |
| (no existe `nuestroMultrizona.yaml` ni `nuestro.yaml`) | No hay configuraciones “nuestro” |

---

## 2. Diferencias de contenido en archivos comunes

### 2.1 `pyproject.toml`

| Aspecto | Este repo | sinergym:arm64 (RPi) |
|---------|-----------|----------------------|
| **Formato** | `[tool.poetry]` (Poetry) | `[project]` (PEP 621 / pip estándar) |
| **Versión** | **3.9.1** | **3.11.0** |
| **Dependencias** | poetry: gymnasium ^1.0.0, numpy ^2.2.0, pandas ^2.2.2, eppy ^0.5.63, etc. | pip: gymnasium>=1.2.0, numpy>=2.3.2, pandas>=2.3.1, eppy>=0.5.63, etc. |
| **Extras** | pytest, pytest-cov, stable-baselines3, wandb, etc. (opcionales Poetry) | wandb, test, drl, notebooks, gcloud, plots, extras (optional-dependencies) |
| **stable-baselines3** | ^2.4.0 | >=2.7.0 (+ sb3-contrib>=2.7.0) |

La imagen de la RPi es una versión **más nueva** del proyecto (3.11.0) y usa formato estándar PEP 621 en lugar de Poetry.

### 2.2 `scripts/try_env.py`

| Aspecto | Este repo | sinergym:arm64 (RPi) |
|---------|-----------|----------------------|
| **Wrappers** | NormalizeAction, NormalizeObservation, LoggerWrapper, CSVLogger | **DatetimeWrapper**, NormalizeAction, NormalizeObservation, LoggerWrapper, CSVLogger |
| **Bucle del episodio** | Acumula `rewards`, log cada mes con `info['month']`, al final log “Mean reward” y “Cumulative Reward” | No acumula rewards en lista; al final solo “Episode finished” y `env.get_obs_dict(obs)` |
| **Imports** | Sin DatetimeWrapper | Con `DatetimeWrapper` |
| **Mensaje final** | `Episode {} - Mean reward: {} - Cumulative Reward: {}` | `Episode {} finished.` + log del diccionario de observación |

En la RPi el script es más simple (sin recompensa por episodio en log) y usa **DatetimeWrapper**.

### 2.3 Configuraciones por defecto (`sinergym/data/default_configuration/`)

| Archivo | Este repo | RPi |
|---------|-----------|-----|
| 1ZoneDataCenterCRAC_wApproachTemp.yaml | Sí | Sí |
| 2ZoneDataCenterHVAC_wEconomizer.yaml | Sí (uno) | No (en su lugar: _DX.yaml y _CW.yaml) |
| 2ZoneDataCenterHVAC_wEconomizer_DX.yaml | No | Sí |
| 2ZoneDataCenterHVAC_wEconomizer_CW.yaml | No | Sí |
| 5ZoneAutoDXVAV.yaml | Sí | Sí |
| ASHRAE901_OfficeMedium_STD2019_Denver.yaml | Sí | Sí |
| ASHRAE901_Warehouse_STD2019_Denver.yaml | Sí | Sí |
| LrgOff_GridStorageScheduled.yaml | Sí | Sí |
| ShopWithPVandBattery.yaml | Sí | Sí |
| radiant_residential_building.yaml | Sí | Sí |
| **nuestro.yaml** | Sí | No |
| **nuestroMultrizona.yaml** | Sí | No |

---

## 3. Resumen de diferencias

1. **Versión:** Este repo = **3.9.1** (Poetry). RPi = **3.11.0** (pyproject PEP 621).
2. **Código de producción:** Solo en este repo: `scripts/train/prod/` (Home Assistant, PyEnvProduction, entrenamiento online).
3. **Configuraciones “nuestro”:** Solo en este repo: `nuestroMultrizona.yaml` y `nuestro.yaml`.
4. **try_env.py:** En la RPi usa DatetimeWrapper y otro formato de log; en el repo usa log por mes y recompensa acumulada.
5. **2ZoneDataCenterHVAC:** En el repo un solo YAML; en la RPi dos: _DX y _CW.
6. **EnergyPlus (ya visto antes):** RPi = 25.2.0; documentación del repo = 24.1.0.

Para usar en la RPi el mismo flujo que en este repo (multizona, producción, HA) hace falta **copiar a la imagen** (o montar como volumen) la carpeta `scripts/train/prod/` y los YAML `nuestroMultrizona.yaml` y `nuestro.yaml`, y tener en cuenta la diferencia de versión (3.9.1 vs 3.11.0) y de `try_env.py` si lo usas.
