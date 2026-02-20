"""
Puente entre el modelo RL y Home Assistant
Toma acciones del modelo, las envía a Home Assistant, y lee sensores desde Home Assistant
"""
import numpy as np
import requests
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings

try:
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  requests no encontrado, instalando...")
    import subprocess
    subprocess.check_call(["pip", "install", "requests"])
    HAS_REQUESTS = True


def _get_forecast_vector_fn():
    """Importa get_forecast_vector desde forecast_utils (funciona ejecutando desde prod/ o como paquete)."""
    try:
        from .forecast_utils import get_forecast_vector
        return get_forecast_vector
    except ImportError:
        import os
        import sys
        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        from forecast_utils import get_forecast_vector
        return get_forecast_vector


class HomeAssistantBridge:
    """
    Puente que conecta el modelo RL con Home Assistant.
    
    Flujo:
    1. Recibe acción del modelo (normalizada)
    2. Desnormaliza la acción al rango real
    3. Envía la acción a Home Assistant (actuadores/entidades)
    4. Lee sensores desde Home Assistant
    5. Convierte las lecturas al formato de observación esperado
    """
    
    def __init__(
        self,
        ha_url: str,
        ha_token: str,
        sensor_entities: Dict[str, str],  # {variable_name: entity_id} (incluye variables + meters)
        actuator_entities: Dict[str, str],  # {action_name: entity_id}
        action_space_low: np.ndarray,
        action_space_high: np.ndarray,
        time_variables: List[str] = ['month', 'day_of_month', 'hour'],
        action_delay: float = 1.0,  # Tiempo de espera después de enviar acción
        sensor_delay: float = 0.5,  # Tiempo de espera antes de leer sensores
        timestep_delay: float = 0.0,  # Delay adicional entre pasos (en segundos)
        flag_entity: Optional[str] = None,  # input_boolean.sinergym_simulator_ready para sincronía con energyplusNexus
        forecast_csv: Optional[str] = None,  # Si se define, se añade forecast como variable paso a paso (sin wrapper)
        forecast_horizon: int = 5,  # Número de pasos/horas de forecast a concatenar a la observación
        variable_names: Optional[List[str]] = None,  # Nombres de variables (sin meters)
        meter_names: Optional[List[str]] = None,  # Nombres de meters
        **kwargs
    ):
        """
        Inicializa el puente con Home Assistant.
        
        Args:
            ha_url: URL base de Home Assistant (ej: "http://localhost:8123")
            ha_token: Token de acceso de Home Assistant (Long-Lived Access Token)
            sensor_entities: Mapeo de nombres de variables a entity_id de sensores
            actuator_entities: Mapeo de nombres de acciones a entity_id de actuadores
            action_space_low: Límites inferiores del action space
            action_space_high: Límites superiores del action space
            time_variables: Variables de tiempo a incluir en observaciones
            action_delay: Tiempo de espera (segundos) después de enviar acción
            sensor_delay: Tiempo de espera (segundos) antes de leer sensores
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests no está disponible. Instálalo con: pip install requests")
        
        self.ha_url = ha_url.rstrip('/')
        self.ha_token = ha_token
        self.sensor_entities = sensor_entities
        self.actuator_entities = actuator_entities
        self.action_space_low = np.array(action_space_low, dtype=np.float32)
        self.action_space_high = np.array(action_space_high, dtype=np.float32)
        self.time_variables = time_variables
        self.action_delay = action_delay
        self.sensor_delay = sensor_delay
        self.timestep_delay = timestep_delay
        self.flag_entity = flag_entity
        self.forecast_csv = forecast_csv
        self.forecast_horizon = forecast_horizon
        self.variable_names = variable_names or []
        self.meter_names = meter_names or []
        
        # Headers para las peticiones a Home Assistant
        self.headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json"
        }
        
        # Estado interno
        self.current_timestep = 0
        self.current_month = 1
        self.current_day = 1
        self.current_hour = 0
        
        # Verificar conexión con Home Assistant
        self._verify_connection()
        
        # Variables de observación esperadas
        self.observation_variable_names = self._build_observation_variable_names()
        
        print(f"✅ HomeAssistantBridge inicializado")
        print(f"   URL: {self.ha_url}")
        print(f"   Sensores: {len(self.sensor_entities)}")
        print(f"   Actuadores: {len(self.actuator_entities)}")
        print(f"   Action delay: {self.action_delay}s")
        print(f"   Sensor delay: {self.sensor_delay}s")
        if self.timestep_delay > 0:
            print(f"   Timestep delay: {self.timestep_delay}s ({self.timestep_delay/60:.1f} minutos entre pasos)")
        if self.flag_entity:
            print(f"   Flag sync: {self.flag_entity} (esperar ON → leer → escribir → OFF)")
        if self.forecast_csv:
            print(f"   Forecast como variable: {self.forecast_csv} (horizon={self.forecast_horizon})")
    
    def _verify_connection(self):
        """Verifica la conexión con Home Assistant."""
        try:
            response = requests.get(
                f"{self.ha_url}/api/",
                headers=self.headers,
                timeout=10  # Aumentar timeout
            )
            if response.status_code == 200:
                print(f"   ✅ Conexión con Home Assistant verificada")
            else:
                raise RuntimeError(f"No se pudo conectar a Home Assistant. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Intentar con diferentes URLs si falla
            print(f"   ⚠️  Error conectando a {self.ha_url}: {e}")
            # No lanzar error inmediatamente, permitir que el sistema intente más tarde
            warnings.warn(f"No se pudo verificar conexión inicial con Home Assistant: {e}")
            # El sistema puede seguir funcionando y reconectar más tarde
    
    def _build_observation_variable_names(self) -> List[str]:
        """Construye la lista de nombres de variables de observación."""
        var_names = []
        
        # Time variables
        var_names.extend(self.time_variables)
        
        # Variables de sensores
        var_names.extend(list(self.sensor_entities.keys()))
        
        # Nota: Los meters se manejan como sensores adicionales en sensor_entities
        # Si necesitas separar meters, puedes agregar un parámetro meter_entities
        
        return var_names
    
    def desnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Desnormaliza una acción del rango del modelo al rango real.
        
        Args:
            action: Acción normalizada (típicamente en [-1, 1] o [0, 1])
            
        Returns:
            Acción desnormalizada en el rango real
        """
        action = np.array(action, dtype=np.float32)
        
        # Detectar rango de entrada
        action_min = np.min(action)
        action_max = np.max(action)
        
        # Si está en [-1, 1], desnormalizar a [low, high]
        if action_min >= -1.1 and action_max <= 1.1:
            # De [-1, 1] a [low, high]
            action_real = (action + 1.0) / 2.0  # A [0, 1]
            action_real = action_real * (self.action_space_high - self.action_space_low) + self.action_space_low
        # Si está en [0, 1], desnormalizar a [low, high]
        elif action_min >= -0.1 and action_max <= 1.1:
            action_real = action * (self.action_space_high - self.action_space_low) + self.action_space_low
        # Si ya está en el rango correcto, usar directamente
        else:
            action_real = action
        
        # Clip a los límites
        action_real = np.clip(action_real, self.action_space_low, self.action_space_high)
        
        return action_real
    
    def _get_sensor_value(self, entity_id: str) -> Optional[float]:
        """
        Obtiene el valor de un sensor desde Home Assistant.
        
        Args:
            entity_id: ID de la entidad (ej: "sensor.temperature")
            
        Returns:
            Valor del sensor o None si hay error
        """
        try:
            response = requests.get(
                f"{self.ha_url}/api/states/{entity_id}",
                headers=self.headers,
                timeout=10  # Aumentar timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                state = data.get('state', 'unknown')
                
                # Intentar convertir a float
                try:
                    if state == 'unknown' or state == 'unavailable':
                        return None
                    return float(state)
                except (ValueError, TypeError):
                    # Si no es numérico, intentar extraer número del string
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', str(state))
                    if numbers:
                        return float(numbers[0])
                    return None
            else:
                # Solo mostrar warning si no es un error de conexión común
                if response.status_code != 404:
                    warnings.warn(f"No se pudo leer sensor {entity_id}. Status: {response.status_code}")
                return None
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Error de conexión - retornar None silenciosamente (se manejará en el nivel superior)
            return None
        except requests.exceptions.RequestException as e:
            # Otros errores HTTP - mostrar warning solo en modo debug
            return None
    
    def _set_actuator_value(self, entity_id: str, value: float, service: str = None) -> bool:
        """
        Establece el valor de un actuador en Home Assistant.
        
        Args:
            entity_id: ID de la entidad (ej: "input_number.setpoint")
            value: Valor a establecer
            service: Servicio a llamar (None = auto-detect)
            
        Returns:
            True si se ejecutó correctamente, False en caso contrario
        """
        try:
            # Determinar el dominio de la entidad (ej: "input_number" de "input_number.setpoint")
            domain = entity_id.split('.')[0]
            
            # Construir el payload según el tipo de entidad
            if domain == "input_number":
                service = service or "set_value"
                payload = {
                    "entity_id": entity_id,
                    "value": float(value)
                }
            elif domain == "number":
                service = service or "set_value"
                payload = {
                    "entity_id": entity_id,
                    "value": float(value)
                }
            elif domain == "climate":
                # Para termostatos, usar set_temperature
                service = service or "set_temperature"
                payload = {
                    "entity_id": entity_id,
                    "temperature": float(value)
                }
            elif domain == "switch" or domain == "light":
                # Para switches/lights, usar turn_on/turn_off según el valor
                if value > 0.5:
                    service = service or "turn_on"
                    payload = {"entity_id": entity_id}
                else:
                    service = service or "turn_off"
                    payload = {"entity_id": entity_id}
            elif domain == "input_boolean":
                # Para booleanos, usar turn_on/turn_off
                if value > 0.5:
                    service = service or "turn_on"
                    payload = {"entity_id": entity_id}
                else:
                    service = service or "turn_off"
                    payload = {"entity_id": entity_id}
            else:
                # Intentar servicio genérico set_value
                service = service or "set_value"
                payload = {
                    "entity_id": entity_id,
                    "value": float(value)
                }
            
            response = requests.post(
                f"{self.ha_url}/api/services/{domain}/{service}",
                headers=self.headers,
                json=payload,
                timeout=10  # Aumentar timeout
            )
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"      ✅ Respuesta OK ({response.status_code})")
                return True
            else:
                print(f"      ❌ Error HTTP {response.status_code}: {response.text[:200]}")
                warnings.warn(f"No se pudo establecer actuador {entity_id}. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Error de conexión
            print(f"      ❌ Error de conexión: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Error de solicitud: {e}")
            warnings.warn(f"Error estableciendo actuador {entity_id}: {e}")
            return False
    
    def _get_time_variables(self) -> Dict[str, int]:
        """Obtiene las variables de tiempo actuales."""
        from datetime import datetime
        now = datetime.now()
        return {
            'month': now.month,
            'day_of_month': now.day,
            'hour': now.hour
        }
    
    def wait_for_flag_on(self, poll_interval: float = 1.0) -> None:
        """
        Espera hasta que la bandera de sincronización esté ON.
        El simulador (energyplusNexus) pone la bandera ON cuando ya escribió los sensores.
        """
        if not self.flag_entity:
            return
        while True:
            try:
                response = requests.get(
                    f"{self.ha_url}/api/states/{self.flag_entity}",
                    headers=self.headers,
                    timeout=10,
                )
                if response.status_code == 200:
                    state = response.json().get("state", "").lower()
                    if state == "on":
                        return
                time.sleep(poll_interval)
            except requests.exceptions.RequestException:
                time.sleep(poll_interval)
    
    def set_flag_off(self) -> None:
        """Pone la bandera en OFF para indicar que el modelo ya subió actuadores."""
        if not self.flag_entity:
            return
        try:
            domain = self.flag_entity.split(".")[0]
            requests.post(
                f"{self.ha_url}/api/services/{domain}/turn_off",
                headers=self.headers,
                json={"entity_id": self.flag_entity},
                timeout=10,
            ).raise_for_status()
        except requests.exceptions.RequestException as e:
            warnings.warn(f"No se pudo bajar la bandera {self.flag_entity}: {e}")
    
    def get_observation_from_sensors(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solo lee sensores y construye el vector de observación (sin escribir actuadores).
        Útil para el protocolo: esperar ON → leer sensores → modelo → escribir actuadores → OFF.
        """
        if self.sensor_delay > 0:
            time.sleep(self.sensor_delay)
        sensor_values = {}
        for var_name, entity_id in self.sensor_entities.items():
            value = self._get_sensor_value(entity_id)
            sensor_values[var_name] = value if value is not None else 0.0
        time_vars = self._get_time_variables()
        self.current_month = time_vars["month"]
        self.current_day = time_vars["day_of_month"]
        self.current_hour = time_vars["hour"]
        observation = []
        for tv in self.time_variables:
            observation.append(float(time_vars.get(tv, 0)))
        for var_name in self.variable_names:
            observation.append(float(sensor_values.get(var_name, 0.0)))
        for meter_name in self.meter_names:
            observation.append(float(sensor_values.get(meter_name, 0.0)))
        obs = np.array(observation, dtype=np.float32)
        # Forecast como variable paso a paso (sin wrapper)
        if self.forecast_csv:
            _get_forecast = _get_forecast_vector_fn()
            forecast_vec = _get_forecast(
                self.current_timestep, self.forecast_csv, self.forecast_horizon
            )
            obs = np.concatenate([obs, forecast_vec], axis=0)
        info = {
            "sensor_values": sensor_values,
            "timestep": self.current_timestep,
            "month": self.current_month,
            "day_of_month": self.current_day,
            "hour": self.current_hour,
        }
        return obs, info
    
    def write_actuators(self, action: np.ndarray, *, normalized: bool = True) -> Dict[str, Any]:
        """
        Escribe los valores de los actuadores en Home Assistant.
        Si normalized=True, desnormaliza la acción de [-1,1] al rango real.
        """
        action_array = np.array(action, dtype=np.float32)
        if normalized:
            action_real = self.desnormalize_action(action_array)
        else:
            action_real = np.clip(action_array, self.action_space_low, self.action_space_high)
        actuator_names = list(self.actuator_entities.keys())
        action_results = {}
        for i, actuator_name in enumerate(actuator_names):
            if i < len(action_real):
                entity_id = self.actuator_entities[actuator_name]
                value = float(action_real[i])
                success = self._set_actuator_value(entity_id, value)
                action_results[actuator_name] = {
                    "entity_id": entity_id,
                    "value": value,
                    "success": success,
                }
        if self.action_delay > 0:
            time.sleep(self.action_delay)
        return {"action": action_real.tolist(), "action_results": action_results}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ejecuta un paso: envía acción a Home Assistant y lee sensores.
        Si flag_entity está configurado, sigue el protocolo: escribir actuadores → OFF → esperar ON → leer sensores.
        
        Args:
            action: Acción del modelo (normalizada)
            
        Returns:
            Tuple con (observación, info_dict)
        """
        # Flujo con bandera de sincronización (energyplusNexus)
        if self.flag_entity:
            write_result = self.write_actuators(action, normalized=True)
            self.set_flag_off()
            self.wait_for_flag_on()
            obs, info = self.get_observation_from_sensors()
            self.current_timestep += 1
            info["action"] = write_result.get("action", [])
            info["action_results"] = write_result.get("action_results", {})
            return obs, info
        
        # 1. PROBLEMA CRÍTICO: NormalizeAction wrapper usa np.rint() que redondea a enteros
        # Esto causa que [0.45, 0.215, 15.9] se convierta en [0, 0, 16]
        # SOLUCIÓN TEMPORAL: Intentar recibir la acción normalizada desde info si está disponible
        # SOLUCIÓN PERMANENTE: Modificar el wrapper o interceptar antes del wrapper
        
        action_array = np.array(action, dtype=np.float32)
        action_min = np.min(action_array)
        action_max = np.max(action_array)
        
        print(f"\n📤 Acción recibida del wrapper: {action_array}")
        print(f"   Rango: [{action_min:.4f}, {action_max:.4f}]")
        
        # Intentar obtener la acción normalizada original desde el contexto
        # Por ahora, si la acción está mal desnormalizada, intentamos corregirla
        # detectando el patrón de redondeo
        
        # Si la acción está en rango normalizado, desnormalizar manualmente
        if action_min >= -1.1 and action_max <= 1.1:
            print(f"   → Acción todavía normalizada, desnormalizando...")
            action_real = self.desnormalize_action(action)
            print(f"   ✅ Desnormalizada: {action_real}")
        else:
            # La acción ya fue desnormalizada por el wrapper (probablemente mal)
            # Intentar detectar si está mal redondeada y corregirla
            # Por ahora, usamos la acción tal como está
            action_real = action_array
            print(f"   ⚠️  Acción ya desnormalizada por wrapper (puede tener errores de redondeo)")
        
        print(f"   Action space esperado: [{self.action_space_low}] → [{self.action_space_high}]")
        
        # 2. Enviar acciones a Home Assistant
        action_results = {}
        actuator_names = list(self.actuator_entities.keys())
        
        print(f"\n📤 Enviando {len(actuator_names)} acciones a Home Assistant:")
        for i, actuator_name in enumerate(actuator_names):
            if i < len(action_real):
                entity_id = self.actuator_entities[actuator_name]
                value = float(action_real[i])
                print(f"   [{i}] {actuator_name:35s} → {entity_id:40s} = {value:.4f}")
                success = self._set_actuator_value(entity_id, value)
                if success:
                    print(f"      ✅ Enviado correctamente")
                else:
                    print(f"      ❌ Error al enviar")
                action_results[actuator_name] = {
                    'entity_id': entity_id,
                    'value': value,
                    'success': success
                }
        
        # 3. Esperar un poco para que Home Assistant procese las acciones
        if self.action_delay > 0:
            time.sleep(self.action_delay)
        
        # 4. Esperar un poco antes de leer sensores
        if self.sensor_delay > 0:
            time.sleep(self.sensor_delay)
        
        # 5. Leer sensores desde Home Assistant
        sensor_values = {}
        for var_name, entity_id in self.sensor_entities.items():
            value = self._get_sensor_value(entity_id)
            sensor_values[var_name] = value if value is not None else 0.0
        
        # 6. Obtener variables de tiempo (usar hora actual del sistema)
        # El avance de hora se maneja en pyenv_production después del paso
        time_vars = self._get_time_variables()
        self.current_month = time_vars['month']
        self.current_day = time_vars['day_of_month']
        self.current_hour = time_vars['hour']
        
        # 7. Construir vector de observación
        # Formato: time_variables + variables + meters
        observation = []
        
        # Time variables
        for tv in self.time_variables:
            observation.append(float(time_vars.get(tv, 0)))
        
        # Variables (solo las que están en variable_names)
        for var_name in self.variable_names:
            observation.append(float(sensor_values.get(var_name, 0.0)))
        
        # Meters (solo los que están en meter_names)
        for meter_name in self.meter_names:
            observation.append(float(sensor_values.get(meter_name, 0.0)))
        
        observation = np.array(observation, dtype=np.float32)
        if self.forecast_csv:
            _get_forecast = _get_forecast_vector_fn()
            forecast_vec = _get_forecast(self.current_timestep, self.forecast_csv, self.forecast_horizon)
            observation = np.concatenate([observation, forecast_vec], axis=0)
        # 8. Esperar el delay del timestep (para que cada paso tome el tiempo real esperado)
        if self.timestep_delay > 0:
            time.sleep(self.timestep_delay)
        # 9. Actualizar contador
        self.current_timestep += 1
        
        # 10. Construir info completo
        info_complete = {
            'action': action_real.tolist(),
            'action_normalized': action.tolist(),
            'action_results': action_results,
            'sensor_values': sensor_values,
            'timestep': self.current_timestep,
            'month': self.current_month,
            'day_of_month': self.current_day,
            'hour': self.current_hour,  # Hora después del paso (avanzada 1 hora)
        }
        
        return observation, info_complete
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el estado (lee sensores iniciales).
        Si flag_entity está configurado, espera bandera ON y luego lee sensores.
        
        Returns:
            Tuple con (observación inicial, info_dict)
        """
        if self.flag_entity:
            self.wait_for_flag_on()
            obs, info = self.get_observation_from_sensors()
            self.current_timestep = 0
            return obs, info
        # 1. Esperar un poco antes de leer sensores
        if self.sensor_delay > 0:
            time.sleep(self.sensor_delay)
        # 2. Leer sensores desde Home Assistant
        sensor_values = {}
        for var_name, entity_id in self.sensor_entities.items():
            value = self._get_sensor_value(entity_id)
            sensor_values[var_name] = value if value is not None else 0.0
        
        # 3. Obtener variables de tiempo
        time_vars = self._get_time_variables()
        self.current_month = time_vars['month']
        self.current_day = time_vars['day_of_month']
        self.current_hour = time_vars['hour']
        
        # 4. Construir vector de observación
        # Formato: time_variables + variables + meters
        observation = []
        
        # Time variables
        for tv in self.time_variables:
            observation.append(float(time_vars.get(tv, 0)))
        
        # Variables (solo las que están en variable_names)
        for var_name in self.variable_names:
            observation.append(float(sensor_values.get(var_name, 0.0)))
        
        # Meters (solo los que están en meter_names)
        for meter_name in self.meter_names:
            observation.append(float(sensor_values.get(meter_name, 0.0)))
        
        observation = np.array(observation, dtype=np.float32)
        if self.forecast_csv:
            _get_forecast = _get_forecast_vector_fn()
            forecast_vec = _get_forecast(self.current_timestep, self.forecast_csv, self.forecast_horizon)
            observation = np.concatenate([observation, forecast_vec], axis=0)
        # 5. Resetear contador
        self.current_timestep = 0
        # 6. Construir info completo
        info_complete = {
            'timestep': 0,
            'month': self.current_month,
            'day_of_month': self.current_day,
            'hour': self.current_hour,
            'sensor_values': sensor_values,
        }
        
        return observation, info_complete
    
    def close(self):
        """Cierra la conexión con Home Assistant."""
        print(f"✅ HomeAssistantBridge cerrado")
    
    def __del__(self):
        """Destructor: cierra la conexión si está abierta."""
        self.close()
