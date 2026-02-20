"""
PyEnvProduction - Entorno de producción compatible con Sinergym
Versión optimizada para aprendizaje online continuo
"""
import numpy as np
import gymnasium as gym
from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import json
import os
import threading
import warnings
from datetime import datetime
from queue import Queue, Empty

try:
    from sinergym.utils.common import get_delta_seconds
    from sinergym.utils.rewards import NuestroRewardMultizona
    HAS_SINERGYM = True
except ImportError:
    HAS_SINERGYM = False
    print("⚠️  Sinergym no encontrado, usando versiones mínimas")

class PyEnvProduction(gym.Env):
    """
    Entorno de producción que se comporta como EplusEnv pero usa datos reales.
    Compatible 100% con todos los wrappers y scripts de Sinergym.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(
        self,
        # Parámetros OBLIGATORIOS para compatibilidad con Sinergym
        building_file: str,
        weather_files: List[str],
        variables: Dict[str, Any],
        meters: Dict[str, str],
        actuators: Dict[str, Tuple[str, str, str]],
        action_space: gym.spaces.Box,
        reward: Callable,
        reward_kwargs: Dict[str, Any],
        env_name: str = 'Eplus-pyenv-v1',
        config_params: Dict[str, Any] = {},
        time_variables: List[str] = [],  # Variables de tiempo (month, day_of_month, hour, etc.)
        
        # Parámetros específicos de producción
        production_config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__()
        
        # ============ GUARDAR PARÁMETROS ORIGINALES ============
        self._building_file = building_file
        self._weather_files = weather_files
        self._env_name = env_name
        self._config_params = config_params or {}
        self._actuators = actuators
        self._meters = meters
        
        # ============ PROCESAR VARIABLES ============
        # Variables vienen como dict del YAML: {'Variable Name': {'variable_names': 'name', 'keys': [...]}}
        # O ya procesadas: {'variable_name': ('Variable Name', 'key')}
        self.time_variables = time_variables or []
        self.variables_info = variables
        self.variable_names = self._extract_variable_names(variables)
        self.meter_names = list(meters.keys()) if meters else []
        
        # observation_variables = time_variables + variable_names + meter_names (igual que EplusEnv)
        self.observation_variables = self.time_variables + self.variable_names + self.meter_names
        
        # ============ ESPACIOS ============
        self.action_space = action_space
        # El espacio de observación incluye time_variables + variables + meters
        obs_size = len(self.time_variables) + len(self.variable_names) + len(self.meter_names)
        # Forecast como variable (sin wrapper): opcional desde api_config.homeassistant
        self._forecast_csv = None
        self._forecast_horizon = 5
        _ha = (production_config or {}).get('api_config', {}).get('homeassistant', {})
        if _ha.get('forecast_csv'):
            self._forecast_csv = _ha.get('forecast_csv')
            self._forecast_horizon = _ha.get('forecast_horizon', 5)
            try:
                from forecast_utils import get_forecast_extra_dim
            except ImportError:
                import sys
                _p = os.path.dirname(os.path.abspath(__file__))
                if _p not in sys.path:
                    sys.path.insert(0, _p)
                from forecast_utils import get_forecast_extra_dim
            obs_size += get_forecast_extra_dim(self._forecast_csv, self._forecast_horizon)
        # DEBUG: Mostrar desglose de dimensiones
        print(f"\n📊 DESGLOSE DE DIMENSIONES DEL ENTORNO (BASE, SIN WRAPPERS):")
        print(f"   time_variables ({len(self.time_variables)}): {self.time_variables}")
        print(f"   variable_names ({len(self.variable_names)}):")
        for i, vn in enumerate(self.variable_names):
            print(f"      [{i}] {vn}")
        print(f"   meter_names ({len(self.meter_names)}): {self.meter_names}")
        print(f"   TOTAL BASE: {obs_size} dimensiones")
        print(f"   ESPERADO: 19 dimensiones (3 time + 15 variables + 1 meter)")
        if obs_size != 19:
            print(f"   ❌ DIFERENCIA: {obs_size - 19} dimensiones de más")
        
        # IMPORTANTE: Usar los mismos rangos que el modelo entrenado
        # El modelo fue entrenado con low=-5e7, high=5e7 (no -5e6, 5e6)
        self.observation_space = gym.spaces.Box(
            low=-5e7, high=5e7,  # Coincidir con el modelo entrenado
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # ============ RECOMPENSA ============
        if reward and callable(reward):
            self.reward_fn = reward(**reward_kwargs)
        else:
            # Por defecto, LinearReward
            from sinergym.utils.rewards import NuestroRewardMultizona
            self.reward_fn = NuestroRewardMultizona(**reward_kwargs)
        
        # ============ CONFIGURACIÓN DE PRODUCCIÓN ============
        self.production_config = production_config or {}
        self.data_mode = self.production_config.get('data_mode', 'terminal')
        
        # Inicializar bridges (se hará después de crear workspace_path)
        self.energyplus_bridge = None
        self.homeassistant_bridge = None
        self.api_config = self.production_config.get('api_config', {})
        self.safety_limits = self.production_config.get('safety_limits', {})
        
        # ============ ESTADO INTERNO ============
        self.episode_count = 0
        self.step_count = 0
        self.total_steps = 0
        self.simulation_time = 0
        self._is_running = False
        
        # Para compatibilidad con wrappers de tiempo
        self.year = config_params.get('start_year', datetime.now().year)
        self.month = config_params.get('start_month', datetime.now().month)
        self.day = config_params.get('start_day', datetime.now().day)
        self.hour = datetime.now().hour
        self.minute = datetime.now().minute
        
        # Workspace (igual que EplusEnv)
        self.workspace_path = f"./workspaces/{env_name}"
        os.makedirs(self.workspace_path, exist_ok=True)
        
        # Inicializar bridges (ahora que workspace_path existe)
        if self.data_mode == 'energyplus':
            try:
                from energyplus_bridge import EnergyPlusBridge
                self.energyplus_bridge = EnergyPlusBridge(
                    building_file=building_file,
                    weather_file=weather_files[0] if weather_files else 'dummy.epw',
                    variables=variables,
                    meters=meters,
                    actuators=actuators,
                    action_space_low=action_space.low,
                    action_space_high=action_space.high,
                    time_variables=time_variables,
                    workspace_path=self.workspace_path,
                    timesteps_per_hour=self._config_params.get('timesteps_per_hour', 4),
                    reward=reward,
                    reward_kwargs=reward_kwargs
                )
                print(f"   ✅ EnergyPlus bridge inicializado")
            except Exception as e:
                print(f"   ⚠️  Error inicializando EnergyPlus bridge: {e}")
                import traceback
                traceback.print_exc()
                print(f"   ⚠️  Usando modo terminal como fallback")
                self.data_mode = 'terminal'
        
        elif self.data_mode == 'homeassistant':
            try:
                from homeassistant_bridge import HomeAssistantBridge
                
                # Obtener configuración de Home Assistant
                ha_config = self.api_config.get('homeassistant', {})
                ha_url = ha_config.get('url', 'http://localhost:8123')
                ha_token = ha_config.get('token', '')
                
                if not ha_token:
                    raise ValueError("Home Assistant token no proporcionado en api_config.homeassistant.token")
                
                # Mapear variables a sensores de Home Assistant
                # Formato esperado: {variable_name: entity_id}
                sensor_entities = ha_config.get('sensor_entities', {})
                if not sensor_entities:
                    # Si no está configurado, crear mapeo automático basado en variable_names
                    sensor_entities = {}
                    for var_name in self.variable_names:
                        # Intentar mapeo automático (ej: "temperature" -> "sensor.temperature")
                        sensor_entities[var_name] = f"sensor.{var_name.lower().replace(' ', '_')}"
                    print(f"   ⚠️  No se encontró sensor_entities, usando mapeo automático: {sensor_entities}")
                
                # Agregar meters a sensor_entities si no están ya incluidos
                # Los meters también se leen como sensores desde Home Assistant
                for meter_name in self.meter_names:
                    if meter_name not in sensor_entities:
                        # Si no está configurado, crear mapeo automático para meters
                        sensor_entities[meter_name] = f"sensor.{meter_name.lower().replace(' ', '_')}"
                        print(f"   ⚠️  Meter '{meter_name}' agregado a sensor_entities automáticamente")
                
                # Mapear actuadores a entidades de Home Assistant
                # Formato esperado: {action_name: entity_id}
                actuator_entities = ha_config.get('actuator_entities', {})
                if not actuator_entities:
                    # Si no está configurado, crear mapeo automático basado en actuators
                    actuator_entities = {}
                    actuator_names = list(actuators.keys())[:len(action_space.low)]
                    for act_name in actuator_names:
                        # Intentar mapeo automático
                        actuator_entities[act_name] = f"input_number.{act_name.lower().replace(' ', '_')}"
                    print(f"   ⚠️  No se encontró actuator_entities, usando mapeo automático: {actuator_entities}")
                
                self.homeassistant_bridge = HomeAssistantBridge(
                    ha_url=ha_url,
                    ha_token=ha_token,
                    sensor_entities=sensor_entities,
                    actuator_entities=actuator_entities,
                    action_space_low=action_space.low,
                    action_space_high=action_space.high,
                    time_variables=time_variables,
                    action_delay=ha_config.get('action_delay', 1.0),
                    sensor_delay=ha_config.get('sensor_delay', 0.5),
                    timestep_delay=ha_config.get('timestep_delay_seconds', 0.0),
                    flag_entity=ha_config.get('flag_entity'),
                    forecast_csv=ha_config.get('forecast_csv') or getattr(self, '_forecast_csv', None),
                    forecast_horizon=ha_config.get('forecast_horizon', 5) if ha_config.get('forecast_csv') else getattr(self, '_forecast_horizon', 5),
                    variable_names=self.variable_names,
                    meter_names=self.meter_names
                )
                print(f"   ✅ Home Assistant bridge inicializado")
            except Exception as e:
                print(f"   ⚠️  Error inicializando Home Assistant bridge: {e}")
                import traceback
                traceback.print_exc()
                print(f"   ⚠️  Intentando reconectar en el primer reset()...")
                # No cambiar a terminal inmediatamente, intentar reconectar en reset()
                # self.data_mode = 'terminal'  # Comentado para permitir reintento
        
        # Configurar streaming si es necesario
        self._init_streaming()
        
        print(f"✅ PyEnvProduction inicializado: {env_name}")
        print(f"   Variables: {self.variable_names}")
        print(f"   Modo datos: {self.data_mode}")
        print(f"   Action space: {action_space}")
    
    # ============ MÉTODOS PÚBLICOS (MISMA INTERFAZ QUE EplusEnv) ============
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reinicia el entorno para nuevo episodio."""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.step_count = 0
        self.simulation_time = 0
        self._is_running = True
        
        # Actualizar episode_path para wrappers
        self._episode_path = os.path.join(self.workspace_path, f'episode-{self.episode_count}')
        os.makedirs(self._episode_path, exist_ok=True)
        
        # Actualizar hora real
        now = datetime.now()
        self.hour = now.hour
        self.minute = now.minute
        
        print(f"\n{'='*60}")
        print(f"🎬 EPISODIO DE PRODUCCIÓN #{self.episode_count}")
        print(f"   Inicio: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Obtener primera observación
        # Si está usando EnergyPlus o Home Assistant, inicializar el bridge
        if self.data_mode == 'energyplus' and self.energyplus_bridge is not None:
            obs, info = self.energyplus_bridge.reset()
        elif self.data_mode == 'homeassistant':
            # Intentar inicializar bridge si no existe o reconectar
            if self.homeassistant_bridge is None:
                try:
                    from homeassistant_bridge import HomeAssistantBridge
                    ha_config = self.api_config.get('homeassistant', {})
                    ha_url = ha_config.get('url', 'http://localhost:8123')
                    ha_token = ha_config.get('token', '')
                    
                    if ha_token:
                        sensor_entities = ha_config.get('sensor_entities', {})
                        actuator_entities = ha_config.get('actuator_entities', {})
                        
                        # Agregar meters a sensor_entities si no están incluidos
                        for meter_name in self.meter_names:
                            if meter_name not in sensor_entities:
                                sensor_entities[meter_name] = f"sensor.{meter_name.lower().replace(' ', '_')}"
                        
                        self.homeassistant_bridge = HomeAssistantBridge(
                            ha_url=ha_url,
                            ha_token=ha_token,
                            sensor_entities=sensor_entities,
                            actuator_entities=actuator_entities,
                            action_space_low=self.action_space.low,
                            action_space_high=self.action_space.high,
                            time_variables=self.time_variables,
                            action_delay=ha_config.get('action_delay', 1.0),
                            sensor_delay=ha_config.get('sensor_delay', 0.5),
                            timestep_delay=ha_config.get('timestep_delay_seconds', 0.0),
                            flag_entity=ha_config.get('flag_entity'),
                            forecast_csv=ha_config.get('forecast_csv') or getattr(self, '_forecast_csv', None),
                            forecast_horizon=ha_config.get('forecast_horizon', 5) if ha_config.get('forecast_csv') else getattr(self, '_forecast_horizon', 5),
                            variable_names=self.variable_names,
                            meter_names=self.meter_names
                        )
                        print(f"   ✅ Home Assistant bridge inicializado en reset()")
                except Exception as e:
                    print(f"   ⚠️  No se pudo inicializar bridge en reset(): {e}")
                    print(f"   ⚠️  Usando modo terminal como fallback")
                    self.data_mode = 'terminal'
                    obs = self._get_production_observation()
                    info = self._create_info_dict()
                else:
                    obs, info = self.homeassistant_bridge.reset()
            else:
                obs, info = self.homeassistant_bridge.reset()
        else:
            obs = self._get_production_observation()
            # Info para wrappers
            info = self._create_info_dict()
        
        # Loggear inicio
        self._log_event('episode_start', {
            'episode': self.episode_count,
            'timestamp': now.isoformat(),
            'initial_observation': obs.tolist()
        })
        
        return obs.astype(np.float32), info
    
    def _step_energyplus(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta un paso usando EnergyPlus bridge.
        
        Args:
            action: Acción del modelo (normalizada)
            
        Returns:
            Tuple con (observación, recompensa, terminated, truncated, info)
        """
        # Ejecutar paso en EnergyPlus
        obs, info = self.energyplus_bridge.step(action)
        
        # Calcular recompensa
        reward = 0.0
        reward_info = {}
        
        if self.reward_fn:
            # Construir obs_dict para la función de recompensa
            obs_dict = {}
            
            # 1. Time variables
            for i, tv in enumerate(self.time_variables):
                obs_dict[tv] = obs[i]
            
            # 2. Variables
            var_start = len(self.time_variables)
            var_end = var_start + len(self.variable_names)
            for i, var_name in enumerate(self.variable_names):
                obs_dict[var_name] = obs[var_start + i]
            
            # 3. Meters
            meter_start = var_end
            for i, meter_name in enumerate(self.meter_names):
                obs_dict[meter_name] = obs[meter_start + i]
            
            # Calcular recompensa
            reward, reward_info = self.reward_fn(obs_dict)
        
        # Actualizar info con recompensa
        info.update({
            'reward': reward,
            **reward_info
        })
        
        # Determinar si el episodio terminó
        terminated = info.get('terminated', False)
        truncated = info.get('truncated', False)
        
        # Actualizar contadores y tiempo simulado
        self.step_count += 1
        self.total_steps += 1
        self.simulation_time += 300  # 5 minutos por paso (300 segundos)
        
        # Avanzar tiempo simulado - cada paso avanza 5 minutos
        self.minute += 5
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
            if self.hour >= 24:
                self.hour = 0
                self.day += 1
                if self.day > 28:  # Simplificado
                    self.day = 1
                    self.month += 1
                    if self.month > 12:
                        self.month = 1
                        self.year += 1
        
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info
    
    def _step_homeassistant(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta un paso usando Home Assistant bridge.
        
        Args:
            action: Acción del modelo (puede estar desnormalizada por NormalizeAction wrapper)
            
        Returns:
            Tuple con (observación, recompensa, terminated, truncated, info)
        """
        # IMPORTANTE: El NormalizeAction wrapper usa np.rint() que redondea a enteros
        # Esto causa problemas con acciones continuas (ej: [0.45, 0.215] → [0, 0])
        # 
        # El wrapper ya procesó la acción, pero podemos intentar obtener la acción
        # normalizada original desde info si está disponible, o pasar la acción
        # tal como está y dejar que el bridge intente corregirla
        
        # Intentar obtener acción normalizada original guardada antes del wrapper
        action_normalized_original = None
        if hasattr(self, '_last_normalized_action'):
            action_normalized_original = self._last_normalized_action
        
        # Si tenemos la acción normalizada original, usarla para desnormalización correcta
        if action_normalized_original is not None:
            print(f"   🔧 Usando acción normalizada original (evitando np.rint del wrapper)")
            obs, info = self.homeassistant_bridge.step(action_normalized_original)
            # Limpiar después de usar
            self._last_normalized_action = None
        else:
            # Pasar la acción tal como está (ya desnormalizada por wrapper, puede tener errores)
            obs, info = self.homeassistant_bridge.step(action)
        
        # Calcular recompensa
        reward = 0.0
        reward_info = {}
        
        if self.reward_fn:
            # Construir obs_dict para la función de recompensa
            obs_dict = {}
            
            # 1. Time variables
            for i, tv in enumerate(self.time_variables):
                obs_dict[tv] = obs[i]
            
            # 2. Variables
            var_start = len(self.time_variables)
            var_end = var_start + len(self.variable_names)
            for i, var_name in enumerate(self.variable_names):
                obs_dict[var_name] = obs[var_start + i]
            
            # 3. Meters
            meter_start = var_end
            for i, meter_name in enumerate(self.meter_names):
                obs_dict[meter_name] = obs[meter_start + i]
            
            # Calcular recompensa
            reward, reward_info = self.reward_fn(obs_dict)
        
        # Actualizar info con recompensa
        info.update({
            'reward': reward,
            **reward_info
        })
        
        # Determinar si el episodio terminó
        terminated = info.get('terminated', False)
        truncated = info.get('truncated', False)
        
        # Actualizar contadores y tiempo simulado
        self.step_count += 1
        self.total_steps += 1
        self.simulation_time += 300  # 5 minutos por paso (300 segundos)
        
        # Avanzar tiempo simulado - cada paso avanza 5 minutos
        self.minute += 5
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
            if self.hour >= 24:
                self.hour = 0
                self.day += 1
                if self.day > 28:  # Simplificado
                    self.day = 1
                    self.month += 1
                    if self.month > 12:
                        self.month = 1
                        self.year += 1
        
        # Actualizar info con la hora después del paso
        info['hour'] = self.hour
        info['minute'] = self.minute
        info['day_of_month'] = self.day
        info['month'] = self.month
        info['year'] = self.year
        
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta un paso en el entorno real."""
        
        # Si está usando EnergyPlus o Home Assistant bridge, delegar el step completo
        if self.data_mode == 'energyplus' and self.energyplus_bridge is not None:
            return self._step_energyplus(action)
        elif self.data_mode == 'homeassistant' and self.homeassistant_bridge is not None:
            return self._step_homeassistant(action)
        
        self.step_count += 1
        self.total_steps += 1
        self.simulation_time += 300  # 5 minutos por paso (300 segundos)
        
        # Actualizar tiempo simulado - cada paso avanza 5 minutos
        self.minute += 5
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
            if self.hour >= 24:
                self.hour = 0
                self.day += 1
                # Manejar cambio de mes (simplificado, asumimos meses de 30 días)
                if self.day > 30:
                    self.day = 1
                    self.month += 1
                    if self.month > 12:
                        self.month = 1
                        self.year += 1
        
        print(f"\n{'='*60}")
        print(f"🔄 PRODUCCIÓN - Paso {self.step_count}")
        print(f"   Acción: {action}")
        print(f"   Hora simulación: {self.hour:02d}:{self.minute:02d} (paso de 5 minutos)")
        
        # 1. Ejecutar acción en sistema real
        action_success = self._execute_production_action(action)
        
        if not action_success:
            print("⚠️  Advertencia: La acción no se ejecutó completamente")
        
        # 2. Esperar intervalo real (configurable)
        delay = self.production_config.get('action_delay', 1.0)
        time.sleep(delay)
        
        # 3. Obtener nueva observación
        obs = self._get_production_observation()
        
        # 4. Calcular recompensa
        # La observación completa es: time_variables + variables + meters
        # NuestroRewardMultizona necesita time_variables (month, day_of_month, hour) + variables + meters
        reward = 0.0
        reward_info = {}
        if self.reward_fn:
            # Construir obs_dict completo: time_variables + variables + meters
            obs_dict = {}
            
            # 1. Agregar time_variables
            for i, tv in enumerate(self.time_variables):
                obs_dict[tv] = obs[i]
            
            # 2. Agregar variables
            var_start = len(self.time_variables)
            var_end = var_start + len(self.variable_names)
            for i, var_name in enumerate(self.variable_names):
                obs_dict[var_name] = obs[var_start + i]
            
            # 3. Agregar meters
            meter_start = var_end
            for i, meter_name in enumerate(self.meter_names):
                obs_dict[meter_name] = obs[meter_start + i]
            
            reward, reward_info = self.reward_fn(obs_dict)
        
        # 5. Determinar fin de episodio
        terminated = False
        max_steps = self.production_config.get('max_steps_per_episode', 288)  # 24 horas = 288 pasos (12 pasos/hora, 5 min/paso)
        truncated = self.step_count >= max_steps
        
        # 6. Info para wrappers
        info = self._create_info_dict()
        info.update({
            'action': action.tolist(),
            'action_success': action_success,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'time_elapsed': self.simulation_time,
            'real_timestamp': datetime.now().isoformat(),
        })
        info.update(reward_info)
        
        # 7. Loggear paso
        self._log_event('step', {
            'episode': self.episode_count,
            'step': self.step_count,
            'action': action.tolist(),
            'observation': obs.tolist(),
            'reward': float(reward),
        })
        
        # 8. Monitoreo de seguridad
        self._safety_monitoring(obs, action)
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def close(self):
        """Cierra el entorno y guarda logs."""
        self._is_running = False
        
        # Cerrar bridges
        if self.energyplus_bridge is not None:
            self.energyplus_bridge.close()
        if self.homeassistant_bridge is not None:
            self.homeassistant_bridge.close()
        
        # Guardar estado final
        self._save_state()
        
        print(f"\n✅ PyEnvProduction cerrado.")
        print(f"   Episodios: {self.episode_count}")
        print(f"   Pasos totales: {self.total_steps}")
        print(f"   Workspace: {self.workspace_path}")
    
    def render(self, mode='human'):
        """Renderiza estado actual."""
        if mode == 'human':
            print(f"\n📊 PyEnvProduction - Episodio {self.episode_count}, Paso {self.step_count}")
            print(f"   Hora: {self.hour:02d}:{self.minute:02d}")
            print(f"   Variables: {len(self.variable_names)}")
            print(f"   Modo: {self.data_mode}")
    
    # ============ MÉTODOS DE COMPATIBILIDAD CON SINERGYM ============
    
    def get_wrapper_attr(self, name: str):
        """
        Método CRÍTICO para compatibilidad con wrappers.
        Los wrappers de Sinergym llaman a esto para acceder a atributos.
        """
        # Atributos especiales que necesitan procesamiento
        if name == 'episode_path':
            # Crear episode_path si no existe
            if not hasattr(self, '_episode_path') or self._episode_path is None:
                episode_num = getattr(self, 'episode_count', 0)
                if episode_num == 0:
                    episode_num = 1  # Empezar desde 1
                self._episode_path = os.path.join(self.workspace_path, f'episode-{episode_num}')
                os.makedirs(self._episode_path, exist_ok=True)
            return self._episode_path
        
        # Diccionario de atributos que los wrappers necesitan
        attr_map = {
            'timestep_per_episode': 288,  # 24 horas * 12 pasos por hora
            'timestep_per_hour': 12,  # 12 pasos por hora (cada paso = 5 minutos)
            'workspace_path': self.workspace_path,
            'is_running': self._is_running,
            'name': self._env_name,
            'episode': self.episode_count,
            'timestep': self.step_count,
            'total_steps': self.total_steps,
            'simulation_time': self.simulation_time,
            'output_directory': self.workspace_path,
            'building_path': self._building_file,
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
            'variables': self.variable_names,
            'observation_variables': self.observation_variables,  # ✅ CRÍTICO para wrappers como DatetimeWrapper
            'time_variables': self.time_variables,
            'meters': self._meters,
            'actuators': self._actuators,
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'to_dict': lambda: self._to_dict(),
        }
        
        if name in attr_map:
            return attr_map[name]
        elif name == 'episode_path':
            # Crear episode_path si no existe (crítico para NormalizeObservation)
            if not hasattr(self, '_episode_path') or self._episode_path is None:
                episode_num = getattr(self, 'episode_count', 0)
                if episode_num == 0:
                    episode_num = 1  # Empezar desde 1
                self._episode_path = os.path.join(self.workspace_path, f'episode-{episode_num}')
                os.makedirs(self._episode_path, exist_ok=True)
            return self._episode_path
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            # Para evitar errores en wrappers, retornamos None con warning
            warnings.warn(f"PyEnvProduction: Atributo '{name}' no encontrado, retornando None")
            return None
    
    def _to_dict(self):
        """Para WandBLogger y otros wrappers que necesitan serialización."""
        return {
            'env_name': self._env_name,
            'building_file': self._building_file,
            'weather_files': self._weather_files,
            'variables': self.variables_info,
            'meters': self._meters,
            'actuators': self._actuators,
            'action_space': str(self.action_space),
            'workspace_path': self.workspace_path,
            'production_config': self.production_config,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'is_running': self._is_running,
        }
    
    def _extract_variable_names(self, variables: Dict[str, Any]) -> List[str]:
        """
        Extrae nombres de variables desde el formato YAML de Sinergym.
        
        Soporta dos formatos:
        1. Formato YAML crudo: {'Variable Name': {'variable_names': 'name', 'keys': [...]}}
        2. Formato procesado: {'variable_name': ('Variable Name', 'key')}
        
        Args:
            variables: Diccionario de variables en formato YAML o procesado
            
        Returns:
            Lista de nombres de variables para las observaciones
        """
        var_names = []
        
        for var_key, var_value in variables.items():
            # Caso 1: Formato YAML crudo (dict con 'variable_names' y 'keys')
            if isinstance(var_value, dict):
                if 'variable_names' in var_value:
                    var_names_field = var_value['variable_names']
                    keys_field = var_value.get('keys', [])
                    
                    # variable_names puede ser string o lista
                    if isinstance(var_names_field, str):
                        # keys puede ser string o lista
                        if isinstance(keys_field, str):
                            # Caso simple: una variable, una key
                            var_names.append(var_names_field)
                        elif isinstance(keys_field, list):
                            # Múltiples keys: crear nombres con prefijo
                            for key in keys_field:
                                prename = key.lower().replace(' ', '_') + '_'
                                var_names.append(prename + var_names_field)
                        else:
                            # keys vacío o None
                            var_names.append(var_names_field)
                    elif isinstance(var_names_field, list):
                        # variable_names es lista
                        if isinstance(keys_field, list) and len(keys_field) == len(var_names_field):
                            # Mapeo directo
                            var_names.extend(var_names_field)
                        else:
                            # Usar los nombres directamente
                            var_names.extend(var_names_field)
                    else:
                        # Fallback: usar la clave
                        var_names.append(var_key)
                else:
                    # Dict sin 'variable_names': usar la clave
                    var_names.append(var_key)
            
            # Caso 2: Formato procesado (tupla o ya es el nombre)
            elif isinstance(var_value, (tuple, list)) and len(var_value) >= 1:
                # Formato: ('Variable Name', 'key') o ya procesado
                # La clave ya es el nombre de la variable
                var_names.append(var_key)
            else:
                # Fallback: usar la clave como nombre
                var_names.append(var_key)
        
        return var_names
    
    # ============ MÉTODOS DE PRODUCCIÓN ============
    
    def _get_production_observation(self):
        """Obtiene observación del sistema real (time_variables + variables + meters)."""
        mode = self.data_mode
        
        try:
            # Obtener valores de variables (sin time_variables ni meters)
            if mode == 'terminal':
                var_values = self._get_observation_terminal()
            elif mode == 'api':
                var_values = self._get_observation_api()
            elif mode == 'mqtt':
                var_values = self._get_observation_mqtt()
            elif mode == 'database':
                var_values = self._get_observation_database()
            elif mode == 'simulated':
                var_values = self._get_observation_simulated()
            else:
                # Por defecto, terminal
                var_values = self._get_observation_terminal()
            
            # Construir observación completa: time_variables + variables + meters
            obs = self._build_complete_observation(var_values)
            return obs
            
        except Exception as e:
            print(f"⚠️  Error obteniendo observación: {e}")
            var_values = self._get_safe_observation()
            return self._build_complete_observation(var_values)
    
    def _build_complete_observation(self, var_values: np.ndarray) -> np.ndarray:
        """
        Construye la observación completa: time_variables + variables + meters.
        
        Args:
            var_values: Array con valores de variables (sin time_variables ni meters)
            
        Returns:
            Array completo con time_variables + variables + meters
        """
        obs_parts = []
        
        # 1. Time variables
        for tv in self.time_variables:
            if tv == 'month':
                obs_parts.append(float(self.month))
            elif tv == 'day_of_month':
                obs_parts.append(float(self.day))
            elif tv == 'hour':
                obs_parts.append(float(self.hour))
            elif tv == 'minute':
                obs_parts.append(float(self.minute))
            elif tv == 'year':
                obs_parts.append(float(self.year))
            else:
                # Variable de tiempo desconocida, usar 0
                obs_parts.append(0.0)
        
        # 2. Variables (ya obtenidas)
        obs_parts.extend(var_values.tolist())
        
        # 3. Meters (simulados o desde configuración)
        for meter_name in self.meter_names:
            # Por ahora, usar 0.0 o puedes implementar lectura real
            # En producción, estos vendrían de sensores reales
            obs_parts.append(0.0)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def _execute_production_action(self, action):
        """Ejecuta acción en el sistema real."""
        try:
            print(f"   [ACCION REAL] Enviando a sistema: {action}")
            
            # Aquí iría tu código para enviar acciones a:
            # - API REST
            # - MQTT
            # - Base de datos
            # - Sistema de control directo
            
            # Ejemplo para API:
            if self.data_mode == 'api' and 'action_endpoint' in self.api_config:
                import requests
                endpoint = self.api_config['action_endpoint']
                payload = {
                    'action': action.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                response = requests.post(endpoint, json=payload, timeout=5)
                return response.status_code == 200
            
            return True
        except Exception as e:
            print(f"❌ Error ejecutando acción: {e}")
            return False
    
    def _get_observation_terminal(self):
        """Obtiene datos por terminal (solo variables, sin time_variables ni meters)."""
        print(f"\n📥 Ingrese {len(self.variable_names)} valores REALES (solo variables):")
        for i, var in enumerate(self.variable_names):
            print(f"   {i+1:2d}. {var}")
        if self.time_variables:
            print(f"   (Time variables: {', '.join(self.time_variables)} se agregan automáticamente)")
        if self.meter_names:
            print(f"   (Meters: {', '.join(self.meter_names)} se agregan automáticamente)")
        print("   (Formato: valor1,valor2,... o Enter para valores simulados)")
        
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == 'auto' or user_input == '':
                # Modo automático (valores simulados realistas)
                return self._get_observation_simulated()
            else:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != len(self.variable_names):
                    raise ValueError(f"Se esperaban {len(self.variable_names)} valores, recibidos {len(values)}")
                return np.array(values, dtype=np.float32)
                
        except Exception as e:
            print(f"   Error: {e}. Usando valores simulados.")
            return self._get_observation_simulated()
    
    def _get_observation_api(self):
        """Obtiene datos de API REST."""
        import requests
        
        endpoint = self.api_config.get('observation_endpoint', '')
        if not endpoint:
            print("   ⚠️  No hay endpoint configurado. Usando valores simulados.")
            return self._get_observation_simulated()
        
        try:
            response = requests.get(endpoint, timeout=5)
            data = response.json()
            
            # Mapear datos de la API a nuestras variables
            values = []
            for var_name in self.variable_names:
                # Buscar directamente
                if var_name in data:
                    values.append(float(data[var_name]))
                else:
                    # Intentar mapeo configurado
                    mapping = self.api_config.get('field_mapping', {})
                    if var_name in mapping:
                        mapped_field = mapping[var_name]
                        if mapped_field in data:
                            values.append(float(data[mapped_field]))
                        else:
                            # Valor por defecto
                            values.append(0.0)
                    else:
                        # Valor por defecto
                        values.append(0.0)
            
            return np.array(values, dtype=np.float32)
            
        except Exception as e:
            print(f"   Error API: {e}. Usando valores simulados.")
            return self._get_observation_simulated()
    
    def _get_observation_simulated(self):
        """Genera datos simulados realistas para pruebas."""
        values = []
        current_hour = datetime.now().hour
        
        print(f"\n🔮 Generando valores simulados (hora: {current_hour:02d}:00):")
        
        for var_name in self.variable_names:
            var_lower = var_name.lower()
            
            # Lógica de simulación realista
            if 'temp' in var_lower and 'outdoor' in var_lower:
                # Temperatura exterior: variación diurna
                value = 10.0 + 10.0 * np.sin(2 * np.pi * current_hour / 24)
                value += np.random.randn() * 2  # Ruido
                print(f"   🌡️  {var_name}: {value:.2f}°C (temperatura exterior)")
            elif 'temp' in var_lower:
                # Temperatura interior
                value = 22.0 + np.random.randn() * 0.5
                print(f"   🌡️  {var_name}: {value:.2f}°C (temperatura interior)")
            elif 'hum' in var_lower:
                # Humedad
                value = 50.0 + np.random.randn() * 5
                print(f"   💧 {var_name}: {value:.2f}% (humedad)")
            elif 'power' in var_lower or 'energy' in var_lower:
                # Consumo energético
                if 8 <= current_hour <= 20:
                    value = 1500.0 + np.random.randn() * 200
                else:
                    value = 500.0 + np.random.randn() * 100
                print(f"   ⚡ {var_name}: {value:.2f}W (consumo energético)")
            elif 'occupancy' in var_lower:
                # Ocupación
                if 9 <= current_hour <= 18 and datetime.now().weekday() < 5:
                    value = 1.0
                else:
                    value = 0.0
                print(f"   👥 {var_name}: {value:.0f} (ocupación)")
            else:
                value = 0.0
                print(f"   📊 {var_name}: {value:.2f} (valor por defecto)")
            
            values.append(value)
        
        values_array = np.array(values, dtype=np.float32)
        print(f"   ✅ Vector completo: {values_array}")
        print(f"   📏 Dimensiones: {len(values_array)} valores")
        
        return values_array
    
    def _get_safe_observation(self):
        """Observación segura por defecto."""
        values = []
        for var_name in self.variable_names:
            if 'temp' in var_name.lower():
                values.append(22.0)  # Temperatura segura
            elif 'hum' in var_name.lower():
                values.append(50.0)  # Humedad segura
            elif 'power' in var_name.lower():
                values.append(1000.0)  # Consumo normal
            else:
                values.append(0.0)
        
        return np.array(values, dtype=np.float32)
    
    # ============ MÉTODOS AUXILIARES ============
    
    def _create_info_dict(self):
        """Crea dictionary de info compatible con wrappers."""
        return {
            'timestep': self.step_count,
            'time_elapsed': self.simulation_time,
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'real_time': datetime.now().isoformat(),
            'is_production': True,
            'data_mode': self.data_mode,
        }
    
    def _safety_monitoring(self, obs, action):
        """Monitorea límites de seguridad."""
        for i, (var_name, value) in enumerate(zip(self.variable_names, obs)):
            if 'temp' in var_name.lower() and not 'outdoor' in var_name.lower():
                max_temp = self.safety_limits.get('max_zone_temperature', 28.0)
                min_temp = self.safety_limits.get('min_zone_temperature', 16.0)
                
                if value > max_temp:
                    print(f"🚨 ALERTA SEGURIDAD: {var_name} = {value:.1f}°C > {max_temp}°C")
                elif value < min_temp:
                    print(f"🚨 ALERTA SEGURIDAD: {var_name} = {value:.1f}°C < {min_temp}°C")
    
    def _log_event(self, event_type, data):
        """Loggea eventos a archivo."""
        log_entry = {
            'event': event_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        log_file = os.path.join(self.workspace_path, 'production_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _save_state(self):
        """Guarda estado del entorno."""
        state = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat(),
            'workspace_path': self.workspace_path,
        }
        
        state_file = os.path.join(self.workspace_path, 'production_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _init_streaming(self):
        """Inicializa sistema de streaming (opcional)."""
        # Puedes implementar threads para streaming continuo aquí
        pass
    
    # Placeholders para otras fuentes de datos
    def _get_observation_mqtt(self):
        """Obtiene datos de MQTT."""
        # Implementar según tu broker MQTT
        return self._get_observation_simulated()
    
    def _get_observation_database(self):
        """Obtiene datos de base de datos."""
        # Implementar según tu base de datos
        return self._get_observation_simulated()