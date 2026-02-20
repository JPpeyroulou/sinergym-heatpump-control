"""
Puente entre el modelo RL y EnergyPlus
Toma acciones del modelo, las inyecta en EnergyPlus, y convierte las salidas en observaciones
"""
import numpy as np
import subprocess
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile
import shutil

try:
    from sinergym.envs.eplus_env import EplusEnv
    from sinergym.utils.common import parse_variables_settings, parse_meters_settings
    HAS_SINERGYM = True
except ImportError:
    HAS_SINERGYM = False
    print("⚠️  Sinergym no encontrado, usando modo básico")


class EnergyPlusBridge:
    """
    Puente que conecta el modelo RL con EnergyPlus.
    
    Flujo:
    1. Recibe acción del modelo (normalizada)
    2. Desnormaliza la acción al rango real
    3. Inyecta la acción en EnergyPlus (actuadores)
    4. Ejecuta un timestep de EnergyPlus
    5. Lee las salidas de EnergyPlus (variables y meters)
    6. Convierte las salidas al formato de observación esperado
    """
    
    def __init__(
        self,
        building_file: str,
        weather_file: str,
        variables: Dict[str, Any],
        meters: Dict[str, str],
        actuators: Dict[str, Tuple[str, str, str]],
        action_space_low: np.ndarray,
        action_space_high: np.ndarray,
        time_variables: List[str] = ['month', 'day_of_month', 'hour'],
        workspace_path: Optional[str] = None,
        timesteps_per_hour: int = 4,
        reward: Optional[Any] = None,
        reward_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Inicializa el puente con EnergyPlus.
        
        Args:
            building_file: Archivo del edificio (.epJSON o .idf)
            weather_file: Archivo de clima (.epw)
            variables: Diccionario de variables de observación (formato Sinergym)
            meters: Diccionario de meters (formato Sinergym)
            actuators: Diccionario de actuadores {name: (element_type, value_type, key)}
            action_space_low: Límites inferiores del action space
            action_space_high: Límites superiores del action space
            time_variables: Variables de tiempo a incluir en observaciones
            workspace_path: Ruta donde EnergyPlus guardará los resultados
            timesteps_per_hour: Pasos por hora en EnergyPlus
        """
        self.building_file = building_file
        self.weather_file = weather_file
        self.variables = variables
        self.meters = meters
        self.actuators = actuators
        self.action_space_low = np.array(action_space_low, dtype=np.float32)
        self.action_space_high = np.array(action_space_high, dtype=np.float32)
        self.time_variables = time_variables
        self.timesteps_per_hour = timesteps_per_hour
        self.reward = reward
        self.reward_kwargs = reward_kwargs or {}
        
        # Workspace para EnergyPlus
        if workspace_path is None:
            self.workspace_path = tempfile.mkdtemp(prefix='eplus_bridge_')
        else:
            self.workspace_path = workspace_path
            os.makedirs(workspace_path, exist_ok=True)
        
        # Procesar variables y meters al formato EnergyPlus
        self.variables_parsed = parse_variables_settings(variables) if HAS_SINERGYM else {}
        self.meters_parsed = parse_meters_settings(meters) if HAS_SINERGYM else meters
        
        # Mapeo de actuadores
        # IMPORTANTE: Solo usar los actuadores que corresponden a las dimensiones del action_space
        # Según nuestroMultrizona.yaml, los 5 actuadores son:
        # 1. ZonaNorth (electrovalve_north)
        # 2. ZonaSouth (electrovalve_south)
        # 3. ZonaEast (electrovalve_east)
        # 4. ZonaWest (electrovalve_west)
        # 5. Temperatura_Calefaccion_Schedule (bomba)
        action_dim = len(action_space_low)
        
        # Orden esperado de actuadores (según nuestroMultrizona.yaml)
        expected_actuators = ['ZonaNorth', 'ZonaSouth', 'ZonaEast', 'ZonaWest', 'Temperatura_Calefaccion_Schedule']
        
        # Filtrar solo los actuadores que están en la lista esperada y en el diccionario
        self.actuator_names = [name for name in expected_actuators[:action_dim] if name in actuators]
        
        if len(self.actuator_names) != action_dim:
            print(f"   ⚠️  ADVERTENCIA: Se esperaban {action_dim} actuadores pero se encontraron {len(self.actuator_names)}")
            print(f"   ⚠️  Actuadores disponibles: {list(actuators.keys())}")
            print(f"   ⚠️  Actuadores esperados: {expected_actuators[:action_dim]}")
            # Usar los primeros N actuadores disponibles como fallback
            self.actuator_names = list(actuators.keys())[:action_dim]
        
        self.actuator_mapping = {k: actuators[k] for k in self.actuator_names}
        
        print(f"   📋 Actuadores seleccionados ({len(self.actuator_names)}): {self.actuator_names}")
        
        # Estado de la simulación
        self.energyplus_env = None
        self.current_timestep = 0
        self.current_month = 1
        self.current_day = 1
        self.current_hour = 0
        
        # Variables de observación esperadas
        self.observation_variable_names = self._build_observation_variable_names()
        
        print(f"✅ EnergyPlusBridge inicializado")
        print(f"   Building: {building_file}")
        print(f"   Weather: {weather_file}")
        print(f"   Actuators: {len(self.actuator_names)}")
        print(f"   Variables: {len(self.variables_parsed)}")
        print(f"   Meters: {len(self.meters_parsed)}")
        print(f"   Workspace: {self.workspace_path}")
    
    def _build_observation_variable_names(self) -> List[str]:
        """Construye la lista de nombres de variables de observación."""
        var_names = []
        
        # Time variables
        var_names.extend(self.time_variables)
        
        # Variables de EnergyPlus
        var_names.extend(list(self.variables_parsed.keys()))
        
        # Meters
        var_names.extend(list(self.meters_parsed.keys()))
        
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
    
    def initialize_energyplus(self):
        """Inicializa el entorno de EnergyPlus."""
        if not HAS_SINERGYM:
            raise RuntimeError("Sinergym no está disponible. No se puede usar EnergyPlus.")
        
        try:
            # Crear entorno de EnergyPlus (EplusEnv)
            # Necesitamos un action_space dummy para EplusEnv
            import gymnasium as gym
            dummy_action_space = gym.spaces.Box(
                low=self.action_space_low,
                high=self.action_space_high,
                dtype=np.float32
            )
            
            # Construir building_config con timesteps_per_hour
            building_config = {
                'timesteps_per_hour': self.timesteps_per_hour
            }
            
            # Usar la reward function proporcionada o NuestroRewardMultizona por defecto
            if self.reward is None:
                try:
                    from sinergym.utils.rewards import NuestroRewardMultizona
                    reward_fn = NuestroRewardMultizona
                except ImportError:
                    from sinergym.utils.rewards import LinearReward
                    reward_fn = LinearReward
                    # Parámetros mínimos para LinearReward
                    if not self.reward_kwargs:
                        self.reward_kwargs = {
                            'temperature_variables': [],
                            'energy_variables': [],
                            'range_comfort_winter': (20.0, 23.5),
                            'range_comfort_summer': (23.0, 26.0)
                        }
            else:
                reward_fn = self.reward
            
            # Convertir actuadores del formato tupla al formato dict de Sinergym
            # Formato recibido: {'ZonaNorth': ['Schedule:Compact', 'Schedule Value', 'ZonaNorth']}
            # Formato esperado: {'ZonaNorth': ('Schedule:Compact', 'Schedule Value', 'ZonaNorth')}
            actuators_formatted = {}
            for name, spec in self.actuator_mapping.items():
                if isinstance(spec, list):
                    actuators_formatted[name] = tuple(spec)
                elif isinstance(spec, tuple):
                    actuators_formatted[name] = spec
                else:
                    actuators_formatted[name] = spec
            
            print(f"   🔧 Creando EplusEnv con:")
            print(f"      Building: {self.building_file}")
            print(f"      Weather: {self.weather_file}")
            print(f"      Actuators: {list(actuators_formatted.keys())}")
            print(f"      Action space: {dummy_action_space.shape[0]} dimensiones")
            
            self.energyplus_env = EplusEnv(
                building_file=self.building_file,
                weather_files=self.weather_file,
                action_space=dummy_action_space,
                time_variables=self.time_variables,
                variables=self.variables_parsed,
                meters=self.meters_parsed,
                actuators=actuators_formatted,
                env_name='EplusBridge',
                building_config=building_config,
                reward=reward_fn,
                reward_kwargs=self.reward_kwargs
            )
            
            print(f"✅ EnergyPlus inicializado")
            print(f"   Timesteps por hora: {self.timesteps_per_hour}")
            
        except Exception as e:
            print(f"❌ Error inicializando EnergyPlus: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ejecuta un paso: inyecta acción en EnergyPlus y obtiene observación.
        
        Args:
            action: Acción del modelo (normalizada)
            
        Returns:
            Tuple con (observación, info_dict)
        """
        if self.energyplus_env is None:
            raise RuntimeError("EnergyPlus no está inicializado. Llama a initialize_energyplus() primero.")
        
        # 1. Desnormalizar acción
        action_real = self.desnormalize_action(action)
        
        # 2. Ejecutar paso en EnergyPlus
        obs, reward, terminated, truncated, info = self.energyplus_env.step(action_real)
        
        # 3. La observación ya viene como vector numpy de EplusEnv
        observation = np.array(obs, dtype=np.float32)
        
        # 4. Actualizar tiempo interno
        self.current_timestep += 1
        if 'month' in info:
            self.current_month = info['month']
        if 'day_of_month' in info:
            self.current_day = info['day_of_month']
        if 'hour' in info:
            self.current_hour = info['hour']
        
        # 5. Construir info completo
        info_complete = {
            'action': action_real.tolist(),
            'action_normalized': action.tolist(),
            'timestep': self.current_timestep,
            'month': self.current_month,
            'day_of_month': self.current_day,
            'hour': self.current_hour,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            **info
        }
        
        return observation, info_complete
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia la simulación de EnergyPlus.
        
        Returns:
            Tuple con (observación inicial, info_dict)
        """
        if self.energyplus_env is None:
            self.initialize_energyplus()
        
        # Resetear entorno
        obs, info = self.energyplus_env.reset()
        observation = np.array(obs, dtype=np.float32)
        
        # Resetear contadores
        self.current_timestep = 0
        if 'month' in info:
            self.current_month = info['month']
        if 'day_of_month' in info:
            self.current_day = info['day_of_month']
        if 'hour' in info:
            self.current_hour = info['hour']
        
        info_complete = {
            'timestep': 0,
            'month': self.current_month,
            'day_of_month': self.current_day,
            'hour': self.current_hour,
            **info
        }
        
        return observation, info_complete
    
    def _dict_to_observation(self, obs_dict: Dict[str, float], info: Dict[str, Any]) -> np.ndarray:
        """
        Convierte un diccionario de observaciones a un vector numpy.
        (Método auxiliar, pero EplusEnv ya devuelve el vector directamente)
        
        Args:
            obs_dict: Diccionario con valores de variables y meters
            info: Diccionario con información adicional (incluye time_variables)
            
        Returns:
            Vector numpy con la observación completa
        """
        obs_parts = []
        
        # 1. Time variables
        for tv in self.time_variables:
            if tv in info:
                obs_parts.append(float(info[tv]))
            elif tv == 'month' and 'month' in info:
                obs_parts.append(float(info['month']))
            elif tv == 'day_of_month' and 'day_of_month' in info:
                obs_parts.append(float(info['day_of_month']))
            elif tv == 'hour' and 'hour' in info:
                obs_parts.append(float(info['hour']))
            else:
                # Valor por defecto
                obs_parts.append(0.0)
        
        # 2. Variables de EnergyPlus (en el orden esperado)
        for var_name in self.variables_parsed.keys():
            if var_name in obs_dict:
                obs_parts.append(float(obs_dict[var_name]))
            else:
                obs_parts.append(0.0)
        
        # 3. Meters (en el orden esperado)
        for meter_name in self.meters_parsed.keys():
            if meter_name in obs_dict:
                obs_parts.append(float(obs_dict[meter_name]))
            else:
                obs_parts.append(0.0)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def close(self):
        """Cierra la simulación de EnergyPlus."""
        if self.energyplus_env is not None:
            self.energyplus_env.close()
            self.energyplus_env = None
            print(f"✅ EnergyPlus cerrado")
    
    def __del__(self):
        """Destructor: cierra EnergyPlus si está abierto."""
        self.close()
