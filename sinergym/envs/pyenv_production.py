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
    from sinergym.utils.rewards import LinearReward
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
        # Variables vienen como dict: {'nombre': ('variable_name', 'keys')}
        # Convertimos a lista simple de nombres
        self.variables_info = variables
        self.variable_names = list(variables.keys())
        
        # ============ ESPACIOS ============
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6,
            shape=(len(self.variable_names),),
            dtype=np.float32
        )
        
        # ============ RECOMPENSA ============
        if reward and callable(reward):
            self.reward_fn = reward(**reward_kwargs)
        else:
            # Por defecto, LinearReward
            from sinergym.utils.rewards import LinearReward
            self.reward_fn = LinearReward(**reward_kwargs)
        
        # ============ CONFIGURACIÓN DE PRODUCCIÓN ============
        self.production_config = production_config or {}
        self.data_mode = self.production_config.get('data_mode', 'terminal')
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
        
        # Actualizar hora real
        now = datetime.now()
        self.hour = now.hour
        self.minute = now.minute
        
        print(f"\n{'='*60}")
        print(f"🎬 EPISODIO DE PRODUCCIÓN #{self.episode_count}")
        print(f"   Inicio: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Obtener primera observación
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta un paso en el entorno real."""
        self.step_count += 1
        self.total_steps += 1
        self.simulation_time += 900  # 15 minutos por paso
        
        # Actualizar tiempo simulado
        self.minute += 15
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
        
        print(f"\n{'='*60}")
        print(f"🔄 PRODUCCIÓN - Paso {self.step_count}")
        print(f"   Acción: {action}")
        print(f"   Hora simulación: {self.hour:02d}:{self.minute:02d}")
        
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
        reward = 0.0
        reward_info = {}
        if self.reward_fn:
            obs_dict = dict(zip(self.variable_names, obs))
            reward, reward_info = self.reward_fn(obs_dict)
        
        # 5. Determinar fin de episodio
        terminated = False
        max_steps = self.production_config.get('max_steps_per_episode', 288)
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
        # Diccionario de atributos que los wrappers necesitan
        attr_map = {
            'timestep_per_episode': 288,  # 24 horas * 4 steps por hora
            'timestep_per_hour': 4,
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
            'meters': self._meters,
            'actuators': self._actuators,
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'to_dict': lambda: self._to_dict(),
        }
        
        if name in attr_map:
            return attr_map[name]
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
    
    # ============ MÉTODOS DE PRODUCCIÓN ============
    
    def _get_production_observation(self):
        """Obtiene observación del sistema real."""
        mode = self.data_mode
        
        try:
            if mode == 'terminal':
                return self._get_observation_terminal()
            elif mode == 'api':
                return self._get_observation_api()
            elif mode == 'mqtt':
                return self._get_observation_mqtt()
            elif mode == 'database':
                return self._get_observation_database()
            elif mode == 'simulated':
                return self._get_observation_simulated()
            else:
                # Por defecto, terminal
                return self._get_observation_terminal()
        except Exception as e:
            print(f"⚠️  Error obteniendo observación: {e}")
            return self._get_safe_observation()
    
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
        """Obtiene datos por terminal."""
        print(f"\n📥 Ingrese {len(self.variable_names)} valores REALES:")
        for i, var in enumerate(self.variable_names):
            print(f"   {i+1:2d}. {var}")
        print("   (Formato: valor1,valor2,... o Enter para valores simulados)")
        
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == 'auto' or user_input == '':
                # Modo automático (valores simulados realistas)
                return self._get_observation_simulated()
            else:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != len(self.variable_names):
                    raise ValueError(f"Se esperaban {len(self.variable_names)} valores")
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
        
        for var_name in self.variable_names:
            var_lower = var_name.lower()
            
            # Lógica de simulación realista
            if 'temp' in var_lower and 'outdoor' in var_lower:
                # Temperatura exterior: variación diurna
                value = 10.0 + 10.0 * np.sin(2 * np.pi * current_hour / 24)
                value += np.random.randn() * 2  # Ruido
            elif 'temp' in var_lower:
                # Temperatura interior
                value = 22.0 + np.random.randn() * 0.5
            elif 'hum' in var_lower:
                # Humedad
                value = 50.0 + np.random.randn() * 5
            elif 'power' in var_lower or 'energy' in var_lower:
                # Consumo energético
                if 8 <= current_hour <= 20:
                    value = 1500.0 + np.random.randn() * 200
                else:
                    value = 500.0 + np.random.randn() * 100
            elif 'occupancy' in var_lower:
                # Ocupación
                if 9 <= current_hour <= 18 and datetime.now().weekday() < 5:
                    value = 1.0
                else:
                    value = 0.0
            else:
                value = 0.0
            
            values.append(value)
        
        return np.array(values, dtype=np.float32)
    
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