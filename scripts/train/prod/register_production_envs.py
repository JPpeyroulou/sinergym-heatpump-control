"""
REGISTRO DEL ENTORNO DE PRODUCCIÓN
Este archivo REGISTRA PyEnvProduction en Gymnasium/Sinergym
"""
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# Importar tu entorno (asegúrate de que pyenv_production.py esté en el mismo directorio)
try:
    from pyenv_production import PyEnvProduction
    HAS_PYENV = True
except ImportError:
    HAS_PYENV = False
    print("⚠️  No se pudo importar PyEnvProduction")

if HAS_PYENV:
    # ============ REGISTRAR EN GYMNASIUM ============
    register(
        id='Eplus-pyenv-multizona-production-v1',
        entry_point='pyenv_production:PyEnvProduction',
        kwargs={
            # Valores por defecto - serán sobreescritos por la configuración
            'building_file': 'OfficeMedium_Zone_4.pkl',
            'weather_files': ['USA_CO_Denver.Intl.AP.725650_TMY3.epw'],
            'variables': {
                'zone_air_temperature_1': ('Zone Air Temperature', 'Zone 1'),
                'zone_air_temperature_2': ('Zone Air Temperature', 'Zone 2'),
                'zone_air_temperature_3': ('Zone Air Temperature', 'Zone 3'),
                'zone_air_temperature_4': ('Zone Air Temperature', 'Zone 4'),
                'total_electric_demand': ('Facility Total Electric Demand Power', 'Whole Building'),
                'outdoor_air_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment')
            },
            'meters': {
                'electricity': 'Electricity:Facility'
            },
            'actuators': {
                'heating_setpoint_1': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 1'),
                'cooling_setpoint_1': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 1'),
                'heating_setpoint_2': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 2'),
                'cooling_setpoint_2': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 2'),
                'heating_setpoint_3': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 3'),
                'cooling_setpoint_3': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 3'),
                'heating_setpoint_4': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 4'),
                'cooling_setpoint_4': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 4')
            },
            'action_space': gym.spaces.Box(
                low=np.array([-20.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0], dtype=np.float32),
                high=np.array([30.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0], dtype=np.float32),
                dtype=np.float32
            ),
            'reward': None,  # Se definirá en la configuración
            'reward_kwargs': {
                'temperature_variables': ['zone_air_temperature_1', 'zone_air_temperature_2', 
                                         'zone_air_temperature_3', 'zone_air_temperature_4'],
                'energy_variable': 'total_electric_demand',
                'range_comfort_winter': (20.0, 23.5),
                'range_comfort_summer': (23.0, 26.0),
                'lambda_energy': 5e-4,
                'lambda_temperature': 1.0
            },
            'env_name': 'Eplus-pyenv-multizona-production-v1',
            'config_params': {
                'timesteps_per_hour': 4,
                'runperiod': (1, 1, 12, 31),
                'action_definition': {}
            },
            'production_config': {
                'data_mode': 'terminal',
                'action_delay': 1.0,
                'max_steps_per_episode': 96,
                'safety_limits': {
                    'max_zone_temperature': 28.0,
                    'min_zone_temperature': 16.0
                }
            }
        }
    )
    
    print("✅ Entorno 'Eplus-pyenv-multizona-production-v1' registrado en Gymnasium")

# ============ REGISTRAR TAMBIÉN EN SINERGYM ============
try:
    from sinergym.utils.registry import registry
    
    # Solo si importamos correctamente
    if HAS_PYENV:
        # Crear una función de fábrica para Sinergym
        def create_production_env(**kwargs):
            return PyEnvProduction(**kwargs)
        
        # Registrar en el registro de Sinergym
        registry['Eplus-pyenv-multizona-production-v1'] = {
            'entry_point': create_production_env,
            'kwargs': {}  # Se llenará al crear el entorno
        }
        
        print("✅ Entorno registrado en Sinergym")
    else:
        print("⚠️  No se pudo registrar en Sinergym (PyEnvProduction no disponible)")
        
except ImportError:
    print("⚠️  Sinergym no disponible, solo registrado en Gymnasium")