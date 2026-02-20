"""Common utilities."""

import importlib
import importlib.util
import os
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import xlsxwriter
import yaml
from eppy.modeleditor import IDF
from gymnasium.envs.registration import registry

try:
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError:
    pass

import sinergym
from sinergym.utils.constants import LOG_COMMON_LEVEL
from sinergym.utils.logger import TerminalLogger

logger = TerminalLogger().getLogger(
    name='COMMON',
    level=LOG_COMMON_LEVEL)

# ---------------------------------------------------------------------------- #
#                                Dynamic imports                               #
# ---------------------------------------------------------------------------- #


def import_from_path(dotted_or_file_path: str):
    """
    Import a class or function from a dotted module path or a file path.

    Args:
        dotted_or_file_path (str): Either 'module:attr' or '/path/to/file.py:attr'

    Returns:
        The imported attribute (function, class, etc.)
    """
    if ':' not in dotted_or_file_path:
        raise ValueError(
            f"Invalid format: '{dotted_or_file_path}'. Expected format: 'module:attr' or 'file.py:attr'")

    path_part, attr_name = dotted_or_file_path.split(':', 1)

    if os.path.isfile(path_part):  # Es una ruta de archivo
        module_name = os.path.splitext(os.path.basename(path_part))[0]
        spec = importlib.util.spec_from_file_location(module_name, path_part)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from file: {path_part}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:  # Es un módulo en notación de puntos
        module = importlib.import_module(path_part)

    try:
        return getattr(module, attr_name)
    except AttributeError:
        raise ImportError(
            f"Module '{path_part}' does not have attribute '{attr_name}'")

# ---------------------------------------------------------------------------- #
#                            Dictionary deep update                            #
# ---------------------------------------------------------------------------- #


def deep_update(source: Dict, updates: Dict) -> Dict:
    """
    Recursively update a dictionary with another dictionary.

    Args:
        source (Dict): The original dictionary to update.
        updates (Dict): The dictionary with updates.

    Returns:
        Dict: The updated dictionary.
    """
    result = deepcopy(source)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

# ---------------------------------------------------------------------------- #
#                           ENVIRONMENT CONSTRUCTION                           #
# ---------------------------------------------------------------------------- #


def create_environment(
        env_id: str,
        env_params: Dict,
        wrappers: Dict,
        env_deep_update: bool = True) -> gym.Env:
    """Create a environment with the given parameters and wrappers.
    
    MODIFIED to support PyEnvProduction for real-time production.
    
    Args:
        env_id (str): Environment ID.
        env_params (Dict): Environment parameters to overwrite the environment ID defaults.
        wrappers (Dict): Wrappers to be applied to the environment.
        env_deep_update (bool): If True, the environment parameters will be deeply updated 
                               instead of overwritten. Defaults to True.
    Returns:
        gym.Env: The created environment.
    """

    # ========== PATCH PARA PyEnvProduction ==========
    # Detectar si es un entorno PyEnv de producción
    # Busca patrones como 'pyenv', 'production', 'real' en el ID
    is_pyenv = any(pattern in env_id.lower() for pattern in ['pyenv', 'production', 'real', 'online'])
    
    if is_pyenv:
        print(f"🔧 [PATCH] Detected PyEnv/Production environment: {env_id}")
        print(f"   Creating PyEnvProduction for real-time data")
        
        try:
            # Importar PyEnvProduction
            from sinergym.envs.pyenv_production import PyEnvProduction
            
            # Procesar action_space si viene como string
            if 'action_space' in env_params and isinstance(env_params['action_space'], str):
                import numpy as np
                import gymnasium as gym
                # Evaluar la string de action_space (ej: "gym.spaces.Box(...)")
                action_space_str = env_params['action_space']
                # IMPORTANTE: Esto usa eval, solo en entornos confiables
                env_params['action_space'] = eval(action_space_str)
                print(f"   Converted action_space from string to object")
            
            # Extraer configuración de producción
            production_config = {}
            if env_deep_update and 'production_config' in env_params:
                production_config = env_params.pop('production_config')
            elif 'production_settings' in env_params:
                production_config = env_params.pop('production_settings', {})
            
            # Asegurar que tenemos todos los parámetros necesarios
            required_params = {
                'building_file': env_params.get('building_file', 'dummy.epJSON'),
                'weather_files': env_params.get('weather_files', ['dummy.epw']),
                'variables': env_params.get('variables', {}),
                'meters': env_params.get('meters', {}),
                'actuators': env_params.get('actuators', {}),
                'action_space': env_params.get('action_space'),
                'reward': env_params.get('reward'),
                'reward_kwargs': env_params.get('reward_kwargs', {}),
                'env_name': env_params.get('env_name', env_id),
                'config_params': env_params.get('config_params', {}),
                'production_config': production_config,
            }
            
            # Verificar que tenemos variables
            if not required_params['variables']:
                print(f"⚠️  Warning: No variables defined. Using default.")
                required_params['variables'] = {'temperature': ('Temperature', 'C')}
            
            # Crear entorno PyEnvProduction DIRECTAMENTE (sin gym.make)
            env = PyEnvProduction(**required_params)
            
            # Aplicar wrappers
            if wrappers:
                # Asegurar que tenemos las funciones de wrappers
                if 'apply_wrappers_info' not in globals():
                    from sinergym.utils.wrappers import apply_wrappers_info, get_wrappers_info
                env = apply_wrappers_info(env, wrappers)
                get_wrappers_info(env)
            
            print(f"✅ PyEnvProduction created successfully")
            print(f"   Mode: {env.data_mode if hasattr(env, 'data_mode') else 'unknown'}")
            return env
            
        except Exception as e:
            print(f"❌ Error creating PyEnvProduction: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Falling back to standard gym.make()")
            # Continuar con el flujo normal como fallback
            pass
    # ========== END OF PATCH ==========

    # Original Sinergym code for EplusEnv (only if NOT PyEnv)
    environment = env_id

    # Make environment
    if env_deep_update:
        # Get default environment parameters associated to the environment ID
        env_default_kwargs = registry[environment].kwargs
        # Deeply update environment parameters with the given parameters
        env_params = deep_update(env_default_kwargs, env_params)

    # Make environment with the remaining parameters
    env = gym.make(environment, **env_params)

    # Apply wrappers
    if wrappers:
        # Apply wrappers to environment
        env = apply_wrappers_info(env, wrappers)
        # Write wrappers configuration to yaml file
        get_wrappers_info(env)

    return env

# ---------------------------------------------------------------------------- #
#                                   WRAPPERS                                   #
# ---------------------------------------------------------------------------- #


def is_wrapped(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    Args:
        env (gym.Env): Environment to check
        wrapper_class (Type[gym.Wrapper]): Wrapper class to look for

    Returns:
        bool: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_wrapper(env: gym.Env,
                   wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a wrapper object by recursively searching.

    Args:
        env (gym.Env): Environment to unwrap
        wrapper_class (Type[gym.Wrapper]): Wrapper to look for

    Returns:
        Optional[gym.Wrapper]: Wrapper object if found, else None
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp.env  # type: ignore
        env_tmp = env_tmp.env
    return None  # type: ignore


def get_wrappers_info(
        env: gym.Env, path_to_save: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get ordered information about the wrappers applied to the environment.

    Args:
        env (gym.Env): Environment to get wrapper information from.
        path_to_save (str, optional): Path to save the information in a YAML file. Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with wrapper module and class as keys and their arguments dict as values.
    """
    wrappers_info = []

    if not path_to_save:
        path_to_save = f'{
            env.get_wrapper_attr(
                name='workspace_path')}/wrappers_config.pyyaml'  # type: ignore

    # Traverse the wrappers and collect their metadata
    while isinstance(env, gym.Wrapper):
        wrapper_cls = env.__class__
        wrapper_name = f'{wrapper_cls.__module__}:{wrapper_cls.__name__}'
        if env.has_wrapper_attr('__metadata__'):
            wrappers_info.append(
                (wrapper_name, env.get_wrapper_attr(
                    name='__metadata__')))  # type: ignore
        env = env.env  # type: ignore

    # Reverse to get application order: outermost to innermost
    wrappers_info.reverse()

    # Convert to a regular dict (in insertion order)
    wrappers_dict = {name: metadata for name, metadata in wrappers_info}

    # Save to YAML
    if path_to_save:
        with open(path_to_save, 'w') as file:
            yaml.dump(
                wrappers_dict,
                file,
                sort_keys=False,
                default_flow_style=False)

    return wrappers_dict


def apply_wrappers_info(env: gym.Env,
                        wrappers_info: Union[Dict[str,
                                                  Dict[str,
                                                       Any]],
                                             str]) -> gym.Env:
    """Apply wrapper information to the environment.

    Args:
        env (gym.Env): Environment to apply wrapper information to.
        wrappers_info (Union[Dict[str, Dict[str, Any]], str]): Dictionary with wrapper information or path to a YAML file containing the information.

    Returns:
        gym.Env: Environment with applied wrappers.
    """

    if isinstance(wrappers_info, str):
        with open(wrappers_info, 'r') as file:
            wrappers_info_dict = yaml.load(file, Loader=yaml.FullLoader)
    else:
        wrappers_info_dict = wrappers_info

    for wrapper_class_name, wrapper_params in wrappers_info_dict.items():
        # Dynamically import the wrapper class
        wrapper_cls = import_from_path(wrapper_class_name)
        env = wrapper_cls(env, **wrapper_params)

    return env

# ---------------------------------------------------------------------------- #
#                               BUILDING MODELING                              #
# ---------------------------------------------------------------------------- #


def get_delta_seconds(
        st_year: int,
        st_mon: int,
        st_day: int,
        end_year: int,
        end_mon: int,
        end_day: int) -> float:
    """Returns the delta seconds between st year:st mon:st day:0:0:0 and
    end year:end mon:end day:24:0:0.

    Args:
        st_year (int): Start year.
        st_mon (int): Start month.
        st_day (int): Start day.
        end_year (int): End year.
        end_mon (int): End month.
        end_day (int): End day.

    Returns:
        float: Time difference in seconds.

    """
    start_time = datetime(st_year, st_mon, st_day)
    end_time = datetime(end_year, end_mon, end_day) + timedelta(days=1)
    return (end_time - start_time).total_seconds()


def eppy_element_to_dict(element: IDF) -> Dict[str, Dict[str, str]]:
    """Converts an eppy element into a dictionary following the EnergyPlus epJSON standard.

    Args:
        element (IDF): eppy element to be converted.

    Returns:
        Dict[str,Dict[str,str]]: Python dictionary with epJSON format of eppy element.
    """
    fields = {
        fieldname.lower().replace('drybulb', 'dry_bulb'):
        'WetBulb' if (value := element[fieldname]  # type: ignore
                      ) == 'Wetbulb' else value
        for fieldname in element.fieldnames  # type: ignore
        if fieldname not in {'Name', 'key'
                             } and element[fieldname] != ''  # type: ignore
    }

    return {getattr(element, 'Name', '').lower(): fields}


def export_schedulers_to_excel(
        schedulers: Dict[str, Dict[str, Union[str, Dict[str, str]]]], path: str) -> None:  # pragma: no cover
    """Exports scheduler information from a dictionary to an Excel file.

    Args:
        schedulers (Dict[str, Dict[str, Union[str, Dict[str, str]]]]): Dictionary with the correct format.
        path (str): Relative path where the Excel file will be created.
    """

    # Creating workbook and sheet
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()

    # Creating cell format configuration
    keys_format = workbook.add_format(
        {'bold': True, 'font_size': 20, 'align': 'center', 'bg_color': 'gray', 'border': True})
    cells_format = workbook.add_format({'align': 'center'})
    actuator_format = workbook.add_format(
        {'bold': True, 'align': 'center', 'bg_color': 'gray'})

    # Headers
    worksheet.write(0, 0, 'Name', keys_format)
    worksheet.write(0, 1, 'Type', keys_format)

    current_row = 1
    max_col = 1

    for key, info in schedulers.items():
        worksheet.write(current_row, 0, key, actuator_format)
        worksheet.write(
            current_row, 1, info.get(
                'Type', 'Unknown'), cells_format)

        col_offset = 2  # Offset inicial después de 'Type'
        for object_name, values in info.items():
            if isinstance(values, dict):
                worksheet.write(
                    current_row,
                    col_offset,
                    f'Name: {object_name}')
                worksheet.write(
                    current_row,
                    col_offset + 1,
                    f'Field: {
                        values.get(
                            "field_name",
                            "N/A")}')
                worksheet.write(
                    current_row,
                    col_offset + 2,
                    f'Table type: {
                        values.get(
                            "table_name",
                            "N/A")}')
                col_offset += 3  # Avanzar columnas según los datos escritos

        max_col = max(max_col, col_offset)
        current_row += 1

    # Adjusting column width
    worksheet.set_column(0, max_col, 40)

    # Add object columns
    for i, col in enumerate(range(2, max_col, 3), start=1):
        worksheet.merge_range(0, col, 0, col + 2, f'OBJECT {i}', keys_format)

    workbook.close()

# ---------------------------------------------------------------------------- #
#                          ORNSTEIN UHLENBECK PROCCESS                         #
# ---------------------------------------------------------------------------- #


def ornstein_uhlenbeck_process(
        data: pd.DataFrame,
        variability_config: Dict[str, Tuple[float, float, float]]) -> pd.DataFrame:
    """Add noise to the data using the Ornstein-Uhlenbeck process.

    Args:
        data (pd.DataFrame): Data to be modified.
        variability_config (Dict[str, Tuple[float, float, float]]): Dictionary with the variability configuration for each variable (sigma, mu and tau constants).

    Returns:
        pd.DataFrame: Data with noise added.
    """

    # deepcopy for weather_data
    data_mod = deepcopy(data)

    # Total time.
    T = 1.
    # get first column of df
    n = data_mod.shape[0]
    # dt = T / n  # tau defined as percentage of epw
    dt = T / 1.0  # tau defined as epw rows (hours)
    # t = np.linspace(0., T, n)  # Vector of times.

    # Sigma = standard deviation.
    # Mu = mean.
    # Tau = time constant.

    for variable, (sigma, mu, tau) in variability_config.items():
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        # Create noise
        noise = np.zeros(n)
        for i in range(n - 1):
            noise[i + 1] = noise[i] + dt * (-(noise[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()

        # Add noise
        data_mod[variable] += noise

    return data_mod

# ---------------------------------------------------------------------------- #
#                    READING YAML ENVIRONMENT CONFIGURATION                    #
# ---------------------------------------------------------------------------- #


def parse_variables_settings(
        variables: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Convert Sinergym YAML variable settings to EnergyPlus API format.

    Args:
        variables (Dict[str, Any]): Dictionary from Sinergym YAML configuration.

    Returns:
        Dict[str, Tuple[str, str]]: Dictionary adapted for EnergyPlus API.
    """

    output = {}

    for variable, specification in variables.items():
        var_names = specification['variable_names']
        keys = specification['keys']

        if isinstance(var_names, str) and isinstance(keys, str):
            output[var_names] = (variable, keys)

        elif isinstance(var_names, str) and isinstance(keys, list):
            for key in keys:
                prename = key.lower().replace(' ', '_') + '_'
                output[prename + var_names] = (variable, key)

        elif isinstance(var_names, list) and isinstance(keys, list):
            if len(var_names) != len(keys):
                raise ValueError(
                    f"'variable_names' and 'keys' must have the same length in {variable}")
            for var_name, key in zip(var_names, keys):
                output[var_name] = (variable, key)

        else:
            logger.error(
                f'Invalid variable_names or keys format in {variable}')
            raise RuntimeError

    return output


def parse_meters_settings(meters: Dict[str, str]) -> Dict[str, str]:
    """Convert meters dictionary from Sinergym YAML settings to EnergyPlus format.

    Args:
        meters (Dict[str, str]): Dictionary with meters information.

    Returns:
        Dict[str, str]: Reformatted meters dictionary for EnergyPlus API.
    """
    return {v: k for k, v in meters.items()}


def parse_actuators_settings(
        actuators: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[str, str, str]]:
    """Convert actuators dictionary from Sinergym YAML settings to EnergyPlus format.

    Args:
        actuators (Dict[str, Dict[str, str]]): Actuators information from Sinergym YAML.

    Returns:
        Dict[str, Tuple[str, str, str]]: Reformatted actuators dictionary for EnergyPlus API.
    """
    return {
        spec['variable_name']: (spec['element_type'], spec['value_type'], name)
        for name, spec in actuators.items()
    }


def convert_conf_to_env_parameters(
        conf: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert YAML configuration to a dictionary of possible environments.

    Args:
        conf (Dict[str, Any]): Dictionary from a YAML configuration file.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with environment constructor kwargs.
    """

    configurations = {}

    variables = parse_variables_settings(conf['variables'])
    meters = parse_meters_settings(conf['meters'])
    actuators = parse_actuators_settings(conf['actuators'])
    context = parse_actuators_settings(conf['context'])

    weather_keys = conf['weather_specification']['keys']
    weather_files = conf['weather_specification']['weather_files']

    # Check weathers configuration
    if len(weather_keys) != len(weather_files):
        logger.error(
            f'Weather files and id keys must have the same length: ' f'{
                len(weather_files)} weather files != {
                len(weather_keys)} keys')
        raise ValueError

    # Base ID and kwargs
    base_id = 'Eplus-' + conf['id_base']
    base_kwargs = {
        'building_file': conf['building_file'],
        'action_space': eval(conf['action_space']),
        'time_variables': conf['time_variables'],
        'variables': variables,
        'meters': meters,
        'actuators': actuators,
        'context': context,
        'initial_context': conf.get('initial_context'),
        'reward': import_from_path(conf['reward']),
        'reward_kwargs': conf['reward_kwargs'],
        'max_ep_store': conf['max_ep_store'],
        'building_config': conf.get('building_config')
    }

    weather_variability = conf.get('weather_variability')
    # Convert lists to tuples for consistency
    if weather_variability:
        weather_variability = {
            var: tuple(tuple(param) if isinstance(param, list) else param
                       for param in params)
            for var, params in weather_variability.items()
        }

    # Build environment configurations
    for weather_id, weather_file in zip(weather_keys, weather_files):
        configurations[f'{base_id}-{weather_id}-continuous-v1'] = {**base_kwargs,
                                                                   'weather_files': weather_file,
                                                                   'env_name': f'{base_id}-{weather_id}-continuous-v1'}
        # Build stochastic versions if weather variability is present
        if weather_variability:
            configurations[f'{base_id}-{weather_id}-continuous-stochastic-v1'] = {
                **base_kwargs, 'weather_files': weather_file, 'weather_variability': weather_variability,
                'env_name': f'{base_id}-{weather_id}-continuous-stochastic-v1'
            }

    return configurations

# ---------------------------------------------------------------------------- #
#                          Process YAML configurations                         #
# ---------------------------------------------------------------------------- #


def process_environment_parameters(env_params: dict) -> dict:  # pragma: no cover
    # Transform required str's into Callables or lists in tuples
    if env_params.get('action_space'):
        env_params['action_space'] = eval(
            env_params['action_space'])

    if env_params.get('variables'):
        for variable_name, components in env_params['variables'].items():
            env_params['variables'][variable_name] = tuple(components)

    if env_params.get('actuators'):
        for actuator_name, components in env_params['actuators'].items():
            env_params['actuators'][actuator_name] = tuple(components)

    if env_params.get('weather_variability'):
        env_params['weather_variability'] = {
            var_name: tuple(
                tuple(param) if isinstance(param, list) else param
                for param in var_params
            )
            for var_name, var_params in env_params['weather_variability'].items()
        }

    if env_params.get('reward'):
        env_params['reward'] = import_from_path(env_params['reward'])

    if env_params.get('reward_kwargs'):
        for reward_param_name, reward_value in env_params.items():
            if reward_param_name in [
                'range_comfort_winter',
                'range_comfort_summer',
                'summer_start',
                    'summer_final']:
                env_params['reward_kwargs'][reward_param_name] = tuple(
                    reward_value)

    if env_params.get('building_config'):
        if env_params['building_config'].get('runperiod'):
            env_params['building_config']['runperiod'] = tuple(
                env_params['building_config']['runperiod'])
    # Add more keys if needed...

    return env_params


def process_algorithm_parameters(alg_params: dict):  # pragma: no cover

    # Transform required str's into Callables or list in tuples
    if alg_params.get('train_freq') and isinstance(
            alg_params.get('train_freq'), list):
        alg_params['train_freq'] = tuple(alg_params['train_freq'])

    if alg_params.get('action_noise'):
        alg_params['action_noise'] = eval(alg_params['action_noise'])
    # Add more keys if needed...

    return alg_params

 # --------------------- Functions that are no longer used -------------------- #

# def ranges_getter(output_path: str,
#                   last_result: Optional[Dict[str, List[float]]] = None
#                   ) -> Dict[str, List[float]]:  # pragma: no cover
#     """Given a path with simulations outputs, this function is used to extract max and min absolute values of all episodes in each variable. If a dict ranges is given, will be updated.

#     Args:
#         output_path (str): Path with simulation output (Eplus-env-<env_name>).
# last_result (Optional[Dict[str, List[float]]], optional): Last ranges
# dict to be updated. This will be created if it is not given.

#     Returns:
#         Dict[str, List[float]]: list min,max of each variable as a key.

#     """

#     if last_result is not None:
#         result = last_result
#     else:
#         result = {}

#     content = os.listdir(output_path)
#     for episode_path in content:
#         if os.path.isdir(
#             output_path +
#             '/' +
#                 episode_path) and episode_path.startswith('Eplus-env'):
#             simulation_content = os.listdir(output_path + '/' + episode_path)

#             if os.path.isdir(
#                 output_path +
#                 '/' +
#                     episode_path):
#                 monitor_path = output_path + '/' + episode_path + '/monitor.csv'
#                 print('Reading ' + monitor_path + ' limits.')
#                 data = pd.read_csv(monitor_path)

#                 if len(result) == 0:
#                     for column in data:
#                         # variable : [min,max]
#                         result[column] = [np.inf, -np.inf]

#                 for column in data:
#                     if np.min(data[column]) < result[column][0]:
#                         result[column][0] = np.min(data[column])
#                     if np.max(data[column]) > result[column][1]:
#                         result[column][1] = np.max(data[column])
#     return result
