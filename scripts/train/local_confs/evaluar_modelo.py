#!/usr/bin/env python3
"""
Evaluar un modelo entrenado corriendo 1 episodio completo (deterministico).
Genera la carpeta con observations.csv, infos.csv, etc. que luego se usan
para las graficas.

Uso:
  python evaluar_modelo.py <ruta_model.zip>

Ejemplo:
  python evaluar_modelo.py Eplus-SAC-training-.../model.zip
"""

import sys
import os
import numpy as np
import yaml
import zipfile
import json

import sinergym
from sinergym.utils.common import create_environment, import_from_path
from sinergym.utils.wrappers import NormalizeAction, NormalizeObservation, LoggerWrapper, CSVLogger
from stable_baselines3 import SAC, PPO

# ============================================================================
# CONFIGURACION
# ============================================================================
ENV_ID = 'Eplus-nuestroMultizona-uru-continuous-v1'
EXPERIMENT_NAME = 'Evaluacion-modelo-final'


def detect_algorithm(model_path):
    """Detectar si el modelo es SAC o PPO leyendo data del zip."""
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            with zf.open('data') as f:
                data = json.loads(f.read())
                cls_name = data.get('policy_class', '')
                if 'SAC' in str(cls_name) or 'sac' in str(cls_name):
                    return SAC
                elif 'ActorCritic' in str(cls_name) or 'PPO' in str(cls_name) or 'ppo' in str(cls_name):
                    return PPO
    except Exception:
        pass
    # Fallback: intentar cargar como SAC, si falla PPO
    try:
        SAC.load(model_path, device='cpu')
        return SAC
    except Exception:
        return PPO


def main():
    if len(sys.argv) < 2:
        print("Uso: python evaluar_modelo.py <ruta_model.zip>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado: {model_path}")
        sys.exit(1)

    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Buscar archivos de normalizacion junto al modelo
    mean_path = os.path.join(model_dir, 'mean.txt')
    var_path = os.path.join(model_dir, 'var.txt')
    count_path = os.path.join(model_dir, 'count.txt')
    has_norm = os.path.exists(mean_path) and os.path.exists(var_path)

    print("=" * 70)
    print("EVALUACION DE MODELO ENTRENADO")
    print("=" * 70)
    print(f"  Modelo: {model_path}")
    print(f"  Entorno: {ENV_ID}")
    print(f"  Normalizacion: {'Si (' + mean_path + ')' if has_norm else 'No encontrada'}")

    # 1. Crear entorno con los mismos wrappers que en entrenamiento
    wrappers = {
        'sinergym.utils.wrappers:NormalizeAction': {},
        'sinergym.utils.wrappers:NormalizeObservation': {},
        'sinergym.utils.wrappers:LoggerWrapper': {
            'storage_class': import_from_path('sinergym.utils.logger:LoggerStorage')
        },
        'sinergym.utils.wrappers:CSVLogger': {},
    }

    env = create_environment(
        env_id=ENV_ID,
        env_params={'env_name': EXPERIMENT_NAME},
        wrappers=wrappers,
        env_deep_update=True
    )
    print(f"\nEntorno creado: {ENV_ID}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")

    # 2. Restaurar calibracion de normalizacion si existe
    if has_norm:
        try:
            mean = np.loadtxt(mean_path)
            var = np.loadtxt(var_path)
            count = float(np.loadtxt(count_path)) if os.path.exists(count_path) else 1e4

            # Buscar el wrapper NormalizeObservation y setear mean/var
            env_tmp = env
            while env_tmp is not None:
                if isinstance(env_tmp, NormalizeObservation):
                    env_tmp.mean = mean
                    env_tmp.var = var
                    env_tmp.count = count
                    env_tmp.deactivate_update()  # No actualizar durante evaluacion
                    print(f"   Normalizacion restaurada (count={count:.0f})")
                    break
                env_tmp = getattr(env_tmp, 'env', None)
        except Exception as e:
            print(f"   Error restaurando normalizacion: {e}")

    # 3. Cargar modelo (detectar SAC vs PPO automaticamente)
    print(f"\nCargando modelo...")
    AlgClass = detect_algorithm(model_path)
    print(f"   Algoritmo detectado: {AlgClass.__name__}")
    model = AlgClass.load(model_path, device='cpu')
    model.set_env(env)
    print(f"   Modelo cargado ({AlgClass.__name__})")

    # 4. Correr 1 episodio completo deterministico
    print(f"\nCorriendo episodio de evaluacion (deterministico)...")
    obs, info = env.reset()

    episode_reward = 0.0
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1

        if step % 1000 == 0:
            print(f"   Paso {step:,d} / ~8,760 | Reward acum: {episode_reward:.1f}")

    print(f"\nEpisodio completado:")
    print(f"   Pasos: {step:,d}")
    print(f"   Reward total: {episode_reward:.2f}")
    print(f"   Reward promedio: {episode_reward / step:.4f}")

    # 5. Obtener ruta del workspace donde quedaron los CSVs
    workspace = env.get_wrapper_attr('workspace_path')
    print(f"\nResultados guardados en: {workspace}")

    # Listar lo que hay
    for root, dirs, files in os.walk(workspace):
        level = root.replace(workspace, '').count(os.sep)
        indent = '   ' + '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '   ' + '  ' * (level + 1)
        for file in files:
            size = os.path.getsize(os.path.join(root, file))
            print(f"{subindent}{file} ({size / 1024:.1f} KB)")

    env.close()

    # 6. Encontrar el observations.csv
    obs_csv = None
    for root, dirs, files in os.walk(workspace):
        if 'observations.csv' in files:
            obs_csv = os.path.join(root, 'observations.csv')
            break

    if obs_csv:
        print(f"\nPara generar graficas ejecuta:")
        print(f"   python generar_graficas.py \"{obs_csv}\"")
    else:
        print(f"\nNo se encontro observations.csv en {workspace}")

    print(f"\nEvaluacion finalizada")


if __name__ == '__main__':
    main()
