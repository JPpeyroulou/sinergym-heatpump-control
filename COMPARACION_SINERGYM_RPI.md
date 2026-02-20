# Comparación: Sinergym (este repo) vs sinergym:arm64 (Raspberry Pi)

## Resumen

| Aspecto | **Este repo (workspace)** | **sinergym:arm64 (RPi)** |
|---------|---------------------------|---------------------------|
| **Sinergym** | 3.9.1 (`pyproject.toml`, `sinergym/version.txt`) | (probablemente mismo si se construyó desde este repo) |
| **EnergyPlus** | Documentado: **24.1.0** (`INSTALL.md`) | **25.2.0** |
| **Base OS** | Ubuntu 24.04 (según tabla INSTALL) | Ubuntu 24.04 |
| **Arquitectura** | amd64 (x86, típico en PC/DevContainer) | **arm64** (Raspberry Pi) |
| **Python** | ^3.12 (`pyproject.toml`) | (implícito en imagen, compatible 3.12) |
| **Comando por defecto** | `python scripts/try_env.py` | `python scripts/try_env.py` |
| **WorkingDir** | — | `/workspaces/sinergym` |
| **Tamaño imagen** | — | ~4.48 GB |

---

## Diferencias importantes

### 1. Versión de EnergyPlus
- **Repo:** `INSTALL.md` indica compatibilidad probada con **EnergyPlus 24.1.0**.
- **RPi:** La imagen usa **EnergyPlus 25.2.0** (binario NREL para Ubuntu 24.04 arm64).
- **Impacto:** Entornos y episodios deberían ser compatibles; 25.2.0 es una versión más nueva. Si en el repo usas solo APIs estándar de Sinergym, no suele haber problema.

### 2. Arquitectura (amd64 vs arm64)
- **Repo:** Se ejecuta en x86/amd64 (por ejemplo en Cursor/DevContainer o tu PC).
- **RPi:** La imagen está compilada para **arm64** (Raspberry Pi 4/5, etc.).
- **Impacto:** No puedes usar la misma imagen Docker en ambos: en la RPi debe usarse una imagen arm64 (como `sinergym:arm64`).

### 3. Dockerfile en el repo
- En el workspace **no hay** `Dockerfile` en la raíz ni en `.devcontainer` (o no está versionado).
- `INSTALL.md` describe un flujo `docker build -t <tag> .` y que el contenedor ejecuta por defecto `python scripts/try_env.py`.
- La imagen de la RPi fue construida con BuildKit (`buildkit.dockerfile.v0`); el contenido exacto del Dockerfile usado para esa imagen no está en este repo.

---

## Entorno de la imagen sinergym:arm64 (RPi)

Variables de entorno relevantes dentro del contenedor:

- `ENERGYPLUS_TAG=v25.2.0`
- `EPLUS_PATH=/usr/local/EnergyPlus-25-2-0`
- `PYTHONPATH=/usr/local/EnergyPlus-25-2-0`
- `PIP_BREAK_SYSTEM_PACKAGES=1`
- `POETRY_*` (Poetry configurado)
- `WANDB_API_KEY=` (vacío)
- `LC_ALL=C`

---

## Conclusión

- **Mismo flujo de uso:** ambos arrancan con `python scripts/try_env.py` y el repo tiene ese script.
- **Diferencias:** EnergyPlus **25.2.0** en la RPi vs **24.1.0** documentado en el repo; y **arm64** en la RPi vs **amd64** en tu entorno habitual.
- Para alinear versiones: o bien construís una imagen desde este repo con EnergyPlus 25.2.0 (y la documentación en `INSTALL.md` actualizada), o bien usás en la RPi una imagen basada en EnergyPlus 24.1.0 si querés coincidir exactamente con lo documentado.
