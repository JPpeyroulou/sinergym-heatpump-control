# 🔧 Solución: Ejecutar desde Docker

## Problema

Estás ejecutando el script desde dentro de un contenedor Docker, pero Home Assistant está en otro lugar (otro contenedor o el host). `localhost:8123` no funciona porque cada contenedor tiene su propio `localhost`.

## Soluciones

### Opción 1: Encontrar la URL automáticamente

Ejecuta el script que busca Home Assistant:

```bash
cd /workspaces/sinergym/scripts/train/prod
python3 test_find_homeassistant.py
```

Este script probará diferentes URLs comunes y te dirá cuál funciona.

### Opción 2: Usar host.docker.internal (Docker Desktop)

Si estás usando Docker Desktop en Windows/Mac:

```bash
python3 test_homeassistant_integration.py \
  --url http://host.docker.internal:8123 \
  --token "TU_TOKEN"
```

### Opción 3: Usar la IP del host (Linux)

Si estás en Linux, encuentra la IP del host:

```bash
# En el host (fuera del contenedor)
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
```

Luego usa esa IP:

```bash
# Dentro del contenedor
python3 test_homeassistant_integration.py \
  --url http://192.168.1.100:8123 \
  --token "TU_TOKEN"
```

### Opción 4: Usar el nombre del contenedor

Si Home Assistant está en otro contenedor Docker en la misma red:

```bash
# Ver contenedores
docker ps | grep homeassistant

# Usar el nombre del contenedor
python3 test_homeassistant_integration.py \
  --url http://homeassistant:8123 \
  --token "TU_TOKEN"
```

### Opción 5: Ejecutar desde el host (más simple)

Si Home Assistant está en el host, ejecuta el script desde el host (no desde el contenedor):

```bash
# En el host
cd /workspaces/sinergym/scripts/train/prod
python3 test_homeassistant_integration.py \
  --url http://localhost:8123 \
  --token "TU_TOKEN"
```

## Verificar dónde está Home Assistant

### Ver contenedores Docker:

```bash
docker ps
```

Busca un contenedor con "homeassistant" en el nombre.

### Ver redes Docker:

```bash
docker network ls
docker network inspect bridge
```

### Ver IP del contenedor de Home Assistant:

```bash
docker inspect <nombre_contenedor_homeassistant> | grep IPAddress
```

## Ejemplo completo

```bash
# 1. Encontrar Home Assistant
python3 test_find_homeassistant.py

# 2. Si encuentra la URL, usar ese resultado
python3 test_homeassistant_integration.py \
  --url <URL_ENCONTRADA> \
  --token "TU_TOKEN"
```
