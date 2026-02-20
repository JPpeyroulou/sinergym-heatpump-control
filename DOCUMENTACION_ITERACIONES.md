# Documentacion de Iteraciones: RL vs ON/OFF Thermostat

## Objetivo
Entrenar un modelo RL (SAC/PPO) para control de climatizacion del edificio multizona (`idf_multiplesZonas_Bomba2.epJSON`) que supere al control ON/OFF con termostato (`idf_multiplesZonas_termostato.idf`) en **costo energetico anual** manteniendo al menos el mismo **confort termico** (PMV).

## Targets a Superar
| Metrica | Valor ON/OFF | Condicion |
|---|---|---|
| Costo anual | $67,344 | < $67,344 |
| Confort East (PMV [-0.5, 0.5]) | 43.6% | >= 43.6% |
| Confort West (PMV [-0.5, 0.5]) | 43.2% | >= 43.2% |

## Baseline ON/OFF (Termostato)
- **Archivo IDF**: `sinergym/data/buildings/20260211/idf_multiplesZonas_termostato.idf`
- **Weather**: `URY_Montevideo.epw`
- **Resultados** (simulacion EnergyPlus 1 anio):
  - Consumo: **11,650 kWh**
  - Costo anual: **$67,344** (tarifa variable: peak L-V 17-20h $14.493/kWh, off-peak $4.556/kWh)
  - Peak: 1,436 kWh (12.3%), Off-peak: 10,214 kWh (87.7%)
  - Confort East: **43.6%** (horas en PMV [-0.5, 0.5])
  - Confort West: **43.2%**
  - PMV medio: East=-0.41, West=-0.42
  - Bomba ON en peak: **35.3%**, ON en off-peak: **37.0%**

## Nota sobre Unidades
- `total_electricity_HVAC` de EnergyPlus esta en **Joules** (Output:Meter).
- Conversion: **kWh = J / 3,600,000**
- Tarifas: peak (L-V 17-20h) = **$14.493/kWh**, off-peak = **$4.556/kWh**
- Confort medido como % de horas con PMV en [-0.5, 0.5] usando formula simplificada:
  `PMV = -7.4928 + 0.2882 * Tdb - 0.0020 * RH + 0.0004 * Tdb * RH`

## Archivos Modificados

### 1. `sinergym/data/default_configuration/nuestroMultrizona.yaml`
**Parametros del reward que se iteraron:**
- `energy_weight`: Se mantuvo en **0.5** (por pedido del usuario)
- `lambda_energy`: Se probo 0.5, 1.0, 2.0, 2.1, 2.5, 3.0, 3.5, 4.0
- `lambda_temperature`: Se probo 28, 30, 40, 50, 100
- `high_price` / `low_price`: Precios reales 14.493/4.556 (tambien se probo 8.0/5.0)
- `action_space high`: Se probo 45 (default) y 35

### 2. `sinergym/utils/rewards.py` - Funcion `NuestroRewardMultizona`
**Componentes iterados:**
- **Waste factor**: Tipo (cliff/cuadratico), coeficiente (5-200), threshold PMV (-0.5, -0.3, 0.0)
- **Dynamic weights**: W_energy variable segun PMV (probado y descartado)
- **Comfort bonus**: Recompensa positiva por PMV en [-0.5, 0.5] (+2, +15, +30, +50 por zona)
- **Peak multiplier**: Multiplicador de penalizacion en horario pico (2x, 3x)
- **Reward Shaping Temporal**: Penalizacion varianza PMV, bonus pre-heat, penalizacion cycling (probado y descartado)

### 3. Training configs
- `train_agent_SAC_nuestroMultizona.yaml`: episodes 10, 15, 20, 40
- `train_agent_PPO_nuestroMultizona.yaml`: episodes 20, n_steps 2048-8192

---

## Tabla Completa de Iteraciones

### Fase 1: Tuning basico (C5-C22) - SAC

| # | Eval | Algo | le | lt | Waste | WF Thr | Dyn W | Bonus | Peak Mult | Ep | kWh | Costo | East% | West% | Notas |
|---|------|------|----|----|-------|--------|-------|-------|-----------|-----|------|-------|-------|-------|-------|
| **BL** | **-** | **ON/OFF** | - | - | - | - | - | - | - | - | **11,650** | **$67,344** | **43.6%** | **43.2%** | **BASELINE** |
| C5 | res8 | SAC | 2.0 | 30 | cliff 100x | PMV>=0 | No | - | - | 10 | 7,931 | $41,796 | 32.6% | 22.4% | Costo OK, conf baja |
| C6 | res9 | SAC | 1.0 | 30 | cliff 100x | PMV>=0 | No | - | - | 10 | 7,751 | $46,049 | 22.3% | 21.1% | Conf muy baja |
| C7 | res10 | SAC | 1.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 22,013 | $127,736 | 58.8% | 70.9% | Conf alta, muy caro |
| **C8** | **res11** | **SAC** | **2.0** | **30** | **quad 50** | **PMV>-0.5** | **No** | **-** | **-** | **10** | **14,611** | **$79,021** | **44.1%** | **44.3%** | **Mejor conf Fase 1** |
| C9 | res12 | SAC | 3.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 8,654 | $49,601 | 26.1% | 23.8% | Costo OK, conf baja |
| C10 | res13 | SAC | 2.0 | 30 | quad 100 | PMV>-0.5 | No | - | - | 10 | 12,528 | $69,145 | 30.4% | 35.3% | Costo ~BL, conf baja |
| C11 | res14 | SAC | 2.0 | 28 | quad 60 | PMV>-0.5 | No | - | - | 10 | 6,408 | $35,024 | 23.9% | 18.2% | Demasiado conservador |
| C12 | res15 | SAC | 2.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 20 | 10,926 | $60,514 | 20.7% | 41.6% | Sobre-entrenado |
| C13 | res16 | SAC | 2.0 | 30 | quad 50 min | PMV>-0.5 | No | - | - | 10 | 5,002 | $25,504 | 23.2% | 18.1% | min_pmv no funciono |
| C14 | res18 | SAC | 2.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 4,264 | $23,536 | 20.8% | 18.8% | Precios reducidos, fallo |
| C15 | res19 | SAC | 0.5 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 22,922 | $135,040 | 64.4% | 60.6% | le bajo = caro |
| C16 | res20 | SAC | 2.5 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 10,147 | $57,641 | 29.1% | 23.0% | Conf baja |
| C17 | res21 | SAC | 2.1 | 30 | quad 50 | PMV>-0.5 | burst | - | - | 10 | 14,405 | $78,195 | 37.9% | 52.3% | Burst penalty parcial |
| C18 | res22 | SAC | 2.0 | 30 | quad 50 | PMV>-0.5 | Si | - | - | 10 | 11,565 | $67,593 | 30.6% | 35.5% | Mejor costo Fase 1 |
| C19 | res23 | SAC | 2.0 | 30 | quad 50 | PMV>0.0 | Si | - | - | 10 | 17,518 | $103,831 | 54.5% | 66.4% | Conf alta, caro |
| C20 | res24 | SAC | 2.0 | 50 | quad 50 | PMV>-0.5 | Si | - | - | 10 | 17,865 | $99,944 | 38.7% | 47.1% | Parcial |
| C21 | res25 | SAC | 2.0 | 40 | quad 50 | PMV>-0.3 | Si | - | - | 10 | 16,615 | $98,298 | 48.2% | 50.6% | Conf OK, costo alto |
| C22 | res26 | SAC | 2.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 15 | 8,668 | $47,745 | 24.1% | 20.9% | Sobre-entrenado |

### Fase 2: Action space + PPO (C23-C34)

| # | Eval | Algo | le | lt | Waste | WF Thr | Dyn W | Bonus | Peak Mult | Ep | kWh | Costo | East% | West% | Notas |
|---|------|------|----|----|-------|--------|-------|-------|-----------|-----|------|-------|-------|-------|-------|
| C23 | res27 | SAC | 2.0 | 30 | quad 50 | PMV>-0.5 | No | - | - | 10 | 5,993 | $32,356 | 22.6% | 19.2% | Action lim 35C: demasiado conserv |
| C24-C28 | res31* | PPO | var | 30 | quad 50 | PMV>-0.5 | No | - | - | 20 | 20,475 | $121,747 | 63.8% | 72.6% | PPO calienta agresivamente |
| C29 | - | PPO | 2.0 | 30 | quad 50 | PMV>-0.5 | Dyn | - | - | 20 | - | - | - | - | Pesos dinamicos, PPO aun mas caro |
| C30 | - | PPO | 2.0 | 30 | quad 50 | PMV>-0.5 | Suave | - | - | 20 | - | - | - | - | Pesos dinamicos suaves, sin mejora |
| C31 | - | PPO | 2.0 | 30 | quad 200 | PMV>-0.5 | No | - | - | 20 | - | - | - | - | WF=200 sin impacto en PPO |
| C32 | - | SAC | 2.0 | 30 | quad 200 | PMV>-0.5 | No | - | - | 40 | - | - | - | - | SAC 40ep: sobre-optimizo energia |
| C33 | - | PPO | 2.5 | 30 | quad 50 | PMV>-0.5 | No | - | - | 20 | - | - | - | - | PPO n_steps=8192, sin breakthrough |
| C34 | - | SAC | 2.0 | 100 | quad 50 | PMV>-0.5 | No | - | - | 20 | - | - | - | - | lt=100: SAC sobrecalento |

*\*res31-34 son evaluaciones del mismo modelo PPO, resultados identicos*

### Fase 3: Comfort bonus + waste factor tuning (C35-C41)

| # | Eval | Algo | le | lt | WF coef | WF Thr | Bonus | Peak Mult | Ep | kWh | Costo | East% | West% | Notas |
|---|------|------|----|----|---------|--------|-------|-----------|-----|------|-------|-------|-------|-------|
| C35 | res35 | SAC | 2.0 | 30 | 50 | PMV>0.0 | - | - | 10 | 7,568 | $42,401 | 20.6% | 18.4% | Threshold alto: modelo no calienta |
| C36 | res36 | SAC | 2.0 | 30 | 50 | PMV>-0.5 | +2/zona | - | 10 | 17,267 | $100,551 | 38.9% | 57.4% | Bonus inicial, mejora West |
| C37 | res37 | SAC | 2.0 | 30 | 50 | PMV>-0.5 | +15/zona | - | 10 | 7,548 | $36,201 | 29.2% | 18.3% | Bonus +15, resultado inconsistente |
| C38 | res38 | SAC | 2.5 | 30 | 50 | PMV>-0.5 | +15/zona | - | 10 | 15,632 | $87,891 | 31.7% | 43.6% | le=2.5, West casi en target |
| C39 | res50 | SAC | 2.0 | 30 | 100 | PMV>-0.5 | +30/zona | - | 10 | 20,786 | $123,019 | 43.8% | 69.0% | **E supera BL! W excepcional. Caro** |
| C40 | res49 | PPO | 2.0 | 30 | 100 | PMV>-0.5 | +30/zona | - | 20 | 19,781 | $117,418 | 36.5% | 58.6% | PPO version de C39, peor East |
| C41 | res51 | SAC | 3.0 | 30 | 100 | PMV>-0.5 | +30/zona | - | 10 | 8,273 | $43,788 | 21.0% | 30.3% | le=3.0 overshot, conf colapso |

### Fase 4: Peak multiplier (C42-C45)

| # | Eval | Algo | le | lt | WF coef | Bonus | Peak Mult | Ep | kWh | Costo | East% | West% | ON pk% | ON off% | Notas |
|---|------|------|----|----|---------|-------|-----------|-----|------|-------|-------|-------|--------|---------|-------|
| C42 | res53 | SAC | 2.0 | 30 | 100 | +30 | 3x | 10 | 11,139 | $56,940 | 27.0% | 37.9% | 16.0% | 35.1% | **Costo < BL!** Shift peak exitoso |
| C43 | res52 | PPO | 2.0 | 30 | 100 | +30 | 3x | 20 | 19,614 | $113,318 | 52.0% | 38.1% | 48.1% | 59.9% | PPO sigue caro |
| C44 | res54 | SAC | 2.0 | 30 | 100 | +50 | 3x | 10 | 13,230 | $73,412 | **43.5%** | 26.3% | 30.6% | 40.4% | **East a 0.1% del BL!** |
| C45 | res45 | SAC | 2.0 | 30 | 100 | +50 | 2x | 10 | 14,163 | $80,548 | 35.2% | 31.6% | 40.0% | 41.7% | Peak mult reducido, sin mejora |

### Fase 5: Reward Shaping Temporal (C46-C47) - DESCARTADO

| # | Eval | Algo | le | lt | WF | Bonus | PMult | Temporal | Ep | kWh | Costo | East% | West% | Notas |
|---|------|------|----|----|-----|-------|-------|----------|-----|------|-------|-------|-------|-------|
| C46 | res55 | SAC | 2.0 | 30 | 100 | +50 | 2x | stab=2, pre=5, peak=15, cyc=1 | 10 | 13,355 | $74,379 | 35.3% | **46.0%** | W supera BL, E cae |
| C47 | res56 | SAC | 2.0 | 30 | 100 | +50 | 2x | stab=8, pre=10, peak=25, cyc=5 | 10 | 8,287 | $45,403 | 27.5% | 24.3% | Coefs altos: modelo deja de calentar |

#### Descripcion del Enfoque: Reward Shaping Temporal

El problema que se buscaba resolver era que el agente RL operaba con **rafagas de potencia maxima** seguidas de periodos sin calefaccion, en lugar de la operacion **moderada y sostenida** del ON/OFF. Esto es termodinamicamente ineficiente y genera picos/valles de temperatura. La idea fue agregar 3 componentes de reward que consideren el **comportamiento a lo largo del tiempo**, no solo el timestep actual.

**Componente 1 - Penalizacion por inestabilidad termica:**
Se guardo un historial de PMV promedio (ultimas 6 horas) y se penalizo la varianza. La logica: el ON/OFF mantiene temperatura estable (varianza ~0.01-0.05), mientras que el RL con rafagas genera alta varianza (~0.1-0.5).

```python
# En __init__:
self.pmv_history: List[float] = []
self.thermal_stability_coef = 2.0  # C46: 2.0, C47: 8.0

# En __call__:
self.pmv_history.append(avg_pmv)
if len(self.pmv_history) > 6:
    self.pmv_history.pop(0)
if len(self.pmv_history) >= 3:
    pmv_var = np.var(self.pmv_history)
    stability_penalty = -thermal_stability_coef * pmv_var
```

**Componente 2 - Bonus por estrategia de pre-calentamiento:**
Se otorgaba bonus por calentar en horas pre-peak (15-16h, off-peak) para que el edificio acumulara inercia termica. Ademas, bonus grande si el agente NO calentaba durante peak (17-20h) Y mantenia confort (PMV >= -0.5), demostrando que logro confort "gratis" aprovechando la inercia.

```python
# Pre-peak (15-16h): bonus por calentar anticipadamente
if is_prepeak and is_on:
    preheat_bonus += preheat_bonus_coef  # C46: 5.0, C47: 10.0

# Peak (17-20h): bonus por mantener confort SIN gastar
if is_peak and (not is_on) and avg_pmv >= -0.5:
    preheat_bonus += peak_comfort_bonus_coef  # C46: 15.0, C47: 25.0
```

**Componente 3 - Penalizacion por cycling (transiciones ON/OFF):**
Cada vez que la bomba cambiaba de estado (encendido→apagado o viceversa), se aplicaba una penalizacion. Esto incentivaba operacion sostenida, similar al ON/OFF con histeresis.

```python
self.prev_was_on: bool = False

# En __call__:
is_on = total_energy > 50.0  # >50W = bomba encendida
if is_on != self.prev_was_on:
    cycling_penalty = -cycling_penalty_coef  # C46: 1.0, C47: 5.0
self.prev_was_on = is_on
```

#### Resultados y Por Que Fallo

**C46 (coeficientes conservadores):** Resultados mixtos. West mejoro a 46.0% (supero baseline), pero East cayo a 35.3%. El modelo aprendio parcialmente la estrategia de pre-heat pero desbalanceo las zonas.

**C47 (coeficientes agresivos):** Fallo catastrofico. El modelo simplemente dejo de calentar (PMV -0.95 en ambas zonas, confort 27%/24%).

**Analisis de la falla - Soluciones Degeneradas:**

El problema fundamental es que las 3 penalizaciones temporales tienen **equilibrios degenerados** que el optimizador de RL encuentra facilmente:

1. **Penalizacion por cycling → No calentar**: La forma mas facil de evitar transiciones ON/OFF es mantener la bomba SIEMPRE apagada. El agente descubre que `0 transiciones = 0 penalizacion`, en lugar de aprender a calentar de forma sostenida.

2. **Penalizacion por varianza PMV → Frio estable**: La varianza mas baja de PMV se logra manteniendo el edificio a temperatura constante... por debajo de la zona de confort. Sin calefaccion, el PMV se estabiliza en ~-1.0 (frio pero estable). El agente logra `varianza ≈ 0` sin esforzarse.

3. **Bonus de pre-heat insuficiente**: Los bonus por pre-calentamiento (+5 o +10 por timestep) no compensan las penalizaciones acumuladas. El agente necesitaria calentar durante 2-4 horas de pre-peak para preparar el peak, pero cada hora de calefaccion implica penalizacion de cycling al encender Y al apagar, mas penalizacion por varianza de PMV (que sube al calentar). El costo neto es negativo.

4. **Asimetria fundamental**: Las penalizaciones se aplican en CADA timestep, pero los bonus solo en ventanas especificas (15-16h pre-peak, 17-20h peak). El agente recibe ~8,760 timesteps de penalizaciones potenciales pero solo ~1,040 timesteps de bonus potenciales (4h peak * 260 dias laborales).

**Diagrama del problema:**
```
Estrategia "correcta" (lo que queriamos):
  Calentar sostenido → transiciones bajas → varianza baja → pre-heat → bonus
  Pero: calentar = energy_penalty alta

Estrategia "degenerada" (lo que el modelo aprendio):
  No calentar → 0 transiciones → varianza ~0 → PMV plano en frio
  Beneficio: evita TODAS las penalizaciones Y la energy_penalty
  Costo: solo pierde comfort_bonus (+50/zona) y comfort_penalty
```

El modelo de RL encontro que el costo de perder comfort_bonus (100 por timestep con 2 zonas) era menor que el costo acumulado de energy_penalty + cycling_penalty + stability_penalty al calentar. Con coeficientes altos (C47), esta desigualdad se amplifica.

**Leccion clave**: En reward engineering para RL, cada penalizacion debe evaluarse contra sus **equilibrios degenerados**. Si existe una forma trivial de evitar la penalizacion que NO implique el comportamiento deseado, el agente la encontrara. Las penalizaciones temporales requerian que el agente "pasara por" un estado costoso (calentar) para llegar a un estado beneficioso (confort estable), pero el policy gradient optimiza instantaneamente y prefiere el atajo de no actuar.

---

## Resumen de Mejores Modelos por Categoria

### Mejor en CONFORT TOTAL: C39 (res50)
- **Config**: SAC, le=2.0, lt=30, wf=100, bonus=+30, 10 ep
- Confort: East=43.8% (**SUPERA BL**), West=69.0% (**SUPERA BL**)
- Costo: $123,019 (+83% vs ON/OFF) ❌
- **Ruta modelo**: `Eplus-SAC-training-nuestroMultizona_2026-02-12_*/model.zip` (eval: res50)

### Mejor en COSTO: C42 (res53)
- **Config**: SAC, le=2.0, lt=30, wf=100, bonus=+30, peak_mult=3x, 10 ep
- Costo: $56,940 (**SUPERA BL**, -15.4%) ✅
- Confort: East=27.0%, West=37.9% ❌
- Shift peak exitoso: ON peak 16.0% vs BL 35.3%

### Mas Cercano al Objetivo Triple: C44 (res54)
- **Config**: SAC, le=2.0, lt=30, wf=100, bonus=+50, peak_mult=3x, 10 ep
- Costo: $73,412 (+9% vs BL) ❌ (cerca)
- Confort East: 43.5% ❌ (**a 0.1% del target**)
- Confort West: 26.3% ❌
- **Ruta modelo**: eval res54

---

## Estado Actual del Reward (Post-C47, revertido a C44)

### `nuestroMultrizona.yaml`
```yaml
energy_weight: 0.5
lambda_energy: 2.0
lambda_temperature: 30
high_price: 14.493
low_price: 4.556
```

### `rewards.py` - NuestroRewardMultizona.__call__
```python
# 1. Energy penalty con tarifa horaria
price_kwh = low_price  # o high_price si L-V 17-20h
energy_penalty = -(total_energy / 1000) * price_kwh

# 2. Peak multiplier (2x en L-V 17-20h)
if is_peak:
    energy_penalty *= 2.0

# 3. Waste factor cuadratico (PMV > -0.5)
if total_energy > 0 and avg_pmv > -0.5:
    excess = avg_pmv + 0.5
    waste_factor = 1.0 + 100.0 * (excess ** 2)
    energy_penalty *= waste_factor

# 4. Comfort bonus (+50 por zona en PMV [-0.5, 0.5])
for pmv in pmv_values:
    if -0.5 <= pmv <= 0.5:
        comfort_bonus += 50.0

# 5. Reward final
reward = W * le * energy_penalty + (1-W) * lt * comfort_penalty + comfort_bonus
```

### `train_agent_SAC_nuestroMultizona.yaml`
```yaml
episodes: 10
algorithm: SAC
```

---

## Lecciones Aprendidas

### 1. Trade-off Costo vs Confort (Frontera de Pareto)
Ninguna configuracion logro superar al ON/OFF en las 3 metricas simultaneamente. Existe un trade-off fundamental:
- **Bajo costo** (<$67k) → confort baja (~20-35%)
- **Alto confort** (>43%) → costo alto (>$79k)
- El ON/OFF opera en un punto eficiente que el RL no puede replicar facilmente

### 2. Peak Multiplier: Herramienta Clave
El multiplicador de penalizacion en horas pico (C42-C44) fue la innovacion mas efectiva:
- Redujo el uso en peak de 35% a 16-30%
- Habilito costos menores que el baseline ($56,940 en C42)
- Pero el shift de energia sacrifica confort

### 3. Comfort Bonus: Impacto Significativo
El bonus positivo por zona en confort (+30 a +50) cambio fundamentalmente el comportamiento:
- Sin bonus: modelo minimiza energia, ignora confort
- Con bonus: modelo BUSCA activamente mantener confort
- C39 (bonus=+30) logro el mejor confort total (43.8%/69.0%)

### 4. Reward Shaping Temporal: FRACASO
Las penalizaciones por cycling y varianza de PMV tienen equilibrios degenerados:
- El modelo aprende a NO calentar para evitar penalizaciones (en vez de calentar sostenidamente)
- C47 (coefs altos): PMV cayo a -0.95 en ambas zonas
- **No recomendado** en su forma actual

### 5. SAC vs PPO
- **SAC**: Mas flexible, puede encontrar soluciones eficientes en energia, pero sensible a hiperparametros
- **PPO**: Tiende a calentar agresivamente (buen confort, alto costo), mas estable pero menos eficiente

### 6. Desbalance East/West
La bomba de calor sirve ambas zonas simultaneamente. Cuando el modelo optimiza una zona, la otra puede sufrir:
- C44: East 43.5% pero West solo 26.3%
- C39: East 43.8% y West 69.0% (mejor balance, pero caro)

---

## Como Revertir al Estado Pre-Iteracion

### Estado original (antes de cualquier iteracion):
```yaml
# nuestroMultrizona.yaml
energy_weight: 0.5
lambda_energy: 1.0
lambda_temperature: 30
high_price: 14.493
low_price: 4.556
```

```python
# rewards.py - Sin waste factor, sin bonus, sin peak multiplier
energy_penalty = -(total_energy/1000) * price
comfort_penalty = -violation (solo PMV < -0.5)
reward = W * le * energy_penalty + (1-W) * lt * comfort_penalty
```

### Modelos entrenados preservados:
Todos los modelos estan en directorios `Eplus-SAC-training-nuestroMultizona_2026-02-1*` y `Eplus-PPO-training-nuestroMultizona_2026-02-1*`. Evaluaciones en `Evaluacion-modelo-final-res*`.

---

## Proximos Pasos Sugeridos

1. **Fine-tuning de C44**: East esta a 0.1% del target. Probar comfort_bonus=55 o 60, o peak_mult=2.5 en lugar de 3x
2. **Mas episodios**: C44 con 15-20 episodios podria converger mejor
3. **Curriculum learning**: Entrenar primero sin penalizacion de energia (solo confort), luego agregar gradualmente
4. **Action space discreto**: Limitar a ON/OFF binario podria igualar la eficiencia del termostato
5. **Seed fija + multiples runs**: Dado la alta varianza de SAC, correr la misma config 3-5 veces y elegir el mejor
