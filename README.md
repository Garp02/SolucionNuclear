# Proyecto — Calidad y Preprocesamiento de Datos

Autores: 
- Díaz Juarez Ana Sofía
- Esteva Gallegos Carlos Fabián
. León Navarrete Edmundo Adam
- Martinez Tapia Fernando
- Munive Ramírez Ibrahim

## Framework
**TDQM (Total Data Quality Management)** aplicado al dominio de monitoreo ambiental de radiación. Contexto regulatorio: CNSNS / IAEA / NRC.

---

## Fuentes de datos

| Fuente | Descripción |
|---|---|
| **EPA RadNet** | Red de monitoreo de radiación ambiental en EE.UU. — 25 CSVs históricos + API REST |
| **Geo Nuclear Data (Kaggle)** | Catálogo global de plantas nucleares — 803 reactores, 16 columnas |

---

## Pipeline

| Script | Función |
|---|---|
| `epa_radnet_api.py` | Descarga CSVs de RadNet (REST + ZIPs históricos) |
| `ingesta.py` | Unifica CSVs, deduplica, calcula `years_of_data` y devuelve `radiation_db.csv` |
| `perfilado.py` | Diagnóstico con `ydata-profiling` devuelve 2 reportes HTML y un resumen de calidad |
| `limpieza.py` | Imputa `gamma_cpm`, marca outliers IQR, estandariza `ReactorType`, geocodifica |
| `fusion.py` | Haversine vectorizado manda top 5 plantas más cercanas por estación |
| `analisis.py` | 15 queries con visualizaciones respondiendo las 5 preguntas de negocio|
|`pipeline.py`| Ejecuta el proyecto desde el perfilado hasta el análisis|

---

## Dimensiones de calidad abordadas

| Dimensión | Dónde aparece |
|---|---|
| **Completitud** | `dose_nSv_h` 69.9% nulos, `gamma_cpm` 1.5%, coordenadas de reactores 5.5% |
| **Consistencia** | `ReactorType` con variantes textuales se estandariza a 8 categorías IAEA |
| **Exactitud** | Eliminación de valores fuera de rango físico en `gamma_cpm` y `dose_nSv_h` |
| **Unicidad** | Deduplicación por `(station_id, timestamp)` con prioridad a `api_current` |
| **Integridad** | Geocodificación de reactores sin coordenadas para construir la tabla de distancias |

---

## 5 Preguntas de negocio

| # | Pregunta | Variable central |
|---|---|---|
| P1 | ¿Las estaciones cercanas a plantas operacionales registran tasas de dosis distintas a las alejadas? | `gamma_cpm`, `distance_km` |
| P2 | ¿Es posible detectar anomalías en lecturas de radiación de forma automática? | `gamma_cpm`, `is_outlier` |
| P3 | ¿Qué tipo de reactor o capacidad instalada correlaciona más con la tasa de conteo gamma en estaciones cercanas? | `reactor_type_std`, `capacity_mw` |
| P4 | ¿Es posible predecir la tasa de conteo gamma basándose en el historial de la estación y su proximidad a plantas? | `gamma_cpm`, `distance_km` |
| P5 | ¿Qué estaciones presentan los mayores problemas de calidad de datos? | `dose_nSv_h`, `gamma_imputed`, `is_outlier` |

> P3 y P4 usan `gamma_cpm` en lugar de `dose_nSv_h` por el 69.9% de nulos en esa columna.
