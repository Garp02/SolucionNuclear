"""
limpieza.py — Limpieza de datos del pipeline de monitoreo de radiación ambiental

Decisiones de diseño aplicadas:
  1. dose_nSv_h: conservar sin imputar (69.9% nulos; valor como evidencia de calidad)
  2. gamma_*_cpm: imputar nulos con mediana por station_id; flag gamma_imputed
  3. Outliers: eliminar fuera de rango físico; marcar IQR por estación con is_outlier
  4. Timestamps: eliminar filas sin timestamp
  5. Coordenadas: geocodificar reactores sin lat/lon con Nominatim; eliminar fallidos
  6. OperationalFrom: conservar todo (filtro Status=Operational queda para fusion.py)
  7. ReactorType: mapear a 8 categorías canónicas IAEA

Uso:
  python limpieza.py \
    --radiation datos_procesados/radiation_db.csv \
    --plants datos/kaggle/nuclear_power_plants.csv \
    --output-dir datos_procesados/

Salidas:
  radiation_clean.csv
  nuclear_plants_clean.csv
  reporte_limpieza.txt
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Configuración de logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# Umbrales físicos para radiación ambiental (valores fuera de rango = error instrumental)
GAMMA_CPM_MIN: float = 0.0 # no pueden existir conteos negativos
GAMMA_CPM_MAX: float = 10_000.0 # > 10 000 cpm sería un evento radiológico mayor
DOSE_NSV_H_MIN: float = 0.0 # dosis negativa = error de sensor
DOSE_NSV_H_MAX: float = 100_000.0 # > 100 µSv/h = accidente nuclear (umbral NRC)

# Multiplicador IQR para marcar outliers estadísticos
IQR_FACTOR: float = 1.5

# Categorías canónicas IAEA para ReactorType
REACTOR_TYPE_MAP: dict[str, str] = {
    # PWR — Pressurized Water Reactor
    "pwr": "PWR", "pressurized water reactor": "PWR",
    "vver": "PWR", "vver-440": "PWR", "vver-1000": "PWR",
    "ap1000": "PWR", "epr": "PWR", "ap600": "PWR",
    "apwr": "PWR", "ksnp": "PWR", "acp": "PWR",
    "hpr1000": "PWR", "acpr": "PWR", "wwer": "PWR",
    # BWR — Boiling Water Reactor
    "bwr": "BWR", "boiling water reactor": "BWR",
    "abwr": "BWR", "esbwr": "BWR", "sbwr": "BWR",
    # PHWR — Pressurized Heavy Water Reactor
    "phwr": "PHWR", "candu": "PHWR",
    "pressurized heavy water reactor": "PHWR",
    "hwpwr": "PHWR",
    # RBMK — Reactor Bolshoy Moshchnosti Kanalnyy
    "rbmk": "RBMK", "rbmk-1000": "RBMK",
    # GCR — Gas Cooled Reactor
    "gcr": "GCR", "agr": "GCR", "magnox": "GCR",
    "gas cooled reactor": "GCR", "advanced gas reactor": "GCR",
    # LWGR — Light Water Graphite Reactor
    "lwgr": "LWGR", "egp": "LWGR",
    # FBR — Fast Breeder Reactor
    "fbr": "FBR", "sfr": "FBR", "lmfbr": "FBR",
    "fast breeder": "FBR", "fast breeder reactor": "FBR",
    "bfr": "FBR"
}

# Utilidades
def _gamma_cols(df: pd.DataFrame) -> list[str]:
    """Detecta las columnas de tasa de conteo gamma (patron gamma_*_cpm)."""
    return [c for c in df.columns if re.match(r"gamma_r\d+_cpm", c, re.IGNORECASE)]

def _report_line(msg: str) -> str:
    log.info(msg)
    return msg + "\n"

# Limpieza de radiation_db
def clean_radiation(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aplica todas las reglas de limpieza a radiation_db.
    Devuelve (dataframe_limpio, lineas_de_reporte).
    """
    lines: list[str] = []
    n_inicial = len(df)
    lines.append(_report_line(f"[radiation_db] Filas iniciales: {n_inicial:,}"))

    # 1. Timestamps nulos (eliminar)
    mask_ts_nulo = df["timestamp"].isna()
    n_ts = mask_ts_nulo.sum()
    df = df[~mask_ts_nulo].copy()
    lines.append(_report_line(f"Timestamps nulos eliminados: {n_ts:,} ({n_ts / n_inicial:.2%})"))

    # Asegurar tipo datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    n_ts_coerce = df["timestamp"].isna().sum()
    if n_ts_coerce:

        df = df[df["timestamp"].notna()].copy()
        lines.append(_report_line(f"Timestamps con formato inválido eliminados: {n_ts_coerce:,}"))

    # 2. dose_nSv_h (conservar); solo eliminar fuera de rango físico
    gcols = _gamma_cols(df)
    lines.append(_report_line(f"Columnas gamma detectadas: {gcols}"))

    # Eliminar valores físicamente imposibles en dose_nSv_h
    n_antes = len(df)
    if "dose_nSv_h" in df.columns:

        mask_dose_invalid = (
            df["dose_nSv_h"].notna() & (
                (df["dose_nSv_h"] < DOSE_NSV_H_MIN) |
                (df["dose_nSv_h"] > DOSE_NSV_H_MAX)
            )
        )
        n_dose_inv = mask_dose_invalid.sum()
        df.loc[mask_dose_invalid, "dose_nSv_h"] = np.nan  # → NaN, no eliminar fila
        lines.append(_report_line(f"dose_nSv_h fuera de rango físico pasa a NaN: {n_dose_inv:,}"))

    # 3. gamma_*_cpm: rango físico → NaN, luego imputar
    for col in gcols:
        mask_inv = (
            df[col].notna() & (
                (df[col] < GAMMA_CPM_MIN) |
                (df[col] > GAMMA_CPM_MAX)
            )
        )
        n_inv = mask_inv.sum()
        df.loc[mask_inv, col] = np.nan
        if n_inv:
            lines.append(_report_line(
                f"  {col} fuera de rango físico pasa a NaN: {n_inv:,}"
            ))

    # 4. Imputación gamma_*_cpm con mediana por station_id
    # Calcular una columna agregada gamma_cpm (promedio de canales disponibles)
    if gcols:
        df["gamma_cpm"] = df[gcols].mean(axis = 1, skipna = True)

    # Flag de imputación (True si TODOS los canales gamma eran nulos)
    df["gamma_imputed"] = False

    for col in (["gamma_cpm"] + gcols):

        if col not in df.columns:
        
            continue
        
        nulos_antes = df[col].isna().sum()
        
        if nulos_antes == 0:
        
            continue

        mediana_por_estacion = (df.groupby("station_id")[col].transform("median"))
        mask_nulo = df[col].isna()
        df.loc[mask_nulo, col] = mediana_por_estacion[mask_nulo]
        df.loc[mask_nulo, "gamma_imputed"] = True

        nulos_despues = df[col].isna().sum()  # casos donde toda la estación era NaN
        lines.append(_report_line(
            f"{col}: {nulos_antes:,} nulos a imputados con mediana/estación "
            f"({nulos_despues:,} sin imputar — estaciones sin historial)"
        ))

    # 5. Outliers IQR por estación (is_outlier)
    df["is_outlier"] = False

    ref_col = "gamma_cpm" if "gamma_cpm" in df.columns else (gcols[0] if gcols else None)

    if ref_col:
     
        def _iqr_mask(grp: pd.Series) -> pd.Series:
            q1 = grp.quantile(0.25)
            q3 = grp.quantile(0.75)
            iqr = q3 - q1
     
            return (grp < q1 - IQR_FACTOR * iqr) | (grp > q3 + IQR_FACTOR * iqr)

        outlier_mask = (df.groupby("station_id")[ref_col].transform(_iqr_mask).fillna(False))
        df["is_outlier"] = outlier_mask.astype(bool)
        n_outliers = df["is_outlier"].sum()
        lines.append(_report_line(f"Outliers IQR marcados en {ref_col}: {n_outliers:,}"
            f"({n_outliers / len(df):.2%}) — columna is_outlier=True"))

    n_final = len(df)
    lines.append(_report_line(f"[radiation_db] Filas finales: {n_final:,}"
        f"(eliminadas: {n_inicial - n_final:,})"))
    
    return df, lines

# Geocodificación de plantas sin coordenadas
def geocode_missing(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """
    Intenta geocodificar filas donde Latitude o Longitude es NaN.
    Usa la columna 'Name' + 'Country' (o equivalente) como query.
    Elimina las que no se resuelvan tras el intento.
    """
    # Detectar columnas de nombre y país (el CSV Kaggle puede variar ligeramente)
    name_col = next((c for c in df.columns if c.lower() == "name"), None)
    country_col = next(
        (c for c in df.columns if c.lower() in {"country", "country_code"}), None
    )

    mask_sin_coords = df["Latitude"].isna() | df["Longitude"].isna()
    n_sin_coords = mask_sin_coords.sum()

    if n_sin_coords == 0:
        lines.append(_report_line("Sin coordenadas faltantes — geocodificación no necesaria."))
        return df

    lines.append(_report_line(f"Reactores sin coordenadas: {n_sin_coords} — iniciando geocodificación con Nominatim"))
    geolocator = Nominatim(user_agent="calidad_datos_proyecto_iimas", timeout=10)
    resueltos = 0
    fallidos_idx: list[int] = []

    for idx in df[mask_sin_coords].index:

        row = df.loc[idx]
        query_parts = []
        
        if name_col and pd.notna(row.get(name_col)):
        
            query_parts.append(str(row[name_col]))
        
        if country_col and pd.notna(row.get(country_col)):
        
            query_parts.append(str(row[country_col]))

        query = ", ".join(query_parts)
        
        if not query.strip():
        
            fallidos_idx.append(idx)
            continue

        try:
            time.sleep(1.1)  # respetar límite de Nominatim (1 req/s)
            location = geolocator.geocode(query)
        
            if location:
        
                df.at[idx, "Latitude"] = location.latitude
                df.at[idx, "Longitude"] = location.longitude
                resueltos += 1
        
            else:
        
                fallidos_idx.append(idx)
        
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
        
            log.warning("Geocodificación fallida para '%s': %s", query, exc)
            fallidos_idx.append(idx)

    lines.append(_report_line(f"Geocodificados correctamente: {resueltos}/{n_sin_coords}"))
    lines.append(_report_line(f"No resueltos (eliminados): {len(fallidos_idx)}"))
    df = df.drop(index = fallidos_idx).reset_index(drop = True)
    
    return df

# Limpieza de nuclear_power_plants
def standardize_reactor_type(valor: str | float) -> str:
    """Normaliza un valor de ReactorType a categoría canónica IAEA."""
    
    if pd.isna(valor) or str(valor).strip() == "":
    
        return "OTHER"
    
    normalizado = str(valor).strip().lower()
    
    # Búsqueda exacta primero
    if normalizado in REACTOR_TYPE_MAP:
    
        return REACTOR_TYPE_MAP[normalizado]
    
    # Búsqueda parcial (el valor puede contener la clave como subcadena)
    for clave, categoria in REACTOR_TYPE_MAP.items():
    
        if clave in normalizado:
    
            return categoria
    
    return "OTHER"

def clean_plants(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aplica todas las reglas de limpieza a nuclear_power_plants.
    Devuelve (dataframe_limpio, lineas_de_reporte).
    """
    lines: list[str] = []
    n_inicial = len(df)
    lines.append(_report_line(f"[nuclear_plants] Filas iniciales: {n_inicial:,}"))

    # 1. Estandarizar ReactorType
    # Detectar columna (puede llamarse ReactorType o Reactor_Type)
    rt_col = next((c for c in df.columns if c.lower().replace("_", "") == "reactortype"), None)
    
    if rt_col:
    
        original_vals = df[rt_col].unique()
        df["ReactorType_std"] = df[rt_col].apply(standardize_reactor_type)
        dist = df["ReactorType_std"].value_counts().to_dict()
        lines.append(_report_line(f"ReactorType estandarizado a categorías IAEA: {dist}"))
    
    else:
    
        log.warning("Columna ReactorType no encontrada; se creará ReactorType_std='OTHER'")
        df["ReactorType_std"] = "OTHER"

    # 2. Geocodificar reactores sin coordenadas
    # Normalizar nombres de columnas de coordenadas
    for old, new in [("latitude", "Latitude"), ("longitude", "Longitude")]:

        if old in df.columns and new not in df.columns:
       
            df.rename(columns={old: new}, inplace=True)

    df = geocode_missing(df, lines)

    # 3. Conservar OperationalFrom con nulos (decisión 6)
    n_op_from_nulos = df["OperationalFrom"].isna().sum() if "OperationalFrom" in df.columns else 0
    lines.append(_report_line(f"OperationalFrom nulos conservados: {n_op_from_nulos} "
        f"(filtro Status=Operational se aplica en fusion.py)"))

    # 4. Capacity: documentar nulos, no imputar
    if "Capacity" in df.columns:
     
        n_cap_nulos = df["Capacity"].isna().sum()
        lines.append(_report_line(f"Capacity nulos: {n_cap_nulos} ({n_cap_nulos / len(df):.1%}) — conservados"))
        # Asegurar tipo numérico
        df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")

    n_final = len(df)
    lines.append(_report_line(
        f"[nuclear_plants] Filas finales: {n_final:,} "
        f"(eliminadas: {n_inicial - n_final:,} — sin coordenadas tras geocodificación)"
    ))
    return df, lines

# Escritura del reporte de limpieza
def write_report(output_dir: Path, lines: list[str]) -> None:
    report_path = output_dir / "reporte_limpieza.txt"
    header = ("REPORTE DE LIMPIEZA — Pipeline de Radiación Ambiental\n")
    report_path.write_text(header + "".join(lines), encoding="utf-8")
    log.info("Reporte escrito en: %s", report_path)


# CLI
def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Limpieza de radiation_db y nuclear_power_plants")
    parser.add_argument(
        "--radiation",
        default=str(project_root / "datos_procesados" / "radiation_db.csv"),
        help="Ruta al CSV de radiation_db (salida de ingesta.py)",
    )
    parser.add_argument(
        "--plants",
        default=str(project_root / "datos" / "kaggle" / "nuclear_power_plants.csv"),
        help="Ruta al catálogo de plantas nucleares (Kaggle)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "datos_procesados"),
        help="Directorio donde se guardan los CSV limpios y el reporte",
    )
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Omitir geocodificación (útil para pruebas rápidas)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []

    # radiation_db
    log.info("Cargando radiation_db desde: %s", args.radiation)
    rad_df = pd.read_csv(args.radiation, low_memory = False, parse_dates = ["timestamp"])
    rad_clean, rad_lines = clean_radiation(rad_df)
    all_lines.extend(rad_lines)

    rad_out = output_dir / "radiation_clean.csv"
    log.info("Guardando radiation_clean.csv (%d filas)", len(rad_clean))
    rad_clean.to_csv(rad_out, index=False)
    log.info("%s", rad_out)

    # nuclear_power_plants
    log.info("Cargando nuclear_power_plants desde: %s", args.plants)
    plants_df = pd.read_csv(args.plants, low_memory=False)

    if args.skip_geocoding:
     
        # Eliminar directamente los que no tienen coordenadas
        mask = plants_df["Latitude"].isna() | plants_df["Longitude"].isna()
        n_drop = mask.sum()
        plants_df = plants_df[~mask].reset_index(drop = True)
        all_lines.append(_report_line(
            f"[--skip-geocoding] Reactores sin coordenadas eliminados sin intentar geocodificar: {n_drop}"
        ))
        # Aún así estandarizar ReactorType
        rt_col = next((c for c in plants_df.columns if c.lower().replace("_", "") == "reactortype"), None)
        
        if rt_col:
        
            plants_df["ReactorType_std"] = plants_df[rt_col].apply(standardize_reactor_type)
        
        plants_clean = plants_df
        plants_lines: list[str] = [_report_line(
            f"[nuclear_plants] Filas finales (sin geocodificación): {len(plants_clean):,}"
        )]
    
    else:
    
        plants_clean, plants_lines = clean_plants(plants_df)

    all_lines.extend(plants_lines)

    plants_out = output_dir / "nuclear_plants_clean.csv"
    log.info("Guardando nuclear_plants_clean.csv (%d filas)", len(plants_clean))
    plants_clean.to_csv(plants_out, index = False)
    log.info("%s", plants_out)

    # Reporte
    write_report(output_dir, all_lines)
    log.info("Limpieza completada.")

if __name__ == "__main__":
    
    main()