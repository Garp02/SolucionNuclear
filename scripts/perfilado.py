"""
perfilado.py — Perfil de calidad de datos (TDQM)
=================================================
Genera reportes HTML con ydata-profiling para las dos fuentes primarias:
  1. radiation_db.csv (17M filas, 1.7 GB) — muestreado
  2. nuclear_power_plants.csv (800 filas) — completo

Uso:
    python scripts/perfilado.py
    python scripts/perfilado.py --sample 50000
    python scripts/perfilado.py --radiation datos_procesados/radiation_db.csv \
                                 --plantas datos/kaggle/nuclear_power_plants.csv \
                                 --output datos_procesados/reportes/

Salida:
    datos_procesados/reportes/reporte_radiation_db.html
    datos_procesados/reportes/reporte_nuclear_plants.html
    datos_procesados/reportes/resumen_calidad.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

# Logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final")
RADIATION = BASE_DIR / "datos_procesados" / "radiation_db.csv"
PLANTAS = BASE_DIR / "datos" / "kaggle" / "nuclear_power_plants.csv"
OUTPUT_DIR = BASE_DIR / "datos_procesados" / "reportes"

# Tamaño de muestra para radiation_db (completo sería >17M filas)
DEFAULT_SAMPLE = 100_000

# Columnas de tipo datetime para forzar parseo correcto
RADIATION_PARSE_DATES = ['timestamp']
PLANTAS_PARSE_DATES = ['ConstructionStartAt', 'OperationalFrom', 'OperationalTo', 'LastUpdatedAt']


# Helpers

def cargar_radiation(path: Path, n_sample: int) -> pd.DataFrame:
    """
    Carga radiation_db.csv completo y extrae una muestra estratificada por estación.
    La muestra mantiene representación proporcional de cada station_id.
    """
    logger.info("Cargando radiation_db desde %s", path)
    t0 = time.time()

    df = pd.read_csv(
        path,
        low_memory=False,
        parse_dates=RADIATION_PARSE_DATES,
    )

    logger.info("Cargado en %.1f s — %s filas, %s columnas", time.time() - t0, f"{len(df):,}", len(df.columns))

    # Muestra estratificada: proporcional al tamaño de cada estación
    n_sample = min(n_sample, len(df))
    logger.info("Extrayendo muestra estratificada de %s filas", f"{n_sample:,}")

    fracs = df["station_id"].value_counts(normalize=True)
    partes = []
    for station, frac in fracs.items():
        sub = df[df["station_id"] == station]
        k = max(1, round(frac * n_sample))
        partes.append(sub.sample(n=min(k, len(sub)), random_state=42))

    muestra = pd.concat(partes).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Muestra final: %s filas de %s estaciones", f"{len(muestra):,}", muestra["station_id"].nunique())
    return muestra


def cargar_plantas(path: Path) -> pd.DataFrame:
    """Carga el catálogo de plantas nucleares completo."""
    logger.info("Cargando nuclear_power_plants desde %s", path)
    df = pd.read_csv(path, low_memory=False, parse_dates=PLANTAS_PARSE_DATES)
    logger.info("Cargado — %s filas, %s columnas", f"{len(df):,}", len(df.columns))
    return df


def resumen_nulos(df: pd.DataFrame, nombre: str) -> str:
    """Genera un bloque de texto con estadísticas básicas de calidad."""
    lineas = [
        f"DIAGNÓSTICO RÁPIDO — {nombre}",
        f"Filas: {len(df):,}",
        f"Columnas: {len(df.columns)}",
        f"Duplicados: {df.duplicated().sum():,}",
        f"Nulos por columna:",
    ]
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    for col in df.columns:
        if nulos[col] > 0:
            lineas.append(f"{col:<35} {nulos[col]:>10,}  ({nulos_pct[col]:.1f}%)")
    if nulos.sum() == 0:
        lineas.append("(sin nulos)")
    lineas.append("")
    return "\n".join(lineas)


def generar_reporte(
    df: pd.DataFrame,
    titulo: str,
    output_path: Path,
    minimal: bool = True,
) -> None:
    """Genera y guarda el reporte HTML de ydata-profiling."""
    logger.info("Generando reporte '%s' …", titulo)
    t0 = time.time()

    perfil = ProfileReport(
        df,
        title=titulo,
        minimal=minimal,
        explorative=False,
        progress_bar=True,
        correlations=None if minimal else {"pearson": {"calculate": True}}
    )

    perfil.to_file(output_path)
    logger.info("Reporte guardado en %s (%.1f s)", output_path, time.time() - t0)


# Main
def main(
    radiation_path: Path,
    plantas_path: Path,
    output_dir: Path,
    n_sample: int,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    resumen_txt = output_dir / "resumen_calidad.txt"
    bloques = []

    # 1. radiation_db
    if not radiation_path.exists():
        logger.error("No se encontró radiation_db en: %s", radiation_path)
        logger.error("Ejecuta primero: python scripts/ingesta.py")
        sys.exit(1)

    df_rad = cargar_radiation(radiation_path, n_sample)
    bloques.append(resumen_nulos(df_rad, "radiation_db (muestra)"))

    generar_reporte(
        df_rad,
        titulo=f"RadNet — radiation_db (muestra {n_sample:,} filas)",
        output_path=output_dir / "reporte_radiation_db.html",
        minimal=True,
    )

    # 2. nuclear_power_plants
    if not plantas_path.exists():
        logger.error("No se encontró nuclear_power_plants en: %s", plantas_path)
        sys.exit(1)

    df_plantas = cargar_plantas(plantas_path)
    bloques.append(resumen_nulos(df_plantas, "nuclear_power_plants"))

    generar_reporte(
        df_plantas,
        titulo="Geo Nuclear Data — nuclear_power_plants",
        output_path=output_dir / "reporte_nuclear_plants.html",
        minimal=False,   # archivo pequeño, puede hacer análisis completo
    )

    # 3. Guardar resumen de texto
    contenido = "\n".join(bloques)
    resumen_txt.write_text(contenido, encoding="utf-8")
    logger.info("Resumen de calidad guardado en: %s", resumen_txt)

    # Imprimir en consola también
    print(contenido)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera reportes de calidad con ydata-profiling."
    )
    parser.add_argument(
        "--radiation",
        type=Path,
        default=RADIATION,
        help="Ruta a radiation_db.csv",
    )
    parser.add_argument(
        "--plantas",
        type=Path,
        default=PLANTAS,
        help="Ruta a nuclear_power_plants.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Carpeta de salida para los reportes HTML",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Tamaño de muestra para radiation_db (default: {DEFAULT_SAMPLE:,})",
    )

    args = parser.parse_args()
    main(args.radiation, args.plantas, args.output, args.sample)