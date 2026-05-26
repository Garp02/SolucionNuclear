"""
ingesta.py
==========
Pipeline de ingesta para datos RadNet (EPA).

Por cada estación construye la serie de tiempo completa combinando:
  - CSV histórico (ZIP descomprimido, años anteriores)
  - CSV del año en curso (REST API)

Resultado: radiation_db.csv con una columna adicional 'years_of_data'
que indica cuántos años de historia tiene cada estación.

Uso:
    python ingesta.py

    # O especificando directorios:
    python ingesta.py --input ./datos --output ./processed
"""

import re
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Logging 
logging.basicConfig(level = logging.INFO, format = '%(asctime)s [%(levelname)s] %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Constantes 
TIMESTAMP_COL = 'SAMPLE COLLECTION TIME'
LOCATION_COL = 'LOCATION_NAME'
STATUS_COL = 'STATUS'

# Columnas en el orden canónico final
CANONICAL_COLS = ['station_id', 'state', 'city', 'timestamp', 'dose_nSv_h',
    'gamma_R02_cpm', 'gamma_R03_cpm', 'gamma_R04_cpm', 'gamma_R05_cpm',
    'gamma_R06_cpm', 'gamma_R07_cpm', 'gamma_R08_cpm', 'gamma_R09_cpm',
    'status', 'source', 'years_of_data']

# Mapeo de columnas originales → canónicas
COL_RENAME = {
    LOCATION_COL: 'location_raw',
    TIMESTAMP_COL: 'timestamp',
    'DOSE EQUIVALENT RATE (nSv/h)': 'dose_nSv_h',
    'GAMMA COUNT RATE R02 (CPM)': 'gamma_R02_cpm',
    'GAMMA COUNT RATE R03 (CPM)': 'gamma_R03_cpm',
    'GAMMA COUNT RATE R04 (CPM)': 'gamma_R04_cpm',
    'GAMMA COUNT RATE R05 (CPM)': 'gamma_R05_cpm',
    'GAMMA COUNT RATE R06 (CPM)': 'gamma_R06_cpm',
    'GAMMA COUNT RATE R07 (CPM)': 'gamma_R07_cpm',
    'GAMMA COUNT RATE R08 (CPM)': 'gamma_R08_cpm',
    'GAMMA COUNT RATE R09 (CPM)': 'gamma_R09_cpm',
    STATUS_COL: 'status'
}


# Helpers
def parse_location(location_raw: str) -> tuple[str, str]:
    """
    Extrae estado y ciudad desde el campo LOCATION_NAME.

    Ejemplos:
        'TX: SAN ANGELO' pasa a ('TX', 'SAN ANGELO')
        'MT: BILLINGS' pasa a ('MT', 'BILLINGS')
        'PA: PITTSBURGH' pasa a ('PA', 'PITTSBURGH')
    """
    parts = location_raw.split(":", maxsplit = 1)
    
    if len(parts) == 2:
    
        return parts[0].strip().upper(), parts[1].strip().upper()
    
    return "??", location_raw.strip().upper()


def infer_source_from_filename(filename: str) -> str:
    """
    Clasifica el archivo como histórico o del año en curso basándose
    en su nombre.

    Patrón histórico: TX_SAN_ANGELO_2025-2007.csv (rango de años)
    Patrón API actual: TX_SAN_ANGELO_2026.csv (año único)
    """
    # Si el nombre contiene un rango tipo '2025-2007'
    if re.search(r"\d{4}-\d{4}", filename):
        
        return "historical"
    
    return "api_current"


def load_csv(path: Path) -> pd.DataFrame | None:
    """
    Carga un CSV de RadNet, renombra columnas al esquema canónico
    y agrega metadatos de origen.

    Retorna None si el archivo está vacío o es ilegible.
    """
    try:

        df = pd.read_csv(path, low_memory=False)
    
    except Exception as exc:
    
        logger.warning("No se pudo leer %s: %s", path.name, exc)
    
        return None

    if df.empty:
    
        logger.warning("Archivo vacío, se omite: %s", path.name)
    
        return None

    # Renombrar columnas
    df = df.rename(columns = COL_RENAME)

    # Parsear timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')

    # Extraer estado y ciudad desde location_raw
    if "location_raw" in df.columns:

        _split = df['location_raw'].str.split(":", n = 1, expand = True)
        df['state'] = _split[0].str.strip().str.upper()
        df['city'] = _split[1].str.strip().str.upper().fillna(df['location_raw'].str.strip().str.upper())
    
    else:
        # Fallback: inferir desde el nombre del archivo
        stem = path.stem.upper() # TX_SAN_ANGELO_2026
        parts = stem.split("_")
        df['state'] = parts[0] if parts else "??"
        df['city'] = "_".join(parts[1:-1]) if len(parts) > 2 else "??"

    # station_id: STATE_CITY en mayúsculas sin espacios
    df['station_id'] = df['state'] + "_" + df['city'].str.replace(" ", "_")

    # Origen del archivo
    df['source'] = infer_source_from_filename(path.name)
    return df


def compute_years_of_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna 'years_of_data' calculada por station_id
    como la diferencia en años entre el primer y último timestamp.
    """
    span = (df.groupby('station_id')['timestamp'].agg(first_ts = "min", last_ts = 'max').reset_index())
    span['years_of_data'] = ((span['last_ts'] - span['first_ts']).dt.days / 365.25).round(2)

    df = df.merge(span[['station_id', 'years_of_data']], on = 'station_id', how = 'left')
    return df


def build_radiation_db(input_dir: Path) -> pd.DataFrame:
    """
    Lee todos los CSV del directorio, los combina y construye
    la base unificada con deduplicación por (station_id, timestamp).

    Parámetros
    ----------
    input_dir : Path
        Directorio que contiene los CSV descargados (históricos + API).

    Retorna
    -------
    pd.DataFrame con el modelo canónico completo.
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    logger.info('CSV encontrados en %s: %d', input_dir, len(csv_files))

    if not csv_files:

        raise FileNotFoundError(f'No se encontraron CSV en: {input_dir}')

    frames: list[pd.DataFrame] = []

    for path in csv_files:
        
        logger.info('Cargando: %s', path.name)
        df = load_csv(path)
        
        if df is not None:
        
            frames.append(df)

    if not frames:
        
        raise ValueError('Ningún archivo pudo cargarse correctamente.')

    logger.info("Concatenando %d DataFrames", len(frames))
    combined = pd.concat(frames, ignore_index = True)
    logger.info('Total filas antes de deduplicar: %s', f'{len(combined):,}')

    # Deduplicación
    # Si un registro aparece en ambas fuentes (traslape año en
    # curso entre ZIP y REST), conservar el de 'api_current' que es más reciente y limpio.
    combined = combined.sort_values( by = ['station_id', 'timestamp', 'source'], ascending = [True, True, False])
    before = len(combined)
    combined = combined.drop_duplicates(subset = ['station_id', 'timestamp'], keep = 'first')
    dupes_removed = before - len(combined)
    logger.info('Duplicados eliminados: %s', f'{dupes_removed:,}')
    logger.info('Total filas después de deduplicar: %s', f'{len(combined):,}')

    # Ordenar cronológicamente
    combined = combined.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Agregar years_of_data
    combined = compute_years_of_data(combined)

    # Seleccionar y ordenar columnas canónicas
    available = [c for c in CANONICAL_COLS if c in combined.columns]
    combined  = combined[available]

    return combined


def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen legible del DataFrame resultante."""
    print(" RESUMEN — radiation_db")
    print(f"Total de registros: {len(df):>12,}")
    print(f"Estaciones únicas: {df['station_id'].nunique():>12,}")
    print(f"Rango temporal: {df['timestamp'].min()} -- {df['timestamp'].max()}")
    print(f"Columnas: {len(df.columns)}")

    print("\nPor estación")
    summary = (
        df.groupby("station_id")
        .agg(
            registros = ("timestamp", "count"),
            inicio = ("timestamp", "min"),
            fin = ("timestamp", "max"),
            years = ("years_of_data", "first"),
            fuentes = ("source", lambda x: ", ".join(sorted(x.unique()))),
        ).reset_index().sort_values("station_id"))
    
    for _, row in summary.iterrows():
    
        print(
            f"{row['station_id']:<30} "
            f"{row['registros']:>8,} registros "
            f"{row['years']:>5.1f} años "
            f"[{row['fuentes']}]")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    
    if not nulls.empty:
        
        print('\n Nulos por columna')
        for col, n in nulls.items():
    
            pct = n / len(df) * 100
            print(f'{col:<30} {n:>8,} ({pct:.1f}%)')

# Entry point
def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents = True, exist_ok = True)
    output_path = output_dir / "radiation_db.csv"

    df = build_radiation_db(input_dir)
    print_summary(df)

    df.to_csv(output_path, index = False)
    logger.info('radiation_db.csv guardado en: %s', output_path.resolve())
    logger.info('Tamaño: %.1f MB', output_path.stat().st_size / 1_048_576)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Pipeline de ingesta RadNet EPA')
    parser.add_argument('--input', type = Path, 
                        default = Path('/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos'),
        help = 'Directorio con los datos (/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos)')
    
    parser.add_argument('--output', type = Path, 
                        default = Path('/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos_procesados'), 
                        help = 'Directorio de salida para radiation_db.csv (default: /home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos_procesados)',)
    
    args = parser.parse_args()
    main(args.input, args.output)