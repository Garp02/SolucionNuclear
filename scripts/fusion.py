"""
fusion.py — Construcción de la tabla station_plant_distances

Cruza las dos fuentes limpias para calcular la distancia Haversine entre cada
estación RadNet y cada planta nuclear operacional, conservando solo las 5
plantas más cercanas por estación.

Entradas:
  radiation_clean.csv      → coordenadas de estaciones RadNet
  nuclear_plants_clean.csv → coordenadas y atributos de plantas nucleares

Salida:
  station_plant_distances.csv  (140 estaciones × 5 plantas = 700 filas aprox.)

Columnas de salida:
  station_id, station_lat, station_lon, station_state, station_city,
  plant_name, plant_lat, plant_lon, plant_country, plant_status,
  reactor_type_std, capacity_mw, operational_from,
  distance_km, rank_proximity

Uso:
  python scripts/fusion.py
  python scripts/fusion.py --top-n 5 --output-dir datos_procesados/
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Haversine vectorizado
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6_371.0


def haversine_matrix(
    lat1: np.ndarray,  # (N,) estaciones
    lon1: np.ndarray,
    lat2: np.ndarray,  # (M,) plantas
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calcula una matriz de distancias (N × M) en kilómetros.
    Usa broadcasting para evitar loops.
    """
    # Convertir a radianes
    lat1 = np.radians(lat1[:, np.newaxis])   # (N, 1)
    lon1 = np.radians(lon1[:, np.newaxis])
    lat2 = np.radians(lat2[np.newaxis, :])   # (1, M)
    lon2 = np.radians(lon2[np.newaxis, :])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de coordenadas de estaciones
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lat_lon_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detecta columnas de latitud y longitud en radiation_clean."""
    lat_candidates = ["latitude", "lat", "station_lat", "lat_dd", "y"]
    lon_candidates = ["longitude", "lon", "lng", "station_lon", "lon_dd", "x"]
    cols_lower = {c.lower(): c for c in df.columns}

    lat_col = next((cols_lower[c] for c in lat_candidates if c in cols_lower), None)
    lon_col = next((cols_lower[c] for c in lon_candidates if c in cols_lower), None)
    return lat_col, lon_col


def geocode_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """
    Geocodifica estaciones usando city + state como query.
    Se llama solo cuando radiation_clean no tiene columnas lat/lon.
    stations: DataFrame con columnas [station_id, city, state]
    Devuelve el mismo DataFrame con columnas lat y lon añadidas.
    """
    log.info("Geocodificando %d estaciones con Nominatim…", len(stations))
    geolocator = Nominatim(user_agent="fusion_iimas_proyecto", timeout=10)

    lats, lons = [], []
    for _, row in stations.iterrows():
        parts = [str(row.get("city", "")), str(row.get("state", "")), "USA"]
        query = ", ".join(p for p in parts if p and p != "nan")
        try:
            time.sleep(1.1)
            loc = geolocator.geocode(query)
            if loc:
                lats.append(loc.latitude)
                lons.append(loc.longitude)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            log.warning("Geocodificación fallida para '%s': %s", query, exc)
            lats.append(np.nan)
            lons.append(np.nan)

    stations = stations.copy()
    stations["lat"] = lats
    stations["lon"] = lons

    n_fallidas = stations["lat"].isna().sum()
    if n_fallidas:
        log.warning("%d estaciones sin coordenadas tras geocodificación — se eliminarán.", n_fallidas)
        stations = stations[stations["lat"].notna()].reset_index(drop=True)

    return stations


def extract_stations(rad: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae una fila por estación con sus coordenadas.
    Prioriza columnas lat/lon existentes en radiation_clean;
    si no existen, geocodifica desde city + state.
    """
    lat_col, lon_col = _detect_lat_lon_cols(rad)

    # Columnas de contexto disponibles
    context_cols = ["station_id"]
    for c in ["state", "city"]:
        if c in rad.columns:
            context_cols.append(c)

    if lat_col and lon_col:
        log.info("Coordenadas detectadas en radiation_clean: '%s', '%s'", lat_col, lon_col)
        stations = (
            rad[context_cols + [lat_col, lon_col]]
            .dropna(subset=[lat_col, lon_col])
            .drop_duplicates(subset=["station_id"])
            .rename(columns={lat_col: "lat", lon_col: "lon"})
            .reset_index(drop=True)
        )
    else:
        log.info(
            "Columnas de coordenadas no encontradas en radiation_clean. "
            "Se geocodificará desde city/state."
        )
        stations = (
            rad[context_cols]
            .drop_duplicates(subset=["station_id"])
            .reset_index(drop=True)
        )
        stations = geocode_stations(stations)

    log.info("Estaciones con coordenadas: %d", len(stations))
    return stations


# ─────────────────────────────────────────────────────────────────────────────
# Preparación de plantas
# ─────────────────────────────────────────────────────────────────────────────

def prepare_plants(plants: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra a plantas con Status = Operational y coordenadas válidas.
    Selecciona y renombra columnas relevantes para la tabla de distancias.
    """
    # Normalizar nombres de columnas a minúsculas para detección robusta
    col_map = {c: c for c in plants.columns}  # identidad

    # Status
    status_col = next(
        (c for c in plants.columns if c.lower() == "status"), None
    )
    if status_col:
        plants = plants[plants[status_col].str.strip().str.lower() == "operational"].copy()
        log.info("Plantas con Status=Operational: %d", len(plants))
    else:
        log.warning("Columna 'Status' no encontrada — se usan todas las plantas.")

    # Coordenadas
    lat_col = next((c for c in plants.columns if c.lower() == "latitude"), None)
    lon_col = next((c for c in plants.columns if c.lower() == "longitude"), None)

    if not lat_col or not lon_col:
        raise ValueError("nuclear_plants_clean no tiene columnas Latitude/Longitude.")

    plants = plants.dropna(subset=[lat_col, lon_col]).copy()
    log.info("Plantas operacionales con coordenadas: %d", len(plants))

    # Detectar columnas de atributos
    def _find(candidates: list[str]) -> str | None:
        return next(
            (c for c in plants.columns if c.lower().replace("_", "") in
             [x.lower().replace("_", "") for x in candidates]),
            None
        )

    name_col         = _find(["name", "plant_name", "plantname"])
    country_col      = _find(["country", "country_code"])
    status_col_out   = _find(["status"])
    rt_std_col       = _find(["reactortype_std", "reactortype"])
    capacity_col     = _find(["capacity", "capacity_mw"])
    op_from_col      = _find(["operationalfrom", "operational_from"])

    rename = {
        lat_col: "plant_lat",
        lon_col: "plant_lon",
    }
    keep = [lat_col, lon_col]

    for src, dst in [
        (name_col,       "plant_name"),
        (country_col,    "plant_country"),
        (status_col_out, "plant_status"),
        (rt_std_col,     "reactor_type_std"),
        (capacity_col,   "capacity_mw"),
        (op_from_col,    "operational_from"),
    ]:
        if src:
            rename[src] = dst
            keep.append(src)

    plants = plants[keep].rename(columns=rename)
    return plants.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de la tabla de distancias
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_table(
    stations: pd.DataFrame,
    plants: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Calcula distancias Haversine (N_estaciones × M_plantas) y
    conserva las top_n plantas más cercanas por estación.
    """
    st_lat = stations["lat"].to_numpy()
    st_lon = stations["lon"].to_numpy()
    pl_lat = plants["plant_lat"].to_numpy()
    pl_lon = plants["plant_lon"].to_numpy()

    log.info(
        "Calculando matriz Haversine %d estaciones × %d plantas…",
        len(stations), len(plants)
    )
    dist_matrix = haversine_matrix(st_lat, st_lon, pl_lat, pl_lon)
    # dist_matrix[i, j] = distancia entre estación i y planta j

    rows = []
    for i, st_row in stations.iterrows():
        dists = dist_matrix[i]                        # (M,)
        idx_sorted = np.argsort(dists)[:top_n]       # índices de las top_n más cercanas

        for rank, j in enumerate(idx_sorted, start=1):
            pl_row = plants.iloc[j]
            entry = {
                "station_id":    st_row["station_id"],
                "station_lat":   st_row["lat"],
                "station_lon":   st_row["lon"],
            }
            # Añadir columnas de contexto de la estación si existen
            for c in ["state", "city"]:
                if c in stations.columns:
                    entry[f"station_{c}"] = st_row.get(c, np.nan)

            entry.update({
                "plant_name":        pl_row.get("plant_name",       np.nan),
                "plant_lat":         pl_row["plant_lat"],
                "plant_lon":         pl_row["plant_lon"],
                "plant_country":     pl_row.get("plant_country",    np.nan),
                "plant_status":      pl_row.get("plant_status",     np.nan),
                "reactor_type_std":  pl_row.get("reactor_type_std", np.nan),
                "capacity_mw":       pl_row.get("capacity_mw",      np.nan),
                "operational_from":  pl_row.get("operational_from", np.nan),
                "distance_km":       round(float(dists[j]), 3),
                "rank_proximity":    rank,
            })
            rows.append(entry)

    result = pd.DataFrame(rows)
    log.info("Tabla de distancias construida: %d filas", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Construcción de station_plant_distances con Haversine"
    )
    parser.add_argument(
        "--radiation",
        default=str(project_root / "datos_procesados" / "radiation_clean.csv"),
        help="Ruta a radiation_clean.csv",
    )
    parser.add_argument(
        "--plants",
        default=str(project_root / "datos_procesados" / "nuclear_plants_clean.csv"),
        help="Ruta a nuclear_plants_clean.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "datos_procesados"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Número de plantas más cercanas por estación (default: 5)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────────
    log.info("Cargando radiation_clean…")
    rad = pd.read_csv(args.radiation, low_memory=False)
    log.info("  %d filas, %d columnas", *rad.shape)

    log.info("Cargando nuclear_plants_clean…")
    plants_raw = pd.read_csv(args.plants, low_memory=False)
    log.info("  %d filas, %d columnas", *plants_raw.shape)

    # ── Preparar inputs ───────────────────────────────────────────────────
    stations = extract_stations(rad)
    plants   = prepare_plants(plants_raw)

    # ── Calcular distancias ───────────────────────────────────────────────
    dist_table = build_distance_table(stations, plants, top_n=args.top_n)

    # ── Guardar ───────────────────────────────────────────────────────────
    out_path = output_dir / "station_plant_distances.csv"
    dist_table.to_csv(out_path, index=False)
    log.info("✓ Guardado: %s (%d filas)", out_path, len(dist_table))

    # Resumen rápido
    log.info(
        "Distancia mínima promedio por estación: %.1f km",
        dist_table[dist_table["rank_proximity"] == 1]["distance_km"].mean()
    )
    log.info(
        "Distancia máxima promedio a la planta más cercana: %.1f km",
        dist_table[dist_table["rank_proximity"] == 1]["distance_km"].max()
    )
    log.info("Fusión completada.")


if __name__ == "__main__":
    main()
