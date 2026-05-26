"""
epa_radnet_api.py  v2.0
========================
API Wrapper para la descarga automatizada de datos de la red
RadNet de la EPA (Environmental Protection Agency).

URL fuente: https://www.epa.gov/radnet/radnet-csv-file-downloads

Estructura de la página:
La página NO contiene enlaces directos a archivos .csv. Ofrece dos tipos:

  1. REST API (año en curso):
       https://radnet.epa.gov/cdx-radnet-rest/api/rest/csv/{año}/fixed/{ESTADO}/{CIUDAD}
       Devuelve un CSV en texto plano. No tiene extensión .csv en la URL.

  2. ZIP histórico (años anteriores):
       https://www.epa.gov/system/files/other-files/{fecha}/{state}_{city}_{años}.zip
       Archivo ZIP que contiene un CSV dentro.

Este wrapper detecta y descarga ambos tipos.

Instalación de dependencias:
    conda install -c conda-forge requests beautifulsoup4

Python : 3.8+
"""

import re
import time
import logging
import zipfile
from enum import Enum
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


# Logging
logging.basicConfig(level = logging.INFO,
 format = "%(asctime)s [%(levelname)s] %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# Enumerado: tipo de recurso
class ResourceType(str, Enum):
    API_CSV = "api_csv" # REST endpoint a CSV en texto plano
    ZIP = "zip" # Archivo ZIP histórico con CSV dentro


# Dataclass: metadatos de cada recurso encontrado
@dataclass
class RadNetFile:
    """
    Representa un recurso de datos disponible en la página de RadNet.

    Atributos
    ---------
    filename : str
        Nombre de archivo local (ej. 'TX_SAN_ANGELO_2026.csv').
    url : str
        URL completa del recurso.
    state : str
        Código de estado de dos letras (ej. 'TX').
    city : str
        Nombre de la ciudad en mayúsculas (ej. 'SAN_ANGELO').
    resource_type : ResourceType
        Indica si es un endpoint REST o un ZIP histórico.
    size_kb : float | None
        Tamaño en KB tras la descarga.
    downloaded : bool
        True si el archivo ya fue guardado en disco.
    """
    filename: str
    url: str
    state: str
    city: str
    resource_type: ResourceType
    size_kb: Optional[float] = None
    downloaded: bool = False

    def __str__(self) -> str:
        size_str = f"({self.size_kb:.1f} KB)" if self.size_kb else ""
        tag = "API " if self.resource_type == ResourceType.API_CSV else "ZIP "
        status = "ok" if self.downloaded else "  "
        return f"[{status}][{tag}] {self.filename}{size_str}"


# Clase principal
class EPARadNetAPI:
    """
    API Wrapper para la página de descargas de RadNet (EPA).

    Detecta y descarga los dos tipos de recursos disponibles:
      - Endpoints REST del año en curso (devuelven CSV directamente).
      - Archivos ZIP con datos históricos.

    Parámetros
    ----------
    output_dir : str | Path
        Directorio local donde se guardan los archivos.  Default: './datos'.
    timeout : int
        Segundos máximos por petición HTTP.  Default: 30.
    delay : float
        Pausa en segundos entre descargas consecutivas.  Default: 1.5.
    extract_zips : bool
        Si True, extrae los CSV dentro de cada ZIP y borra el ZIP original.
        Default: True.
    headers : dict | None
        Cabeceras HTTP personalizadas (User-Agent, etc.).
    """

    _BASE_URL = "https://www.epa.gov/radnet/radnet-csv-file-downloads"

    # Patrón del endpoint REST:
    # https://radnet.epa.gov/cdx-radnet-rest/api/rest/csv/2026/fixed/TX/SAN_ANGELO
    _RE_API = re.compile(
        r"https://radnet\.epa\.gov/cdx-radnet-rest/api/rest/csv"
        r"/(\d{4})/fixed/([A-Za-z]{2})/([A-Za-z0-9_%+\-]+)", re.IGNORECASE)

    _DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    def __init__(self, timeout: int = 30, delay: float = 1.5, extract_zips: bool = True, 
        output_dir: str | Path = "/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos",
        headers: Optional[dict] = None) -> None:
        
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.delay = delay
        self.extract_zips = extract_zips
        self._session = requests.Session()
        self._session.headers.update(headers or self._DEFAULT_HEADERS)
        self._catalog: list[RadNetFile] = []
        self.output_dir.mkdir(parents = True, exist_ok = True)
        logger.info("Directorio de descarga: %s", self.output_dir.resolve())

    # Helpers privados
    def _get_page(self, url: str) -> BeautifulSoup:
        """Descarga y parsea la página HTML de RadNet."""
        try:

            logger.info("Obteniendo página: %s", url)
            r = self._session.get(url, timeout = self.timeout)
            r.raise_for_status()
            
            return BeautifulSoup(r.text, "html.parser")
        
        except requests.exceptions.Timeout:
        
            logger.error("Timeout al conectar con: %s", url)
            raise
        
        except requests.exceptions.ConnectionError as exc:
        
            logger.error("Error de conexión: %s", exc)
            raise
        
        except requests.exceptions.HTTPError as exc:
        
            logger.error("Error HTTP: %s", exc)
            raise

    def _parse_catalog(self, soup: BeautifulSoup) -> list[RadNetFile]:
        """
        Recorre todas las etiquetas <a> y clasifica cada enlace en:
          - ResourceType.API_CSV si apunta al endpoint REST de RadNet.
          - ResourceType.ZIP si termina en .zip.
        """
        files: list[RadNetFile] = []
        seen: set[str] = set()

        for tag in soup.find_all("a", href=True):
            
            href: str = tag["href"].strip()
            absolute = href if href.startswith("http") else urljoin(self._BASE_URL, href)

            if absolute in seen:
                
                continue
            
            seen.add(absolute)

            # Tipo 1: endpoint REST
            m = self._RE_API.match(absolute)
            if m:
            
                year  = m.group(1)
                state = m.group(2).upper()
                city  = m.group(3).upper().replace("%20", "_").replace("+", "_")
                files.append(RadNetFile(filename = f"{state}_{city}_{year}.csv", url = absolute, 
                                        state = state, city = city, resource_type = ResourceType.API_CSV))
                
                continue

            # Tipo 2: ZIP histórico
            if absolute.lower().endswith(".zip"):

                filename = Path(urlparse(absolute).path).name
                parts = filename.lower().replace(".zip", "").split("_")
                state = parts[0].upper() if parts else "??"
                city = "_".join(parts[1:-1]).upper() if len(parts) > 2 else "??"
                files.append(RadNetFile(filename = filename, url = absolute,
                    state = state, city = city, resource_type = ResourceType.ZIP))

        return files

    def _download_api_csv(self, rfile: RadNetFile) -> Path:
        """Descarga un endpoint REST que devuelve texto CSV."""
        dest = self.output_dir / rfile.filename

        if dest.exists():
        
            logger.info("Ya existe, se omite: %s", rfile.filename)
            rfile.downloaded = True

            return dest
        
        try:
        
            r = self._session.get(rfile.url, timeout=self.timeout)
            r.raise_for_status()
            dest.write_text(r.text, encoding="utf-8")
            rfile.size_kb = len(r.content) / 1024
            rfile.downloaded = True
            logger.info("ok %s (%.1f KB)", rfile.filename, rfile.size_kb)
        
            return dest
        
        except requests.exceptions.RequestException as exc:
        
            logger.error("Error al descargar %s: %s", rfile.filename, exc)
            raise

    def _download_zip(self, rfile: RadNetFile) -> list[Path]:
        """
        Descarga un ZIP histórico.
        Si extract_zips=True, extrae el CSV interno y borra el ZIP.
        Retorna la lista de rutas guardadas.
        """
        zip_dest = self.output_dir / rfile.filename
        saved_paths: list[Path] = []

        if not zip_dest.exists():

            try:
                r = self._session.get(rfile.url, timeout = self.timeout, stream = True)
                r.raise_for_status()
                total = 0
            
                with open(zip_dest, "wb") as f:
            
                    for chunk in r.iter_content(chunk_size = 65536):
            
                        if chunk:
            
                            f.write(chunk)
                            total += len(chunk)
            
                rfile.size_kb = total / 1024
                logger.info("ok %s (%.1f KB)", rfile.filename, rfile.size_kb)
            
            except requests.exceptions.RequestException as exc:
            
                logger.error("Error al descargar %s: %s", rfile.filename, exc)
                raise

        if self.extract_zips:
            
            try:
            
                with zipfile.ZipFile(zip_dest, "r") as zf:
            
                    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            
                    if not csv_names:
            
                        logger.warning("El ZIP no contiene CSV: %s", rfile.filename)
                        saved_paths.append(zip_dest)
            
                    else:
            
                        for csv_name in csv_names:
            
                            out_csv = self.output_dir / Path(csv_name).name
                            out_csv.write_bytes(zf.read(csv_name))
                            saved_paths.append(out_csv)
                            logger.info("Extraido: %s", out_csv.name)
            
                zip_dest.unlink()
            
            except zipfile.BadZipFile as exc:
            
                logger.error("ZIP corrupto %s: %s", rfile.filename, exc)
                saved_paths.append(zip_dest)
        
        else:
        
            saved_paths.append(zip_dest)

        rfile.downloaded = True
        return saved_paths

    def _download_one(self, rfile: RadNetFile) -> list[Path]:
        """Despacha la descarga al método correcto según el tipo de recurso."""
        if rfile.resource_type == ResourceType.API_CSV:
        
            return [self._download_api_csv(rfile)]
        
        return self._download_zip(rfile)

    # API pública

    def fetch_catalog(self, force_refresh: bool = False) -> list[RadNetFile]:
        """
        Obtiene (y cachea) el catálogo de recursos disponibles.

        Parámetros
        ----------
        force_refresh : bool
            Si True, vuelve a inspeccionar la página ignorando el caché.
        """
        if self._catalog and not force_refresh:
        
            logger.info("Catálogo en caché: %d recursos.", len(self._catalog))
        
            return self._catalog

        soup = self._get_page(self._BASE_URL)
        self._catalog = self._parse_catalog(soup)
        api_n = sum(1 for f in self._catalog if f.resource_type == ResourceType.API_CSV)
        zip_n = sum(1 for f in self._catalog if f.resource_type == ResourceType.ZIP)
        logger.info("Catálogo: %d recursos  (%d endpoints REST, %d ZIPs históricos).",
            len(self._catalog), api_n, zip_n)
        
        return self._catalog

    def list_files(
        self,
        keyword: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
    ) -> list[RadNetFile]:
        """
        Lista los recursos disponibles con filtros opcionales.

        Parámetros
        ----------
        keyword : str | None
            Filtra por nombre, estado o ciudad (insensible a mayúsculas).
            Ejemplos: 'TX', 'Montana', 'Pittsburgh', '2025'.
        resource_type : ResourceType | None
            Si se especifica, devuelve solo ese tipo (API_CSV o ZIP).
        """
        catalog = self.fetch_catalog()
        result  = catalog

        if resource_type:
        
            result = [f for f in result if f.resource_type == resource_type]

        if keyword:
            
            kw = keyword.lower()
            result = [f for f in result if kw in f.filename.lower()
                or kw in f.state.lower() or kw in f.city.lower()]
            logger.info("Filtro '%s': %d / %d recursos coinciden.",
                keyword, len(result), len(catalog))

        return result

    def print_catalog(
        self,
        keyword: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
    ) -> None:
        """Imprime el catálogo filtrado en consola."""
        files = self.list_files(keyword, resource_type)
        title = "RadNet — recursos disponibles"
        
        if keyword:
        
            title += f" [filtro: '{keyword}']"
        
        if resource_type:
        
            title += f" [tipo: {resource_type.value}]"

        print(f"{title} ({len(files)} recursos)")
        
        if not files:
        
            print("(ningún recurso coincide con el criterio)")
        
        for i, f in enumerate(files, 1):
        
            print(f"{i:>4}. {f}")

    def download(self, keyword: str, resource_type: Optional[ResourceType] = None,
        max_files: Optional[int] = None) -> list[Path]:
        """
        Descarga los recursos cuyo nombre/estado/ciudad contenga keyword.

        Parámetros
        ----------
        keyword : str
            Texto de búsqueda (ej. 'TX', 'Montana', 'Pittsburgh', '2026').
        resource_type : ResourceType | None
            Restringe a un solo tipo de recurso.
        max_files : int | None
            Límite de descargas (útil para pruebas).

        Retorna
        -------
        list[Path]  rutas locales de los archivos guardados.
        """
        matches = self.list_files(keyword, resource_type)
        if not matches:

            logger.warning("No se encontraron recursos con el filtro '%s'.", keyword)
            return []

        if max_files:
            
            matches = matches[:max_files]

        all_paths: list[Path] = []
        
        for i, rfile in enumerate(matches):
        
            try:
        
                paths = self._download_one(rfile)
                all_paths.extend(paths)
        
                if i < len(matches) - 1:
        
                    time.sleep(self.delay)
        
            except Exception:
        
                logger.warning("Se omite '%s' por error.", rfile.filename)

        logger.info("Descarga completada: %d archivo(s) en '%s'.", len(all_paths), self.output_dir)
        
        return all_paths

    def download_all(self, resource_type: Optional[ResourceType] = None,
        max_files: Optional[int] = None) -> list[Path]:
        """
        Descarga todos los recursos del catálogo.

        Parámetros
        ----------
        resource_type : ResourceType | None
            Restringe a API_CSV o ZIP.
        max_files : int | None
            Límite de descargas (recomendado para pruebas).
        """
        all_files = self.list_files(resource_type = resource_type)
        if max_files:
        
            all_files = all_files[:max_files]

        all_paths: list[Path] = []
        
        for i, rfile in enumerate(all_files):
        
            try:
        
                paths = self._download_one(rfile)
                all_paths.extend(paths)
        
                if i < len(all_files) - 1:
        
                    time.sleep(self.delay)
        
            except Exception:
                logger.warning("Se omite '%s' por error.", rfile.filename)

        logger.info("download_all completado: %d archivo(s) en '%s'.",
            len(all_paths), self.output_dir)
        
        return all_paths

    def summary(self) -> dict:
        """Retorna un resumen del estado actual del catálogo."""
        total = len(self._catalog)
        downloaded = sum(1 for f in self._catalog if f.downloaded)
        api_total = sum(1 for f in self._catalog if f.resource_type == ResourceType.API_CSV)
        zip_total = sum(1 for f in self._catalog if f.resource_type == ResourceType.ZIP)
        return {
            "total_recursos": total,
            "api_csv": api_total,
            "zips": zip_total,
            "descargados": downloaded,
            "pendientes": total - downloaded,
            "output_dir": str(self.output_dir.resolve()),
        }

    def __repr__(self) -> str:
        return (
            f"EPARadNetAPI(output_dir='{self.output_dir}', "
            f"timeout={self.timeout}s, delay={self.delay}s, "
            f"extract_zips={self.extract_zips})"
        )

if __name__ == "__main__":

    api = EPARadNetAPI(output_dir = "/home/ibra/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final/datos", 
                       timeout = 30, delay = 1.5, extract_zips = True)
    api.print_catalog()
    rutas = api.download_all()