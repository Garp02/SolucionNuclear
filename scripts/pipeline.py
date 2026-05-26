"""
pipeline.py — Ejecución secuencial del pipeline de calidad de datos

Orden:
  1. perfilado.py
  2. limpieza.py
  3. fusion.py
  4. analisis.py

Uso:
  python scripts/pipeline.py
  python scripts/pipeline.py --desde limpieza # reanudar desde un paso
  python scripts/pipeline.py --hasta fusion # detenerse después de fusion
  python scripts/pipeline.py --skip-geocoding # pasa el flag a limpieza y fusion
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
PASOS = ["perfilado", "limpieza", "fusion", "analisis"]

def run(script: str, extra_args: list[str]) -> None:
    """Ejecuta un script de Python y lanza SystemExit si falla."""
    ruta = SCRIPTS_DIR / f"{script}.py"
    cmd = [sys.executable, str(ruta)] + extra_args
    log.info("Iniciando: %s", script)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:

        log.error("%s falló con código %d — pipeline detenido.", script, result.returncode)
        sys.exit(result.returncode)

    log.info("%s completado en %.1f s", script, elapsed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de calidad de datos de forma secuencial")
    parser.add_argument(
        "--desde",
        choices=PASOS,
        default=PASOS[0],
        help="Paso desde el que iniciar (default: perfilado)",
    )
    parser.add_argument(
        "--hasta",
        choices=PASOS,
        default=PASOS[-1],
        help="Paso en el que detenerse, inclusive (default: analisis)",
    )
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Pasa --skip-geocoding a limpieza.py y fusion.py",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    idx_desde = PASOS.index(args.desde)
    idx_hasta = PASOS.index(args.hasta)
    pasos_a_ejecutar = PASOS[idx_desde : idx_hasta + 1]

    log.info("Pipeline: %s", " -> ".join(pasos_a_ejecutar))
    t_total = time.time()

    for paso in pasos_a_ejecutar:
        
        extra: list[str] = []
        
        if args.skip_geocoding and paso in ("limpieza", "fusion"):
            
            extra.append("--skip-geocoding")
        
        run(paso, extra)

    log.info("Pipeline completo en %.1f s", time.time() - t_total)

if __name__ == "__main__":

    main()
