"""Exporta el tablero a un solo archivo HTML.

El tablero normal pide los reportes al servidor local. Esta versión los deja
incrustados dentro del propio HTML, así que el archivo se puede mandar por correo
o abrir con doble clic sin tener Python corriendo.

Uso:
    python scripts/exportar_tablero.py
    python scripts/exportar_tablero.py --salida reports/tablero.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.fusion.reporter import listar_reportes


def exportar(salida: Path | None = None) -> Path:
    plantilla = cfg.ruta_de("web") / "index.html"
    if not plantilla.exists():
        raise FileNotFoundError(f"No se encontró {plantilla}")

    reportes = listar_reportes()
    if not reportes:
        raise SystemExit(
            "No hay reportes que exportar.\n"
            "Genera uno con: python scripts/simular_sesion.py"
        )

    html = plantilla.read_text(encoding="utf-8")
    datos = json.dumps({"reportes": reportes}, ensure_ascii=False)
    inyeccion = (
        # Abierto con doble clic no hay cabecera HTTP que declare la codificación,
        # así que hay que decirla en el propio archivo o los acentos salen rotos.
        '<meta charset="utf-8">\n'
        "<script>\n"
        f"window.DATOS_EMBEBIDOS = {datos};\n"
        "</script>\n"
    )
    # Se inserta antes del script principal para que ya esté definido al cargar.
    marca = "<script>\nconst ESTADOS"
    if marca not in html:
        raise RuntimeError("La plantilla cambió: no se encontró el punto de inserción.")
    html = html.replace(marca, inyeccion + marca, 1)

    salida = salida or cfg.ruta_de("reports") / f"tablero_{datetime.now():%Y%m%d_%H%M}.html"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exporta el tablero como un HTML autónomo")
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()

    ruta = exportar(args.salida)
    n = len(listar_reportes())
    print(f"{n} sesiones incrustadas en:\n  {ruta}")
