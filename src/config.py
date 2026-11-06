"""Carga de config/config.yaml.

Restricción 5 de la especificación: todos los umbrales, rutas, clases y la IP
del ESP32 se leen de un único archivo. Ningún módulo debe tener valores fijos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# .../Semaforo del Silencio/src/config.py -> raíz del proyecto
RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "config" / "config.yaml"

_cache: dict[str, Any] | None = None


def cargar(ruta: Path | str | None = None, recargar: bool = False) -> dict[str, Any]:
    """Devuelve la configuración completa como diccionario."""
    global _cache
    if _cache is not None and not recargar and ruta is None:
        return _cache

    destino = Path(ruta) if ruta else RUTA_CONFIG
    if not destino.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {destino}")

    with destino.open("r", encoding="utf-8") as fh:
        datos = yaml.safe_load(fh)

    if ruta is None:
        _cache = datos
    return datos


def ruta_de(clave: str) -> Path:
    """Convierte una entrada de la sección `rutas` en una ruta absoluta."""
    rutas = cargar()["rutas"]
    if clave not in rutas:
        raise KeyError(f"'{clave}' no está definido en la sección rutas de config.yaml")
    return RAIZ / rutas[clave]


if __name__ == "__main__":
    import json

    print(json.dumps(cargar(), indent=2, ensure_ascii=False))
