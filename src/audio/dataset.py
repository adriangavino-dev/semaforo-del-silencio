"""Lectura y escritura de data/audio/labels.csv.

El archivo tiene tres columnas: clip_id, clase y sesion. La columna `sesion` es
la que permite separar los datos por grabación al entrenar, que es lo que evita
que la exactitud salga inflada.

Ejecutable de forma independiente:
    python -m src.audio.dataset
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

COLUMNAS = ["clip_id", "clase", "sesion"]


def leer_etiquetas(ruta: Path | None = None) -> dict[str, dict[str, str]]:
    """Carga labels.csv indexado por clip_id. Devuelve vacío si no existe."""
    ruta = ruta or cfg.ruta_de("data_audio_labels")
    if not ruta.exists():
        return {}
    with ruta.open("r", encoding="utf-8", newline="") as fh:
        return {fila["clip_id"]: fila for fila in csv.DictReader(fh)}


def escribir_etiquetas(filas: dict[str, dict[str, str]], ruta: Path | None = None) -> None:
    """Guarda labels.csv ordenado por clip_id."""
    ruta = ruta or cfg.ruta_de("data_audio_labels")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=COLUMNAS)
        escritor.writeheader()
        for clip_id in sorted(filas):
            escritor.writerow(filas[clip_id])


def etiquetados(filas: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Solo los fragmentos que ya tienen clase asignada."""
    return [f for f in filas.values() if f.get("clase")]


def ruta_clip(fila: dict[str, str]) -> Path:
    """Ubicación en disco del WAV de un fragmento."""
    return cfg.ruta_de("data_audio_clips") / fila["sesion"] / f"{fila['clip_id']}.wav"


def conteo_por_clase(filas: dict[str, dict[str, str]]) -> Counter[str]:
    return Counter(f["clase"] or "(sin etiquetar)" for f in filas.values())


def conteo_por_sesion(filas: dict[str, dict[str, str]]) -> dict[str, Counter[str]]:
    salida: dict[str, Counter[str]] = {}
    for fila in filas.values():
        salida.setdefault(fila["sesion"], Counter())[f"{fila['clase'] or '(sin etiquetar)'}"] += 1
    return salida


if __name__ == "__main__":
    filas = leer_etiquetas()
    if not filas:
        print("No hay fragmentos registrados todavía.")
        raise SystemExit(0)
    print(f"{len(filas)} fragmentos registrados\n")
    for clase, n in conteo_por_clase(filas).most_common():
        print(f"  {clase:<18} {n:>5}")
    print()
    for sesion, cuenta in sorted(conteo_por_sesion(filas).items()):
        print(f"  {sesion:<18} {dict(cuenta)}")
