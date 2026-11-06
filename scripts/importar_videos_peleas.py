"""Registra el Surveillance Camera Fight Dataset para entrenar los eventos de movimiento.

Dataset: Akti et al., "Vision-based Fight Detection from Surveillance Cameras"
(IPTA 2019), https://github.com/seymanurakti/fight-detection-surv-dataset,
licencia MIT. 300 clips de cámaras de vigilancia: 150 de pelea y 150 normales.

No existe un dataset público de peleas grabadas en aulas; este es el más cercano
que se puede descargar libremente: cámara fija, en alto, con gente de cuerpo
entero, como la cámara del proyecto.

Los clips salen de 44 videos de YouTube y de casi todos se cortaron clips de
pelea Y normales. Cada clip se registra con su video de origen como "sesión":
así el entrenador nunca evalúa un clip con otro del mismo video en el
entrenamiento (serían casi idénticos y el acierto saldría inflado).

Uso:
    python scripts/importar_videos_peleas.py --origen C:/Users/Adrian/datasets/fight_surv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

COLUMNAS = ["archivo", "clase", "sesion"]
SUBCARPETA = "surv"


def origen_por_clip(videos_txt: Path) -> dict[str, str]:
    """{'fi001': 'yt00', 'nofi003': 'yt00', ...} según el video de YouTube del que sale."""
    salida, actual, n_fuentes = {}, None, 0
    for linea in videos_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        linea = linea.strip()
        if linea.startswith("http"):
            actual = f"yt{n_fuentes:02d}"
            n_fuentes += 1
            continue
        # Al final del archivo: "nofi062 - nofi076 : CamNet Dataset" (clips tomados de otro dataset)
        rango = re.match(r"^nofi(\d+)\s*-\s*nofi(\d+)\s*:\s*(\w+)", linea)
        if rango:
            for i in range(int(rango.group(1)), int(rango.group(2)) + 1):
                salida.setdefault(f"nofi{i:03d}", rango.group(3).lower())
            continue
        m = re.match(r"^(n?o?fi\d+)\s*:", linea)
        if m and actual:
            salida[m.group(1)] = actual
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description="Registra el Surveillance Camera Fight Dataset")
    ap.add_argument("--origen", default=r"C:\Users\Adrian\datasets\fight_surv")
    args = ap.parse_args()

    origen = Path(args.origen)
    sesiones = origen_por_clip(origen / "videos.txt")
    destino = cfg.ruta_de("data_video_raw") / SUBCARPETA
    destino.mkdir(parents=True, exist_ok=True)

    ruta_labels = cfg.ruta_de("data_video_labels")
    filas = []
    if ruta_labels.exists():
        with ruta_labels.open(encoding="utf-8", newline="") as fh:
            filas = [f for f in csv.DictReader(fh) if not f["archivo"].startswith(f"{SUBCARPETA}/")]

    sin_origen = 0
    for carpeta, clase in (("fight", "movimiento_anomalo"), ("noFight", "normal")):
        for video in sorted((origen / carpeta).glob("*.mp4")):
            if not (destino / video.name).exists():
                shutil.copy2(video, destino / video.name)
            sesion = sesiones.get(video.stem)
            if sesion is None:
                sin_origen += 1
                sesion = f"surv_{video.stem}"
            filas.append({"archivo": f"{SUBCARPETA}/{video.name}", "clase": clase, "sesion": sesion})

    with ruta_labels.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)

    propias = [f for f in filas if not f["archivo"].startswith(f"{SUBCARPETA}/")]
    surv = [f for f in filas if f["archivo"].startswith(f"{SUBCARPETA}/")]
    print(f"Registrados {len(surv)} clips del dataset de vigilancia "
          f"({sum(f['clase'] == 'movimiento_anomalo' for f in surv)} pelea, "
          f"{sum(f['clase'] == 'normal' for f in surv)} normales) de "
          f"{len({f['sesion'] for f in surv})} videos de origen; {sin_origen} sin origen identificado.")
    print(f"Se conservan {len(propias)} videos propios: {[f['archivo'] for f in propias]}")
    print(f"-> {ruta_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
