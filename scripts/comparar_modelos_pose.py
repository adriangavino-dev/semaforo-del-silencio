"""Compara modelos de YOLOv8-Pose: personas encontradas y velocidad en CPU.

    python scripts/comparar_modelos_pose.py                     # yolov8n-pose vs yolov8s-pose
    python scripts/comparar_modelos_pose.py --modelos yolov8n-pose.pt yolov8m-pose.pt

Mide, con el mismo procedimiento que scripts/verificar_targets.py:
  - SCUT-HEAD Part A (fotos de cámaras de aulas, cabezas marcadas a mano):
    % de personas marcadas que el modelo encuentra.
  - normal_aula_01.mp4 (video propio): personas promedio por cuadro.
  - Velocidad: cuadros por segundo en este equipo, solo el modelo.
Los modelos que no estén en models/ se descargan solos de Ultralytics la primera vez.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

SCUT = Path(r"C:\Users\Adrian\datasets\scut_head\SCUT_HEAD_Part_A")


def cargar(nombre: str):
    from ultralytics import YOLO

    ruta = cfg.RAIZ / "models" / nombre
    if not ruta.exists():
        modelo = YOLO(nombre)                      # descarga en la carpeta actual
        descargado = Path(nombre)
        if descargado.exists():
            shutil.move(str(descargado), ruta)
        return YOLO(str(ruta))
    return YOLO(str(ruta))


def medir(nombre: str, n_fotos: int) -> dict:
    import cv2

    m = cargar(nombre)
    ids = (SCUT / "ImageSets" / "Main" / "test.txt").read_text().split()[:n_fotos]
    total = encontradas = 0
    tiempos = []
    for i in ids:
        img = cv2.imread(str(SCUT / "JPEGImages" / f"{i}.jpg"))
        cabezas = [tuple(float(o.find("bndbox").find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                   for o in ET.parse(SCUT / "Annotations" / f"{i}.xml").getroot().iter("object")]
        t0 = time.perf_counter()
        r = m.predict(img, verbose=False, classes=[0], conf=0.25)[0]
        tiempos.append(time.perf_counter() - t0)
        libres = list(range(len(cabezas)))
        kp = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints is not None and r.keypoints.conf is not None else []
        for j in range(len(r.boxes)):
            vis = [kp[j][q] for q in range(5) if len(kc) and kc[j][q] > 0.3]
            x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
            punto = np.mean(vis, axis=0) if vis else np.array([(x1 + x2) / 2, y1 + 0.1 * (y2 - y1)])
            for h in libres:
                a, b, c, d = cabezas[h]
                mx, my = 0.5 * (c - a), 0.5 * (d - b)
                if a - mx <= punto[0] <= c + mx and b - my <= punto[1] <= d + my:
                    libres.remove(h)
                    encontradas += 1
                    break
        total += len(cabezas)

    cap = cv2.VideoCapture(str(cfg.ruta_de("data_video_raw") / "normal_aula_01.mp4"))
    personas, n = [], 0
    while n < 150:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % 5 == 0:
            personas.append(len(m.predict(frame, verbose=False, classes=[0], conf=0.25)[0].boxes))
    cap.release()
    fps = 1.0 / float(np.median(tiempos[2:])) if len(tiempos) > 3 else 0.0
    peso = (cfg.RAIZ / "models" / nombre).stat().st_size / 1e6
    return {"modelo": nombre, "peso_mb": round(peso, 1), "scut_detectadas": encontradas / max(total, 1),
            "personas_por_cuadro_aula": float(np.mean(personas)) if personas else 0.0, "fps_cpu": fps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["yolov8n-pose.pt", "yolov8s-pose.pt"])
    ap.add_argument("--fotos", type=int, default=150)
    args = ap.parse_args()
    filas = []
    for nombre in args.modelos:
        r = medir(nombre, args.fotos)
        filas.append(r)
        print(f"{r['modelo']:<18} {r['peso_mb']:>5} MB | SCUT-HEAD detectadas {100 * r['scut_detectadas']:5.1f}% | "
              f"personas/cuadro en el aula {r['personas_por_cuadro_aula']:4.1f} | {r['fps_cpu']:4.1f} FPS", flush=True)
    salida = cfg.ruta_de("reports") / "comparacion_modelos_pose.json"
    salida.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
