"""Comprueba la regla de "cabeza baja" de src/vision/sleep.py con fotos reales de aulas.

Dataset: SCB-Dataset, subconjunto BowTurnHead (Yang et al.; Hugging Face
wintonYF/SCB-Dataset; uso académico). 2 410 fotos de cámaras de aulas con cada
alumno marcado como:
    BowHead   cabeza agachada (dormido, escribiendo o leyendo en el pupitre)
    TurnHead  cabeza girada hacia un lado, erguido

Para cada foto se corre YOLOv8-Pose, cada esqueleto se empareja con la caja
etiquetada que más se le solapa (IoU >= 0,3) y se calcula el indicador
(y_hombros - y_cabeza) / torso. Si la regla funciona, BowHead debería quedar
por debajo de vision.dormido.umbral_cabeza y TurnHead por encima.

No mide "dormido" (una foto no tiene tiempo ni movimiento): mide solo la parte
de la regla que dice si la cabeza está baja.

Uso:
    python scripts/evaluar_cabeza_scb.py --carpeta C:/Users/Adrian/datasets/scb_bowturnhead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.pose import Persona  # noqa: E402
from src.vision.sleep import altura_cabeza  # noqa: E402

CLASES = {0: "BowHead", 1: "TurnHead"}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=r"C:\Users\Adrian\datasets\scb_bowturnhead")
    ap.add_argument("--max-fotos", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    conf = cfg.cargar()
    umbral = float(conf["vision"]["dormido"]["umbral_cabeza"])
    modelo = YOLO(str(cfg.RAIZ / "models" / conf["vision"]["modelo_pose"]))

    fotos = sorted(Path(args.carpeta).rglob("images/**/*.jpg"))
    if args.max_fotos:
        fotos = fotos[: args.max_fotos]
    valores: dict[str, list[float]] = {"BowHead": [], "TurnHead": []}
    sin_esqueleto = {"BowHead": 0, "TurnHead": 0}

    for k, foto in enumerate(fotos, 1):
        etiquetas = Path(str(foto).replace("images", "labels")).with_suffix(".txt")
        if not etiquetas.exists():
            continue
        img = cv2.imread(str(foto))
        if img is None:
            continue
        alto, ancho = img.shape[:2]
        cajas = []
        for linea in etiquetas.read_text().split("\n"):
            partes = linea.split()
            if len(partes) != 5 or int(partes[0]) not in CLASES:
                continue
            c, cx, cy, w, h = int(partes[0]), *map(float, partes[1:])
            cajas.append((CLASES[c], ((cx - w / 2) * ancho, (cy - h / 2) * alto,
                                      (cx + w / 2) * ancho, (cy + h / 2) * alto)))
        r = modelo.predict(img, verbose=False, classes=[0], conf=0.25)[0]
        personas = []
        if r.keypoints is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                personas.append(Persona(
                    id=i, recuadro=tuple(r.boxes.xyxy[i].tolist()),
                    keypoints=r.keypoints.xy[i].cpu().numpy(),
                    confianzas=r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17),
                ))
        for clase, caja in cajas:
            mejor = max(personas, key=lambda p: iou(p.recuadro, caja), default=None)
            if mejor is None or iou(mejor.recuadro, caja) < 0.3:
                sin_esqueleto[clase] += 1
                continue
            a = altura_cabeza(mejor)
            if a is not None:
                valores[clase].append(a)
        if k % 200 == 0:
            print(f"  {k}/{len(fotos)} fotos", flush=True)

    print(f"\nUmbral actual de cabeza baja (vision.dormido.umbral_cabeza): {umbral}")
    resumen = {"umbral": umbral, "fotos": len(fotos), "clases": {}}
    for clase, v in valores.items():
        v = np.array(v)
        if not len(v):
            continue
        pct = np.percentile(v, [10, 25, 50, 75, 90]).round(2).tolist()
        bajo = float((v <= umbral).mean())
        resumen["clases"][clase] = {"n": int(len(v)), "sin_esqueleto": sin_esqueleto[clase],
                                    "percentiles_10_25_50_75_90": pct, "fraccion_bajo_umbral": bajo}
        print(f"  {clase:<9} {len(v):>5} alumnos con esqueleto ({sin_esqueleto[clase]} sin esqueleto) | "
              f"percentiles 10/25/50/75/90: {pct} | cabeza baja según la regla: {100 * bajo:.1f}%")

    # Qué umbral separa mejor (exactitud balanceada BowHead vs TurnHead).
    b, t = np.array(valores["BowHead"]), np.array(valores["TurnHead"])
    if len(b) and len(t):
        mejores = []
        for u in np.arange(-0.3, 0.61, 0.05):
            eb = 0.5 * ((b <= u).mean() + (t > u).mean())
            mejores.append((eb, round(float(u), 2)))
        eb, u = max(mejores)
        resumen["mejor_umbral"] = {"umbral": u, "exactitud_balanceada": round(float(eb), 3)}
        print(f"\n  Umbral que mejor separa: {u} (exactitud balanceada {eb:.3f})")
        actual = 0.5 * ((b <= umbral).mean() + (t > umbral).mean())
        print(f"  Con el umbral actual {umbral}: exactitud balanceada {actual:.3f}")
        resumen["exactitud_balanceada_umbral_actual"] = round(float(actual), 3)

    salida = cfg.ruta_de("reports") / "evaluacion_cabeza_scb.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
