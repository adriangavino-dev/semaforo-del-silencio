"""Calibración de la línea base de altura — sección 6.2.

Se graban dos minutos con TODAS las personas sentadas y se ajusta una regresión
lineal que, dada la posición de una persona en la imagen, devuelve la altura de
hombro que tendría si estuviera sentada:

    y_hombro_esperado = a * x_centro + b * y_cadera + c

En inferencia, la diferencia entre esa altura esperada y la observada indica si
la persona está más alta de lo que debería, es decir, de pie.

Este paso es el que decide si el conteo de personas de pie funciona en un aula
real, porque con las carpetas tapando las piernas el ángulo de la rodilla no
sirve.

Uso:
    python -m src.vision.calibrate --camara 1
    python -m src.vision.calibrate --video data/video/raw/s01__silencio.mp4
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.features import LineaBase  # noqa: E402
from src.vision.pose import DetectorPose, dibujar  # noqa: E402


def ajustar(muestras: np.ndarray) -> tuple[LineaBase, float]:
    """Ajusta el plano por mínimos cuadrados. Devuelve (línea base, error medio)."""
    X = np.column_stack([muestras[:, 0], muestras[:, 1], np.ones(len(muestras))])
    y = muestras[:, 2]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = (float(v) for v in coef)
    error = float(np.abs(X @ coef - y).mean())
    return LineaBase(a, b, c, ajustada=True), error


def recolectar(fuente, detector: DetectorPose, conf: dict, segundos: float, mostrar: bool) -> np.ndarray:
    """Recorre los frames y junta (x_centro, y_cadera, y_hombro) por persona."""
    import cv2

    conf_kp = float(conf["vision"]["conf_keypoint"])
    muestras: list[tuple[float, float, float]] = []
    fin = time.time() + segundos
    frames = 0

    while time.time() < fin:
        ok, frame = fuente.read()
        if not ok:
            break
        frames += 1
        personas = detector.detectar(frame)
        for p in personas:
            hombro, c_hombro = p.punto_medio("hombro_izq", "hombro_der")
            cadera, c_cadera = p.punto_medio("cadera_izq", "cadera_der")
            if c_hombro < conf_kp or c_cadera < conf_kp:
                continue
            cx, _ = p.centro
            muestras.append((float(cx), float(cadera[1]), float(hombro[1])))

        if mostrar:
            dibujar(frame, personas)
            restante = max(0.0, fin - time.time())
            cv2.putText(
                frame, f"CALIBRANDO - todos sentados - {restante:4.0f}s - {len(muestras)} muestras",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("Calibracion de linea base", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print(f"  {frames} cuadros, {len(muestras)} muestras")
    return np.array(muestras, dtype=float)


def main() -> int:
    import argparse
    import cv2

    conf = cfg.cargar()
    cc = conf["vision"]["calibracion"]

    ap = argparse.ArgumentParser(description="Calibra la línea base de altura")
    ap.add_argument("--camara", type=int, default=None)
    ap.add_argument("--video", help="usar una grabación en lugar de la cámara")
    ap.add_argument("--segundos", type=float, default=float(cc["duracion_s"]))
    ap.add_argument("--sin-ventana", action="store_true")
    args = ap.parse_args()

    print("Cargando YOLOv8-Pose...")
    detector = DetectorPose(conf).cargar()

    if args.video:
        fuente = cv2.VideoCapture(args.video)
        segundos = 1e9  # el video termina solo
        mostrar = False
    else:
        indice = args.camara if args.camara is not None else int(conf["vision"]["camara_index"])
        fuente = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        fuente.set(cv2.CAP_PROP_FRAME_WIDTH, int(conf["vision"]["ancho"]))
        fuente.set(cv2.CAP_PROP_FRAME_HEIGHT, int(conf["vision"]["alto"]))
        segundos = args.segundos
        mostrar = not args.sin_ventana
        print(f"\nTODAS las personas deben estar SENTADAS durante {segundos:.0f} s.")
        print("Pulsa 'q' para cortar antes.\n")

    if not fuente.isOpened():
        print("No se pudo abrir la fuente de video.", file=sys.stderr)
        return 1

    try:
        muestras = recolectar(fuente, detector, conf, segundos, mostrar)
    finally:
        fuente.release()
        cv2.destroyAllWindows()

    minimo = int(cc["min_muestras"])
    if len(muestras) < minimo:
        print(
            f"Solo se juntaron {len(muestras)} muestras y hacen falta {minimo}.\n"
            "Asegúrate de que haya personas visibles y vuelve a intentarlo.",
            file=sys.stderr,
        )
        return 1

    base, error = ajustar(muestras)
    ruta = base.guardar(extra={
        "calibrado": datetime.now().isoformat(timespec="seconds"),
        "n_muestras": int(len(muestras)),
        "error_medio_px": error,
        "fuente": args.video or f"camara {args.camara if args.camara is not None else conf['vision']['camara_index']}",
    })

    print(f"\ny_hombro_esperado = {base.a:.4f}*x_centro + {base.b:.4f}*y_cadera + {base.c:.2f}")
    print(f"Error medio: {error:.1f} px sobre {len(muestras)} muestras")
    print(f"Guardado en {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
