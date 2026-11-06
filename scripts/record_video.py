"""Graba un video etiquetado para entrenar el clasificador de eventos.

El clasificador de eventos necesita ejemplos de las dos clases: `normal`
(actividad corriente del aula) y `movimiento_anomalo` (movimientos bruscos). Este
script graba la webcam y deja el video registrado en data/video/labels.csv con
su clase, listo para src/vision/train_events.py.

Lo más difícil de conseguir no son los ejemplos anómalos sino los normales: hace
falta bastante grabación de gente sentada, caminando, levantando la mano o
escribiendo, porque eso es lo que el modelo tiene que aprender a NO marcar.

Uso:
    python scripts/record_video.py --nombre aula_normal_01 --clase normal --segundos 120
    python scripts/record_video.py --nombre empujon_01 --clase movimiento_anomalo --segundos 20
    python scripts/record_video.py --registrar pelea.mp4 --clase movimiento_anomalo
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

COLUMNAS = ["archivo", "clase", "sesion"]


def leer_registro(ruta: Path) -> dict[str, dict[str, str]]:
    if not ruta.exists():
        return {}
    with ruta.open("r", encoding="utf-8", newline="") as fh:
        return {f["archivo"]: f for f in csv.DictReader(fh)}


def escribir_registro(filas: dict[str, dict[str, str]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        for k in sorted(filas):
            w.writerow(filas[k])


def registrar(archivo: str, clase: str, sesion: str) -> None:
    ruta = cfg.ruta_de("data_video_labels")
    filas = leer_registro(ruta)
    filas[archivo] = {"archivo": archivo, "clase": clase, "sesion": sesion}
    escribir_registro(filas, ruta)
    print(f"Registrado en {ruta}: {archivo} -> {clase} (sesión {sesion})")


def grabar(nombre: str, segundos: float, conf: dict, mostrar: bool) -> Path | None:
    import cv2

    cv = conf["vision"]
    fps = int(conf["grabacion"]["video_fps"])
    ancho, alto = int(cv["ancho"]), int(cv["alto"])
    destino = cfg.ruta_de("data_video_raw") / f"{nombre}.mp4"
    destino.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(cv["camara_index"]), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.", file=sys.stderr)
        return None

    escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    intervalo = 1.0 / fps
    fin = time.time() + segundos
    n = 0
    try:
        while time.time() < fin:
            proximo = time.time() + intervalo
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (ancho, alto))
            escritor.write(frame)
            n += 1
            if mostrar:
                cv2.putText(frame, f"GRABANDO {nombre}  {fin - time.time():4.0f}s", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.imshow("Grabacion", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"\r  grabando... {fin - time.time():5.0f} s", end="", flush=True)
            time.sleep(max(0.0, proximo - time.time()))
    finally:
        print()
        cap.release()
        escritor.release()
        cv2.destroyAllWindows()

    print(f"{n} cuadros guardados en {destino}")
    return destino


def main() -> int:
    conf = cfg.cargar()
    clases = conf["vision"]["clases_evento"]
    ap = argparse.ArgumentParser(description="Graba o registra videos para entrenar eventos")
    ap.add_argument("--nombre", help="nombre del video a grabar (sin extensión)")
    ap.add_argument("--registrar", help="registrar un archivo ya existente en data/video/raw")
    ap.add_argument("--clase", choices=clases, required=True)
    ap.add_argument("--sesion", default=None, help="día o lugar de grabación")
    ap.add_argument("--segundos", type=float, default=60.0)
    ap.add_argument("--sin-ventana", action="store_true")
    args = ap.parse_args()

    sesion = args.sesion or datetime.now().strftime("v%Y%m%d")

    if args.registrar:
        ruta = cfg.ruta_de("data_video_raw") / args.registrar
        if not ruta.exists():
            print(f"No existe {ruta}", file=sys.stderr)
            return 1
        registrar(args.registrar, args.clase, sesion)
        return 0

    if not args.nombre:
        ap.error("indica --nombre para grabar o --registrar ARCHIVO")

    print(f"Grabando '{args.nombre}' como '{args.clase}' durante {args.segundos:.0f} s.")
    destino = grabar(args.nombre, args.segundos, conf, not args.sin_ventana)
    if destino is None:
        return 1
    registrar(destino.name, args.clase, sesion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
