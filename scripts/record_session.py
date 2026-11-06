"""Graba audio y video de una sesión — Etapa 2.

El audio se guarda a 16 kHz mono en WAV, que es lo que exige YAMNet. El video se
guarda en MP4 y sirve para la Etapa 4; si solo se está juntando audio conviene
usar --sin-video.

Sobre la organización de las grabaciones
----------------------------------------
Cada grabación pertenece a una *sesión* y registra una *situación*. La sesión es
el día o el momento de grabación; la situación es cuál de las cuatro clases se
está representando.

    python scripts/record_session.py --sesion s01 --clase silencio     --minutos 2
    python scripts/record_session.py --sesion s01 --clase voz_docente  --minutos 2
    python scripts/record_session.py --sesion s01 --clase murmullo     --minutos 2
    python scripts/record_session.py --sesion s01 --clase ruido_brusco --minutos 2

Las cuatro clases deben grabarse dentro de cada sesión, y hay que repetir el
juego completo en al menos tres sesiones distintas. El motivo es que el
entrenamiento reserva sesiones enteras para la prueba: si una sesión tuviera una
sola clase, el conjunto de prueba quedaría con una sola clase y la evaluación no
mediría nada. Y si todo se graba el mismo día en el mismo sitio, el modelo
aprende a reconocer ese sitio en lugar del fenómeno.

Indicar --clase al grabar deja los fragmentos ya etiquetados y ahorra pasar por
label_tool.py. Si se graba una sesión mezclada, se omite --clase y se etiqueta
después a mano.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

SEPARADOR = "__"


def nombre_base(sesion: str, clase: str | None) -> str:
    """Nombre del archivo: la clase queda codificada tras un doble guion bajo."""
    return f"{sesion}{SEPARADOR}{clase}" if clase else sesion


def partir_nombre(stem: str) -> tuple[str, str]:
    """Inverso de nombre_base: devuelve (sesion, clase). La clase puede ir vacía."""
    if SEPARADOR in stem:
        sesion, clase = stem.split(SEPARADOR, 1)
        return sesion, clase
    return stem, ""


def grabar_audio(base: str, segundos: float, conf: dict, detener: threading.Event) -> Path:
    """Graba el micrófono completo a un WAV de 16 kHz mono."""
    import sounddevice as sd
    import soundfile as sf

    ca = conf["audio"]
    sr = int(ca["sample_rate"])
    destino = cfg.ruta_de("data_audio_raw") / f"{base}.wav"
    destino.parent.mkdir(parents=True, exist_ok=True)

    bloques: list[np.ndarray] = []

    def callback(indata, frames, tiempo, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        bloques.append(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    with sd.InputStream(
        samplerate=sr,
        channels=int(ca["canales"]),
        device=ca.get("dispositivo"),
        dtype="float32",
        callback=callback,
    ):
        inicio = time.time()
        fin = inicio + segundos
        while time.time() < fin and not detener.is_set():
            time.sleep(0.25)
            restante = fin - time.time()
            print(f"\r  grabando... {restante:5.0f} s restantes", end="", flush=True)
    print()

    onda = np.concatenate(bloques) if bloques else np.zeros(0, dtype=np.float32)
    sf.write(str(destino), onda, sr)
    print(f"[audio] {len(onda) / sr:.1f} s guardados en {destino}")
    return destino


def grabar_video(base: str, segundos: float, conf: dict, detener: threading.Event) -> Path | None:
    """Graba la webcam a un MP4."""
    import cv2

    cv = conf["vision"]
    fps = int(conf["grabacion"]["video_fps"])
    ancho, alto = int(cv["ancho"]), int(cv["alto"])

    destino = cfg.ruta_de("data_video_raw") / f"{base}.mp4"
    destino.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(cv["camara_index"]), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    if not cap.isOpened():
        print("[video] no se pudo abrir la cámara; la sesión sigue solo con audio", file=sys.stderr)
        return None

    escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    intervalo = 1.0 / fps
    fin = time.time() + segundos
    n = 0

    while time.time() < fin and not detener.is_set():
        proximo = time.time() + intervalo
        ok, frame = cap.read()
        if not ok:
            break
        escritor.write(cv2.resize(frame, (ancho, alto)))
        n += 1
        time.sleep(max(0.0, proximo - time.time()))

    cap.release()
    escritor.release()
    print(f"[video] {n} cuadros guardados en {destino}")
    return destino


def main() -> int:
    conf = cfg.cargar()
    clases = conf["audio"]["clases"]

    ap = argparse.ArgumentParser(
        description="Graba una sesión de audio y video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Graba las cuatro clases dentro de cada sesión, y repite en tres sesiones distintas.",
    )
    ap.add_argument("--sesion", help="identificador de la sesión, p. ej. s01")
    ap.add_argument("--clase", choices=clases, help="situación que se está grabando")
    ap.add_argument("--minutos", type=float, default=float(conf["grabacion"]["duracion_min"]))
    ap.add_argument("--sin-video", action="store_true", help="grabar solo audio")
    args = ap.parse_args()

    sesion = args.sesion or datetime.now().strftime("s_%Y%m%d_%H%M")
    base = nombre_base(sesion, args.clase)
    segundos = args.minutos * 60.0

    existente = cfg.ruta_de("data_audio_raw") / f"{base}.wav"
    if existente.exists():
        print(f"Ya existe {existente}. Usa otro --sesion o borra ese archivo.", file=sys.stderr)
        return 1

    print(f"Sesión '{sesion}'" + (f", clase '{args.clase}'" if args.clase else " (sin clase)"))
    print(f"Duración: {args.minutos:.1f} minutos. Ctrl+C corta antes de tiempo.\n")

    detener = threading.Event()
    hilo_video = None
    if not args.sin_video and conf["grabacion"]["guardar_video"]:
        hilo_video = threading.Thread(
            target=grabar_video, args=(base, segundos, conf, detener), daemon=True
        )
        hilo_video.start()

    try:
        grabar_audio(base, segundos, conf, detener)
    except KeyboardInterrupt:
        print("\nGrabación cortada por el usuario.")
    finally:
        detener.set()
        if hilo_video is not None:
            hilo_video.join(timeout=5.0)

    print(f"\nListo. Siguiente paso:\n    python scripts/make_clips.py --todas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
