"""Demostración en vivo — Etapa 1 de la especificación.

Muestra en una sola ventana:
  - la webcam con los esqueletos que detecta YOLOv8-Pose y el conteo de personas;
  - las clases de sonido que YAMNet reconoce por defecto sobre el micrófono, con
    el nivel en dB relativos.

Ninguno de los dos modelos ha sido entrenado por el equipo: ambos son
preentrenados y aquí se usan tal cual. Esto es lo que permite tener algo
demostrable antes de haber grabado el conjunto de datos propio.

Lo que esta etapa NO hace todavía: clasificar las cuatro clases del proyecto
(silencio / voz_docente / murmullo / ruido_brusco), clasificar postura, fusionar
estados ni comunicarse con el ESP32.

Uso:
    python scripts/demo_live.py
    python scripts/demo_live.py --sin-audio      # solo visión
    python scripts/demo_live.py --camara 1
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.audio.capture import CapturaAudio
from src.audio.features import Yamnet
from src.vision.pose import DetectorPose, dibujar

BLANCO = (255, 255, 255)
GRIS = (170, 170, 170)
AMBAR = (0, 200, 255)
VERDE = (0, 255, 120)


class HiloAudio(threading.Thread):
    """Captura y clasifica audio en un hilo aparte.

    La inferencia de audio tarda decenas de milisegundos; ejecutarla en el bucle
    de video haría caer los FPS por debajo del mínimo exigido (sección 10). El
    bucle principal solo lee el último resultado publicado por este hilo.
    """

    def __init__(self, configuracion: dict) -> None:
        super().__init__(daemon=True)
        self.conf = configuracion
        self.captura = CapturaAudio(configuracion)
        self.yamnet = Yamnet(configuracion)
        self._detener = threading.Event()
        self._lock = threading.Lock()

        # Ventana deslizante de la clase dominante (sección 5.5).
        self._historial: deque[str] = deque(maxlen=int(configuracion["audio"]["ventana_moda"]))
        self.top: list[tuple[str, float]] = []
        self.db: float = 0.0
        self.listo = False
        self.error: str | None = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            print("[audio] cargando YAMNet (la primera vez descarga ~17 MB)...")
            self.yamnet.cargar()
            print(f"[audio] YAMNet listo: {len(self.yamnet.nombres_clases)} clases de AudioSet")
            self.captura.iniciar()
            with self._lock:
                self.listo = True
        except Exception as exc:                      # noqa: BLE001
            with self._lock:
                self.error = str(exc)
            print(f"[audio] error de inicialización: {exc}", file=sys.stderr)
            return

        while not self._detener.is_set():
            bloque = self.captura.leer(timeout=0.5)
            if bloque is None:
                continue
            try:
                top = self.yamnet.clases_principales(bloque)
                db = self.captura.db(bloque)
            except Exception as exc:                  # noqa: BLE001
                print(f"[audio] error de inferencia: {exc}", file=sys.stderr)
                continue
            with self._lock:
                self.top = top
                self.db = db
                if top:
                    self._historial.append(top[0][0])

        self.captura.detener()

    # ------------------------------------------------------------------
    def estado(self) -> dict:
        with self._lock:
            historial = list(self._historial)
            dominante = max(set(historial), key=historial.count) if historial else "—"
            return {
                "top": list(self.top),
                "db": self.db,
                "dominante": dominante,
                "listo": self.listo,
                "error": self.error,
            }

    def detener(self) -> None:
        self._detener.set()


def dibujar_panel(frame: np.ndarray, lineas: list[tuple[str, tuple[int, int, int], float]]) -> None:
    """Panel semitransparente con las lecturas de ambos modelos."""
    alto_panel = 22 * len(lineas) + 16
    capa = frame.copy()
    cv2.rectangle(capa, (0, 0), (frame.shape[1], alto_panel), (0, 0, 0), -1)
    cv2.addWeighted(capa, 0.55, frame, 0.45, 0, frame)

    y = 24
    for texto, color, escala in lineas:
        cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)
        y += 22


def main() -> int:
    ap = argparse.ArgumentParser(description="Semáforo del Silencio — demo Etapa 1")
    ap.add_argument("--camara", type=int, default=None, help="índice de la webcam")
    ap.add_argument("--sin-audio", action="store_true", help="ejecutar solo el módulo de visión")
    args = ap.parse_args()

    conf = cfg.cargar()
    conf_vision = conf["vision"]
    indice = args.camara if args.camara is not None else int(conf_vision["camara_index"])

    hilo_audio = None
    if not args.sin_audio:
        hilo_audio = HiloAudio(conf)
        hilo_audio.start()

    print("[vision] cargando YOLOv8-Pose (la primera vez descarga los pesos)...")
    detector = DetectorPose(conf).cargar()
    print("[vision] YOLOv8-Pose listo")

    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(conf_vision["ancho"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(conf_vision["alto"]))
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {indice}. Prueba con --camara 1.", file=sys.stderr)
        if hilo_audio:
            hilo_audio.detener()
        return 1

    print("Demo en marcha. Pulsa 'q' para salir.")
    fps_hist: deque[float] = deque(maxlen=30)
    ventana = "Semaforo del Silencio - Etapa 1 (modelos preentrenados)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("La cámara dejó de entregar frames.", file=sys.stderr)
                break

            t0 = time.time()
            personas = detector.detectar(frame)
            dibujar(frame, personas)
            fps_hist.append(1.0 / max(time.time() - t0, 1e-6))
            fps = sum(fps_hist) / len(fps_hist)

            lineas: list[tuple[str, tuple[int, int, int], float]] = [
                (f"VISION   personas: {len(personas)}    {fps:4.1f} FPS", VERDE, 0.6),
            ]

            if hilo_audio is None:
                lineas.append(("AUDIO    desactivado (--sin-audio)", GRIS, 0.5))
            else:
                est = hilo_audio.estado()
                if est["error"]:
                    lineas.append((f"AUDIO    error: {est['error'][:60]}", (0, 0, 255), 0.5))
                elif not est["listo"]:
                    lineas.append(("AUDIO    cargando YAMNet...", GRIS, 0.5))
                else:
                    lineas.append(
                        (f"AUDIO    {est['db']:5.1f} dB    dominante: {est['dominante']}", AMBAR, 0.6)
                    )
                    for nombre, puntaje in est["top"]:
                        lineas.append((f"           {puntaje:.2f}  {nombre}", BLANCO, 0.5))

            dibujar_panel(frame, lineas)
            cv2.imshow(ventana, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if hilo_audio:
            hilo_audio.detener()
            hilo_audio.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
