"""Grabación de la sesión: video de la cámara y audio del micrófono.

Ambos van a la carpeta de la sesión (sesiones/<fecha>_sesion_<n>/), junto al
reporte y las fotos. Se activan en grabacion.video / grabacion.audio
(config.yaml). Solo con cámara en vivo: al analizar un archivo, el video ya
existe.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


class GrabadorVideo:
    """Escribe los cuadros en un MP4 de FPS fijos con la hora impresa."""

    def __init__(self, ruta: Path, fps: float) -> None:
        self.ruta = Path(ruta)
        self.fps = float(fps)
        # cv2.VideoWriter no acepta rutas con acentos ("Semáforo") en Windows:
        # se graba en Temp y se mueve a la carpeta de la sesión al cerrar.
        self._temporal = self.ruta if str(self.ruta).isascii() else (
            Path(tempfile.gettempdir()) / f"semaforo_{self.ruta.parent.name}_{self.ruta.name}"
        )
        self._w = None
        self._t0 = 0.0
        self._n = 0
        self.fallo = False

    def escribir(self, frame: np.ndarray, t: float | None = None) -> None:
        import cv2

        if self.fallo:
            return
        t = time.time() if t is None else t
        if self._w is None:
            alto, ancho = frame.shape[:2]
            self._w = cv2.VideoWriter(str(self._temporal), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (ancho, alto))
            if not self._w.isOpened():
                print(f"[grabacion] no se pudo crear el video {self._temporal}", file=sys.stderr)
                self.fallo = True
                return
            self._t0 = t
        img = frame.copy()
        cv2.putText(img, f"{datetime.now():%Y-%m-%d %H:%M:%S}", (img.shape[1] - 200, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # La cámara no entrega FPS constantes: se repite el cuadro las veces que
        # haga falta para que el video dure lo mismo que la sesión.
        debidos = int((t - self._t0) * self.fps) + 1
        for _ in range(min(debidos - self._n, int(self.fps))):
            self._w.write(img)
        self._n = max(self._n, debidos)

    def cerrar(self) -> Path | None:
        if not self._w or self.fallo:
            return None
        self._w.release()
        if self._temporal != self.ruta:
            shutil.move(str(self._temporal), str(self.ruta))
        return self.ruta


class GrabadorAudio:
    """WAV mono de 16 bits con los mismos bloques que recibe el clasificador."""

    def __init__(self, ruta: Path, sample_rate: int) -> None:
        self.ruta = Path(ruta)
        self._lock = threading.Lock()
        self._wav = wave.open(str(self.ruta), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(int(sample_rate))

    def escribir(self, bloque: np.ndarray) -> None:
        pcm = (np.clip(bloque, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        with self._lock:
            if self._wav:
                self._wav.writeframes(pcm)

    def cerrar(self) -> Path:
        with self._lock:
            if self._wav:
                self._wav.close()
                self._wav = None
        return self.ruta
