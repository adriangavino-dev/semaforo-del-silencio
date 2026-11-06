"""Captura de audio desde el INMP441 del ESP32, por el cable USB.

Misma interfaz que CapturaAudio (iniciar, leer, detener), para que el hilo de
audio de run_session.py no distinga de dónde sale el sonido: el micrófono de la
laptop o el del aula. Entrega bloques de 1 s a 16 kHz en float32, que es lo que
espera YAMNet.

El ESP32 manda bloques de 32 ms en base64 a 921600 baudios (ver el firmware y
src/comm/esp32_serial.py). Si se pierde algún bloque el audio sigue; se cuenta
en las estadísticas del cliente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402


class CapturaESP32:
    def __init__(self, cliente, configuracion: dict | None = None) -> None:
        conf = (configuracion or cfg.cargar())["audio"]
        self.cliente = cliente
        self.sample_rate = int(conf["sample_rate"])
        self.muestras_por_bloque = int(self.sample_rate * float(conf["duracion_bloque_s"]))
        self.activa = False
        self.bloques_descartados = 0

    def iniciar(self, intentos: int = 6) -> "CapturaESP32":
        # Al abrir el puerto el ESP32 se reinicia (~3 s): se reintenta hasta que conteste.
        import time

        for _ in range(intentos):
            if self.cliente.iniciar_audio():
                self.activa = True
                return self
            time.sleep(1.0)
        raise RuntimeError("el ESP32 no confirmó el envío de audio (¿firmware sin la orden de audio?)")

    def leer(self, timeout: float | None = None) -> np.ndarray | None:
        bloque = self.cliente.leer_audio(self.muestras_por_bloque, timeout=timeout or 2.0)
        if bloque is None and self.activa:
            # Sin audio: puede que el ESP32 se haya reiniciado y olvidado la orden.
            self._sin_audio = getattr(self, "_sin_audio", 0) + 1
            if self._sin_audio >= 3:
                self._sin_audio = 0
                print("[audio] el ESP32 dejó de mandar audio; se vuelve a pedir", file=sys.stderr)
                self.cliente.iniciar_audio()
        else:
            self._sin_audio = 0
        return bloque

    def detener(self) -> None:
        if self.activa:
            self.activa = False
            try:
                self.cliente.detener_audio()
            except Exception:  # noqa: BLE001
                pass
