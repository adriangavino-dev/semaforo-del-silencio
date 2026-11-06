"""Captura de audio en tiempo real desde el micrófono.

Entrega bloques de 1 segundo a 16 kHz mono, que es el formato que exige YAMNet.
La captura corre en el hilo de callback de sounddevice y deposita los bloques en
una cola, de modo que el bucle principal de la demo nunca se bloquea esperando
audio (criterio de rendimiento de la sección 10).

Ejecutable de forma independiente (restricción 6):
    python -m src.audio.capture --listar
    python -m src.audio.capture --segundos 10
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402


def nivel_db(bloque: np.ndarray, referencia: float = 1e-5) -> float:
    """Nivel RMS del bloque en dB relativos.

    No es una medida calibrada de sonómetro: el ESP32 con el INMP441 es el
    responsable de la medición en dB. Aquí sirve como referencia relativa para
    la regla de decisión de la sección 7.
    """
    rms = float(np.sqrt(np.mean(np.square(bloque, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, referencia) / referencia)


class CapturaAudio:
    """Flujo de micrófono que produce bloques de duración fija."""

    def __init__(self, configuracion: dict | None = None) -> None:
        conf = (configuracion or cfg.cargar())["audio"]
        self.sample_rate: int = int(conf["sample_rate"])
        self.canales: int = int(conf["canales"])
        self.dispositivo = conf.get("dispositivo")
        self.duracion_bloque: float = float(conf["duracion_bloque_s"])
        self.referencia_db: float = float(conf["db"]["referencia_rms"])

        self.muestras_por_bloque = int(self.sample_rate * self.duracion_bloque)
        self._cola: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._stream = None
        self.bloques_descartados = 0
        # Vigilante: en Windows (MME) el flujo de entrada puede quedarse mudo
        # tras un "input overflow", p. ej. cuando otro programa empieza a
        # reproducir sonido. Si pasa max_silencio_s sin bloques, se reinicia.
        self.max_silencio_s = 3.0
        self._ultimo_bloque_t = 0.0
        self.reinicios = 0
        # Tras reabrir el flujo, los micrófonos USB con control automático de
        # ganancia entregan un par de bloques con nivel falso; se descartan.
        self._descartar = 0

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] aviso de sounddevice: {status}", file=sys.stderr)
        # copia obligatoria: sounddevice reutiliza el búfer entre llamadas
        bloque = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._ultimo_bloque_t = time.time()
        if self._descartar > 0:
            self._descartar -= 1
            return
        try:
            self._cola.put_nowait(bloque)
        except queue.Full:
            # Se prefiere descartar el bloque más viejo antes que frenar la captura.
            self.bloques_descartados += 1
            try:
                self._cola.get_nowait()
                self._cola.put_nowait(bloque)
            except queue.Empty:
                pass

    # ------------------------------------------------------------------
    def iniciar(self) -> "CapturaAudio":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.canales,
            device=self.dispositivo,
            dtype="float32",
            blocksize=self.muestras_por_bloque,
            callback=self._callback,
        )
        self._stream.start()
        self._ultimo_bloque_t = time.time()
        self._descartar = max(self._descartar, 2)   # también al arrancar el micrófono "calienta"
        return self

    def reiniciar(self) -> None:
        """Cierra y vuelve a abrir el flujo del micrófono."""
        self.reinicios += 1
        print("[audio] el micrófono dejó de entregar audio; se reinicia la captura", file=sys.stderr)
        try:
            self.detener()
        except Exception as exc:  # noqa: BLE001
            print(f"[audio] al cerrar el flujo: {exc}", file=sys.stderr)
        self._descartar = 2
        self.iniciar()

    def detener(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def leer(self, timeout: float | None = None) -> np.ndarray | None:
        """Devuelve el siguiente bloque, o None si no hay ninguno disponible."""
        try:
            return self._cola.get(timeout=timeout) if timeout else self._cola.get_nowait()
        except queue.Empty:
            if self._stream is not None and time.time() - self._ultimo_bloque_t > self.max_silencio_s:
                self.reiniciar()
            return None

    def db(self, bloque: np.ndarray) -> float:
        return nivel_db(bloque, self.referencia_db)

    def __enter__(self) -> "CapturaAudio":
        return self.iniciar()

    def __exit__(self, *exc) -> None:
        self.detener()


def listar_dispositivos() -> None:
    import sounddevice as sd

    print(sd.query_devices())


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Prueba de captura de audio")
    ap.add_argument("--listar", action="store_true", help="listar dispositivos y salir")
    ap.add_argument("--segundos", type=float, default=5.0, help="duración de la prueba")
    args = ap.parse_args()

    if args.listar:
        listar_dispositivos()
        raise SystemExit(0)

    captura = CapturaAudio()
    print(f"Capturando {args.segundos:.0f} s a {captura.sample_rate} Hz mono...")
    with captura:
        fin = time.time() + args.segundos
        while time.time() < fin:
            bloque = captura.leer(timeout=0.5)
            if bloque is not None:
                barra = "#" * int(max(0.0, captura.db(bloque)) / 2)
                print(f"  {captura.db(bloque):6.1f} dB  {barra}")
    print(f"Bloques descartados: {captura.bloques_descartados}")
