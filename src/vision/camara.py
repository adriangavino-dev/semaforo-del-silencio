"""Cámara en vivo sin retraso acumulado.

OpenCV entrega los cuadros en orden. Si la cámara manda más por segundo de los
que el análisis procesa (pasa con cámaras por Wi-Fi como iVCam o DroidCam, a 30
FPS, frente a un análisis de 15-25 FPS), los cuadros se encolan y el video se
va atrasando cada vez más. Aquí un hilo lee sin parar y se queda solo con el
último cuadro: el análisis siempre ve lo que pasa ahora.

Misma interfaz que cv2.VideoCapture para lo que usa run_session.py:
read(), isOpened(), get(), set(), release().
"""
from __future__ import annotations

import threading
import time

import cv2


class CamaraEnVivo:
    def __init__(self, indice: int, ancho: int | None = None, alto: int | None = None) -> None:
        self._cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if ancho:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(ancho))
        if alto:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(alto))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._frame = None
        self._n = 0            # número del último cuadro leído
        self._entregado = 0    # número del último cuadro devuelto por read()
        self._vivo = self._cap.isOpened()
        self.descartados = 0
        if self._vivo:
            threading.Thread(target=self._leer, daemon=True, name="camara").start()

    def _leer(self) -> None:
        while self._vivo:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                if self._n > self._entregado:
                    self.descartados += 1      # el análisis no alcanzó a usar el anterior
                self._frame = frame
                self._n += 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0):
        """El cuadro más reciente que aún no se ha devuelto (espera si hace falta)."""
        fin = time.time() + timeout
        with self._cond:
            while self._vivo and self._n == self._entregado:
                restante = fin - time.time()
                if restante <= 0:
                    return False, None
                self._cond.wait(restante)
            if self._frame is None:
                return False, None
            self._entregado = self._n
            return True, self._frame.copy()

    def isOpened(self) -> bool:
        return self._vivo

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, valor):
        return self._cap.set(prop, valor)

    def release(self) -> None:
        self._vivo = False
        time.sleep(0.05)
        self._cap.release()
