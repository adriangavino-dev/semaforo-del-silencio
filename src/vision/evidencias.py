"""Foto al confirmar un evento o al cambiar el semáforo de color.

Pedido del profesor. Las fotos van a la carpeta de la sesión
(sesiones/<fecha>_sesion_<n>/fotos/) y el tablero las muestra junto a cada
evento y cada cambio de estado.

Configuración en grabacion.fotos (config.yaml):
  - al_evento / al_cambio_estado: qué dispara la foto.
  - difuminar_caras: pixela la zona de la cabeza de cada persona, ubicada con
    los puntos que ya da YOLOv8-Pose (nariz, ojos, orejas); si no se ve la
    cabeza, el tercio superior del recuadro. Apagado desde que la grabación
    está autorizada; se deja por si algún salón no lo autoriza.
  - max_por_sesion: tope de fotos por sesión.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

COLOR = {"VERDE": (60, 180, 60), "AMARILLO": (0, 200, 255), "ROJO": (40, 40, 230)}


def _pixelar(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, bloques: int = 6) -> None:
    import cv2

    alto, ancho = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ancho, x2), min(alto, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    zona = img[y1:y2, x1:x2]
    chica = cv2.resize(zona, (bloques, bloques), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(chica, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def difuminar_caras(img: np.ndarray, personas: list) -> None:
    for p in personas:
        x1, y1, x2, y2 = (int(v) for v in p.recuadro)
        cabeza = [p.keypoints[i] for i in range(5) if p.confianzas[i] > 0.3]
        if cabeza:
            c = np.mean(cabeza, axis=0)
            lado = max(0.22 * (y2 - y1), 0.6 * (x2 - x1), 16)
            _pixelar(img, int(c[0] - lado / 2), int(c[1] - lado * 0.6), int(c[0] + lado / 2), int(c[1] + lado * 0.5))
        else:
            _pixelar(img, x1, y1, x2, y1 + (y2 - y1) // 3)


class Evidencias:
    def __init__(self, conf: dict, carpeta_sesion: Path) -> None:
        cf = conf.get("grabacion", {}).get("fotos", {})
        self.habilitado = bool(cf.get("habilitado", True))
        self.al_evento = bool(cf.get("al_evento", True))
        self.al_cambio = bool(cf.get("al_cambio_estado", True))
        self.difuminar = bool(cf.get("difuminar_caras", False))
        self.maximo = int(cf.get("max_por_sesion", 500))
        self.carpeta = Path(carpeta_sesion) / "fotos"
        self.guardadas = 0

    def guardar(self, frame: np.ndarray, personas: list, estado: str, texto: str, tipo: str) -> str | None:
        """Guarda la foto y devuelve su ruta relativa para el tablero, o None si no toca."""
        import cv2

        if not self.habilitado or self.guardadas >= self.maximo or frame is None:
            return None
        if (tipo == "evento" and not self.al_evento) or (tipo == "cambio" and not self.al_cambio):
            return None
        img = frame.copy()
        if self.difuminar:
            difuminar_caras(img, personas)
        for p in personas:
            x1, y1, x2, y2 = (int(v) for v in p.recuadro)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 120), 1)
        ahora = datetime.now()
        # Dos líneas (estado y hora / motivo), achicadas si no caben: con video
        # vertical (celular) una sola línea se cortaba.
        lineas = [f"{estado}  {ahora:%Y-%m-%d %H:%M:%S}", texto]
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        ancho_util = img.shape[1] - 22
        escala = min(0.55, *(0.55 * ancho_util / max(cv2.getTextSize(l, fuente, 0.55, 1)[0][0], 1) for l in lineas))
        paso = int(26 * escala / 0.55) + 4
        alto_franja = 8 + paso * len(lineas)
        cv2.rectangle(img, (0, 0), (img.shape[1], alto_franja), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 0), (8, alto_franja), COLOR.get(estado, (200, 200, 200)), -1)
        for i, linea in enumerate(lineas):
            cv2.putText(img, linea, (16, 4 + paso * (i + 1) - 6), fuente, escala, (255, 255, 255), 1, cv2.LINE_AA)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        nombre = f"{ahora:%H%M%S}_{tipo}_{self.guardadas:03d}.jpg"
        ok, datos = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        # cv2.imwrite no acepta rutas con acentos ("Semáforo") en Windows.
        (self.carpeta / nombre).write_bytes(datos.tobytes())
        self.guardadas += 1
        return f"sesiones/{self.carpeta.parent.name}/fotos/{nombre}"
