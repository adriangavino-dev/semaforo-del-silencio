"""YOLOv8-Pose como detector de personas y keypoints.

El modelo es preentrenado y no se reentrena (restricción 1). Devuelve, por cada
persona en el frame, su recuadro y sus 17 keypoints COCO. Sobre esos keypoints
se construirán las características de la sección 6.3 en la etapa 4.

Ejecutable de forma independiente (restricción 6):
    python -m src.vision.pose --camara 0
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

# Índices de los 17 keypoints del formato COCO que usa YOLOv8-Pose.
KP = {
    "nariz": 0,
    "ojo_izq": 1, "ojo_der": 2, "oreja_izq": 3, "oreja_der": 4,
    "hombro_izq": 5, "hombro_der": 6,
    "codo_izq": 7, "codo_der": 8,
    "muñeca_izq": 9, "muñeca_der": 10,
    "cadera_izq": 11, "cadera_der": 12,
    "rodilla_izq": 13, "rodilla_der": 14,
    "tobillo_izq": 15, "tobillo_der": 16,
}

# Esqueleto para el dibujo: pares de keypoints que se unen con una línea.
ESQUELETO = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # brazos y hombros
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # piernas
    (0, 1), (0, 2), (1, 3), (2, 4),                # cabeza
]

COLOR_ESQUELETO = (0, 220, 255)
COLOR_KEYPOINT = (0, 90, 255)
COLOR_RECUADRO = (0, 255, 120)


@dataclass
class Persona:
    """Una persona detectada en un frame."""

    id: int
    recuadro: tuple[float, float, float, float]     # x1, y1, x2, y2 en píxeles
    keypoints: np.ndarray = field(repr=False)        # (17, 2) en píxeles
    confianzas: np.ndarray = field(repr=False)       # (17,) confianza por keypoint
    conf_deteccion: float = 0.0
    seguido: bool = False          # True si el id proviene del seguimiento

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.recuadro
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def alto_recuadro(self) -> float:
        return self.recuadro[3] - self.recuadro[1]

    @property
    def ancho_recuadro(self) -> float:
        return self.recuadro[2] - self.recuadro[0]

    def punto(self, nombre: str) -> tuple[np.ndarray, float]:
        """Coordenada y confianza de un keypoint por nombre."""
        i = KP[nombre]
        return self.keypoints[i], float(self.confianzas[i])

    def punto_medio(self, izq: str, der: str) -> tuple[np.ndarray, float]:
        """Punto medio entre un par simétrico; la confianza es la menor de las dos."""
        p_izq, c_izq = self.punto(izq)
        p_der, c_der = self.punto(der)
        return (p_izq + p_der) / 2.0, min(c_izq, c_der)

    def zona(self, ancho_frame: int, alto_frame: int, zonas: dict) -> str:
        """Zona configurada ('docente' o 'alumnos') en la que cae el centro."""
        cx, cy = self.centro
        cx_n, cy_n = cx / ancho_frame, cy / alto_frame
        for nombre, (x1, y1, x2, y2) in zonas.items():
            if x1 <= cx_n <= x2 and y1 <= cy_n <= y2:
                return nombre
        return "alumnos"


class DetectorPose:
    """Envoltura sobre Ultralytics YOLOv8-Pose."""

    def __init__(self, configuracion: dict | None = None, seguimiento: bool = True) -> None:
        conf = (configuracion or cfg.cargar())["vision"]
        self.ruta_modelo: str = conf["modelo_pose"]
        self.conf_minima: float = float(conf["conf_minima"])
        self.zonas: dict = conf["zonas"]
        self.seguimiento = seguimiento
        self._modelo = None

    def cargar(self) -> "DetectorPose":
        """Carga el modelo; Ultralytics descarga los pesos la primera vez."""
        from ultralytics import YOLO

        ruta = Path(self.ruta_modelo)
        if not ruta.is_absolute():
            local = cfg.RAIZ / "models" / ruta.name
            ruta = local if local.exists() else ruta
        self._modelo = YOLO(str(ruta))
        return self

    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    def detectar(self, frame: np.ndarray) -> list[Persona]:
        """Devuelve la lista de personas detectadas en el frame.

        Con seguimiento activo, el id de cada persona se mantiene entre cuadros.
        Eso es lo que permite medir la velocidad de las muñecas: si el id fuera
        la posición en la lista de detecciones cambiaría de un cuadro a otro y se
        acabaría comparando las muñecas de una persona con las de otra, lo que
        produce velocidades enormes y falsos eventos de movimiento.
        """
        if self._modelo is None:
            raise RuntimeError("El modelo no está cargado. Llama a cargar() primero.")

        if self.seguimiento:
            resultados = self._modelo.track(
                frame, conf=self.conf_minima, verbose=False, classes=[0], persist=True
            )
        else:
            resultados = self._modelo.predict(
                frame, conf=self.conf_minima, verbose=False, classes=[0]  # clase 0 = persona
            )
        if not resultados:
            return []

        r = resultados[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []

        cajas = r.boxes.xyxy.cpu().numpy()
        conf_cajas = r.boxes.conf.cpu().numpy()
        # El seguimiento puede no asignar id en los primeros cuadros de una
        # persona; en ese caso se cae al índice, que sirve para dibujar.
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
        kps = r.keypoints.xy.cpu().numpy()                      # (n, 17, 2)
        kps_conf = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints.conf is not None
            else np.ones(kps.shape[:2], dtype=np.float32)
        )

        personas = []
        for i in range(len(cajas)):
            personas.append(
                Persona(
                    id=int(ids[i]) if ids is not None else i,
                    seguido=ids is not None,
                    recuadro=tuple(float(v) for v in cajas[i]),
                    keypoints=kps[i],
                    confianzas=kps_conf[i],
                    conf_deteccion=float(conf_cajas[i]),
                )
            )
        return personas

    def reiniciar_seguimiento(self) -> None:
        """Olvida las identidades acumuladas (útil entre sesiones)."""
        if self._modelo is not None and hasattr(self._modelo, "predictor"):
            trackers = getattr(self._modelo.predictor, "trackers", None)
            for t in trackers or []:
                t.reset()


def dibujar(frame: np.ndarray, personas: list[Persona], umbral_kp: float = 0.5) -> np.ndarray:
    """Dibuja recuadros y esqueletos sobre el frame (in-place)."""
    import cv2

    for p in personas:
        x1, y1, x2, y2 = (int(v) for v in p.recuadro)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RECUADRO, 1)
        cv2.putText(
            frame, f"#{p.id} {p.conf_deteccion:.2f}", (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RECUADRO, 1, cv2.LINE_AA,
        )

        for a, b in ESQUELETO:
            if p.confianzas[a] > umbral_kp and p.confianzas[b] > umbral_kp:
                pa = tuple(int(v) for v in p.keypoints[a])
                pb = tuple(int(v) for v in p.keypoints[b])
                cv2.line(frame, pa, pb, COLOR_ESQUELETO, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(p.keypoints):
            if p.confianzas[i] > umbral_kp:
                cv2.circle(frame, (int(x), int(y)), 3, COLOR_KEYPOINT, -1, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    ap = argparse.ArgumentParser(description="Prueba de YOLOv8-Pose sobre la webcam")
    ap.add_argument("--camara", type=int, default=None)
    args = ap.parse_args()

    conf = cfg.cargar()["vision"]
    indice = args.camara if args.camara is not None else conf["camara_index"]

    print("Cargando YOLOv8-Pose...")
    detector = DetectorPose().cargar()
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {indice}")

    print("Pulsa 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        personas = detector.detectar(frame)
        dibujar(frame, personas)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            frame, f"personas: {len(personas)}  {fps:.1f} FPS", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow("YOLOv8-Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
