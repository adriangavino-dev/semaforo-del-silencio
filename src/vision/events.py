"""Clasificador de eventos sobre ventanas de tiempo — sección 6.4.

La unidad de decisión del evento NO es un cuadro sino una secuencia de ~2 s. Por
cada ventana se calculan la media, el máximo y la desviación estándar de cada
característica por persona, y sobre ese vector decide un RandomForest (o, si aún
no está entrenado, una regla provisional). Un evento se registra solo cuando la
ventana sale anómala varias veces seguidas.

Por qué importa: decidir cuadro a cuadro convierte cualquier salto de un
keypoint —un parpadeo del detector, un cambio de identidad— en una alerta. Una
ventana de 2 s con tres confirmaciones exige que el movimiento se sostenga
durante ~6 s, que es lo que distingue un empujón de un brazo que se estira.

Ejecutable de forma independiente:
    python -m src.vision.events --video data/video/raw/pelea.mp4
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.features import NOMBRES  # noqa: E402

ESTADISTICOS = ("media", "max", "std")
# Cuántas personas hay en la ventana también aporta: una aglomeración inusual
# se ve como muchas personas con poca distancia entre ellas.
EXTRA = ("n_personas",)


def nombres_vector() -> list[str]:
    """Nombres de las columnas del vector de una ventana, en orden."""
    columnas = [f"{r}_{e}" for r in NOMBRES for e in ESTADISTICOS]
    columnas += [f"{x}_{e}" for x in EXTRA for e in ("media", "max")]
    return columnas


def vector_de_ventana(cuadros: list[list[dict[str, float]]]) -> np.ndarray:
    """Resume una ventana de cuadros (cada uno con los rasgos de sus personas).

    Devuelve media/max/std de cada rasgo sobre todas las personas y cuadros,
    más media y máximo del número de personas. Si no hubo nadie, todo cero.
    """
    muestras = [r for cuadro in cuadros for r in cuadro]
    n_personas = np.array([len(c) for c in cuadros], dtype=float)

    partes: list[float] = []
    if muestras:
        matriz = np.array([[m[r] for r in NOMBRES] for m in muestras], dtype=float)
        # distancia_minima usa 999 como "no hay nadie cerca"; se acota para que
        # no domine la desviación estándar.
        col = NOMBRES.index("distancia_minima")
        matriz[:, col] = np.minimum(matriz[:, col], 20.0)
        for j in range(len(NOMBRES)):
            c = matriz[:, j]
            partes += [float(c.mean()), float(c.max()), float(c.std())]
    else:
        partes += [0.0] * (len(NOMBRES) * len(ESTADISTICOS))

    partes += [float(n_personas.mean()), float(n_personas.max())] if len(n_personas) else [0.0, 0.0]
    return np.array(partes, dtype=float)


class DetectorEventos:
    """Acumula cuadros, decide por ventana y confirma por repetición."""

    def __init__(self, configuracion: dict | None = None, fps: float = 15.0) -> None:
        self.conf = configuracion or cfg.cargar()
        cv = self.conf["vision"]
        ce = cv["evento"]
        self.ventana_s = float(ce["ventana_s"])
        self.paso_s = float(ce["paso_ventana_s"])
        self.confirmaciones = int(cv["confirmaciones_evento"])
        self.fps = 0.0
        self.tam_ventana = 5
        self.paso = 1
        self.ajustar_fps(fps)

        # Regla provisional, solo si no hay modelo entrenado.
        self.umbral_media = float(ce["regla"]["velocidad_media_por_s"])
        self.umbral_fraccion = float(ce["regla"]["fraccion_cuadros_rapidos"])
        self.umbral_pico = float(ce["umbral_velocidad_muneca_por_s"])

        self._desde_ultima = 0
        self._racha = 0
        self._modelo = None
        self.clases: list[str] = list(cv["clases_evento"])
        self.entrenado_el: str | None = None
        self.ultima_prediccion: dict = {"evento": "normal", "puntaje": 0.0}

    def ajustar_fps(self, fps: float) -> None:
        """Recalcula el tamaño de la ventana cuando cambian los FPS reales.

        Con la webcam los FPS no se conocen hasta medirlos; sin esto la ventana
        de "2 segundos" tendría una duración distinta según la cámara.
        """
        fps = max(float(fps), 1.0)
        if self.fps and abs(fps - self.fps) / self.fps < 0.2:
            return
        self.fps = fps
        self.tam_ventana = max(int(round(self.ventana_s * fps)), 5)
        self.paso = max(int(round(self.paso_s * fps)), 1)
        cuadros = list(getattr(self, "_cuadros", []))
        self._cuadros = deque(cuadros[-self.tam_ventana:], maxlen=self.tam_ventana)

    # ------------------------------------------------------------------
    def cargar_modelo(self) -> "DetectorEventos":
        """Carga el clasificador entrenado si existe; si no, queda la regla."""
        import joblib

        ruta = cfg.ruta_de("modelo_evento")
        if ruta.exists():
            paquete = joblib.load(ruta)
            self._modelo = paquete["modelo"]
            self.clases = list(paquete["clases"])
            self.entrenado_el = paquete.get("entrenado")
        return self

    @property
    def con_modelo(self) -> bool:
        return self._modelo is not None

    # ------------------------------------------------------------------
    def _predecir(self, vector: np.ndarray, cuadros: list[list[dict[str, float]]]) -> tuple[str, float]:
        if self._modelo is not None:
            probas = self._modelo.predict_proba(vector.reshape(1, -1))[0]
            clases = list(self._modelo.classes_)
            i = int(np.argmax(probas))
            return str(clases[i]), float(probas[i])

        # Regla provisional: la velocidad media de la ventana debe ser alta Y
        # una fracción suficiente de cuadros debe superar el pico. Ninguna de
        # las dos sola basta: un pico aislado no cumple la media, y una media
        # alta por un salto de keypoint no cumple la fracción.
        vel = [r["velocidad_munecas"] for c in cuadros for r in c]
        if not vel:
            self.ultimo_detalle = {"media": 0.0, "fraccion": 0.0}
            return "normal", 0.0
        v = np.array(vel)
        media = float(v.mean())
        fraccion = float((v > self.umbral_pico).mean())
        self.ultimo_detalle = {"media": media, "fraccion": fraccion}
        anomalo = media > self.umbral_media and fraccion > self.umbral_fraccion
        puntaje = min(1.0, 0.5 * media / self.umbral_media + 0.5 * fraccion / self.umbral_fraccion)
        return ("movimiento_anomalo" if anomalo else "normal"), puntaje

    def agregar_cuadro(self, rasgos_personas: list[dict[str, float]]) -> dict:
        """Añade un cuadro. Devuelve el estado del detector tras ese cuadro.

        `evento` es la última decisión de ventana; `confirmado` se vuelve True
        solo cuando `confirmaciones_evento` ventanas seguidas salieron anómalas,
        y `nuevo` marca el cuadro exacto en el que se alcanzó la confirmación,
        para registrar el evento una sola vez.
        """
        self._cuadros.append(rasgos_personas)
        self._desde_ultima += 1
        nuevo = False

        if len(self._cuadros) == self.tam_ventana and self._desde_ultima >= self.paso:
            self._desde_ultima = 0
            cuadros = list(self._cuadros)
            vector = vector_de_ventana(cuadros)
            evento, puntaje = self._predecir(vector, cuadros)
            self.ultima_prediccion = {"evento": evento, "puntaje": puntaje}

            racha_previa = self._racha
            self._racha = self._racha + 1 if evento == "movimiento_anomalo" else 0
            nuevo = self._racha == self.confirmaciones and racha_previa == self.confirmaciones - 1

        detalle = getattr(self, "ultimo_detalle", {"media": 0.0, "fraccion": 0.0})
        return {
            "evento": self.ultima_prediccion["evento"],
            "puntaje": self.ultima_prediccion["puntaje"],
            "confirmado": self._racha >= self.confirmaciones,
            "nuevo": nuevo,
            "racha": self._racha,
            "confirmaciones": self.confirmaciones,
            # Cifras de la regla en la última ventana (para mostrarlas en pantalla).
            "velocidad_media": detalle["media"],
            "fraccion_rapidos": detalle["fraccion"],
            "umbral_media": self.umbral_media,
            "umbral_fraccion": self.umbral_fraccion,
        }

    def reiniciar(self) -> None:
        self._cuadros.clear()
        self._desde_ultima = 0
        self._racha = 0


if __name__ == "__main__":
    import argparse

    import cv2

    from src.vision.features import LineaBase, caracteristicas, muñecas_previas
    from src.vision.pose import DetectorPose

    ap = argparse.ArgumentParser(description="Evalúa el detector de eventos sobre un video")
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    conf = cfg.cargar()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = DetectorPose(conf).cargar()
    eventos = DetectorEventos(conf, fps).cargar_modelo()
    base = LineaBase.cargar()
    print(f"{Path(args.video).name}: {fps:.1f} FPS, ventana {eventos.tam_ventana} cuadros, "
          f"{'modelo entrenado' if eventos.con_modelo else 'regla provisional'}")

    previas: dict[int, np.ndarray] = {}
    n, ventanas, anomalas, confirmados = 0, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        personas = det.detectar(frame)
        rasgos = [caracteristicas(p, personas, base, previas, fps) for p in personas]
        previas = muñecas_previas(personas)
        r = eventos.agregar_cuadro(rasgos)
        if eventos._desde_ultima == 0 and len(eventos._cuadros) == eventos.tam_ventana:
            ventanas += 1
            anomalas += r["evento"] == "movimiento_anomalo"
        if r["nuevo"]:
            confirmados += 1
            print(f"  evento confirmado en el segundo {n / fps:.1f}")
    cap.release()
    print(f"{n} cuadros, {ventanas} ventanas, {anomalas} anómalas, {confirmados} eventos confirmados")
