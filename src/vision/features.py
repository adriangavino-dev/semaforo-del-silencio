"""Características por persona — sección 6.3.

Todo se normaliza respecto del recuadro de la persona, de modo que alguien
sentado al fondo y alguien sentado adelante produzcan valores parecidos aunque
ocupen muy distinta cantidad de píxeles.

La característica principal para la postura es la desviación respecto de la
línea base. En un aula real las piernas quedan tapadas por las carpetas, así que
el ángulo de la rodilla casi nunca sirve; por eso se incluye acompañado de la
confianza de los keypoints de la pierna, para que quien use la característica
sepa cuándo ignorarla.

Ejecutable de forma independiente:
    python -m src.vision.features --camara 1
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.pose import Persona  # noqa: E402

NOMBRES = [
    "desviacion_linea_base",
    "relacion_torso_altura",
    "angulo_cadera_rodilla",
    "confianza_piernas",
    "velocidad_munecas",
    "distancia_minima",
]


class LineaBase:
    """Altura de hombro esperada si la persona estuviera sentada.

    No puede ser un valor único: por la perspectiva, la cabeza de alguien sentado
    al fondo aparece más arriba en el cuadro que la de alguien de pie adelante.
    Por eso se ajusta un plano sobre la posición de la persona en la imagen.

        y_hombro_esperado = a * x_centro + b * y_cadera + c
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0, ajustada: bool = False):
        self.a, self.b, self.c = a, b, c
        self.ajustada = ajustada

    @classmethod
    def cargar(cls, ruta: Path | None = None) -> "LineaBase":
        ruta = ruta or cfg.ruta_de("calibracion")
        if not ruta.exists():
            return cls()
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return cls(datos["a"], datos["b"], datos["c"], ajustada=True)

    def guardar(self, ruta: Path | None = None, extra: dict | None = None) -> Path:
        ruta = ruta or cfg.ruta_de("calibracion")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = {"a": self.a, "b": self.b, "c": self.c}
        datos.update(extra or {})
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        return ruta

    def esperado(self, x_centro: float, y_cadera: float) -> float:
        return self.a * x_centro + self.b * y_cadera + self.c


def angulo(p1: np.ndarray, vertice: np.ndarray, p2: np.ndarray) -> float:
    """Ángulo en grados formado en `vertice` por los segmentos hacia p1 y p2."""
    v1, v2 = p1 - vertice, p2 - vertice
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    coseno = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(coseno))


def caracteristicas(
    persona: Persona,
    personas: list[Persona],
    linea_base: LineaBase,
    previas: dict[int, np.ndarray] | None = None,
    fps: float = 30.0,
    max_velocidad_fisica: float = 12.0,
) -> dict[str, float]:
    """Vector de características de una persona en un frame.

    `fps` es necesario para que la velocidad de las muñecas quede expresada por
    segundo y no por cuadro. Sin eso la medida depende de la cámara: un mismo
    movimiento grabado a 60 FPS da la mitad de desplazamiento por cuadro que a
    30 FPS, y un umbral fijo dejaría de servir al cambiar de fuente.

    `max_velocidad_fisica` (alturas de recuadro por segundo) separa el
    movimiento humano de los saltos del detector.
    """
    alto = max(persona.alto_recuadro, 1.0)
    ancho = max(persona.ancho_recuadro, 1.0)
    cx, _ = persona.centro

    hombro, conf_hombro = persona.punto_medio("hombro_izq", "hombro_der")
    cadera, conf_cadera = persona.punto_medio("cadera_izq", "cadera_der")
    rodilla, conf_rodilla = persona.punto_medio("rodilla_izq", "rodilla_der")
    tobillo, conf_tobillo = persona.punto_medio("tobillo_izq", "tobillo_der")

    # Desviación respecto de la línea base, normalizada por el alto del recuadro.
    # Positiva y grande = la persona está más alta de lo esperado si estuviera sentada.
    if linea_base.ajustada and conf_hombro > 0.2 and conf_cadera > 0.2:
        esperado = linea_base.esperado(cx, float(cadera[1]))
        desviacion = (esperado - float(hombro[1])) / alto
    else:
        desviacion = 0.0

    torso = float(np.linalg.norm(hombro - cadera)) / alto if conf_cadera > 0.2 else 0.0
    ang = angulo(cadera, rodilla, tobillo) if min(conf_rodilla, conf_tobillo) > 0.2 else 0.0
    conf_piernas = float((conf_rodilla + conf_tobillo) / 2.0)

    # Velocidad de muñecas entre frames consecutivos, normalizada por el alto.
    # Solo tiene sentido si la persona conserva su identidad entre cuadros: con
    # ids inestables se compararían las muñecas de dos personas distintas y la
    # velocidad saldría disparada.
    velocidad = 0.0
    if persona.seguido and previas is not None and persona.id in previas:
        antes = previas[persona.id]
        munecas = np.array([persona.keypoints[9], persona.keypoints[10]])
        conf_munecas = min(float(persona.confianzas[9]), float(persona.confianzas[10]))
        if conf_munecas > 0.4:
            # Se toma la menor de las dos muñecas, no la media: si el detector
            # hace saltar una sola muñeca, la otra revela que no hubo movimiento
            # real. Un movimiento brusco genuino mueve las dos.
            desplazamiento = float(np.linalg.norm(munecas - antes, axis=1).min() / alto)
            velocidad = desplazamiento * max(fps, 1.0)   # alturas de recuadro por segundo
            # Por encima de cierta velocidad ya no es un cuerpo humano sino un
            # salto del detector (cambio de identidad, keypoint reubicado). Se
            # descarta en lugar de dejar que dispare una alerta.
            if velocidad > max_velocidad_fisica:
                velocidad = 0.0

    # Distancia a la persona más cercana, normalizada por el ancho del recuadro.
    distancia = 999.0
    for otra in personas:
        if otra.id == persona.id:
            continue
        d = math.dist(persona.centro, otra.centro) / ancho
        distancia = min(distancia, d)

    return {
        "desviacion_linea_base": desviacion,
        "relacion_torso_altura": torso,
        "angulo_cadera_rodilla": ang,
        "confianza_piernas": conf_piernas,
        "velocidad_munecas": velocidad,
        "distancia_minima": distancia,
    }


def postura_por_umbral(rasgos: dict[str, float], umbral: float) -> str:
    """Postura estimada solo con la línea base.

    Es una regla provisional, no el clasificador entrenado que pide la Etapa 4.
    Sirve para que el conteo de personas de pie exista antes de tener ese modelo,
    y debe reemplazarse cuando el clasificador esté entrenado.
    """
    return "de_pie" if rasgos["desviacion_linea_base"] > umbral else "sentado"


def postura_geometrica(
    persona: Persona, rasgos: dict[str, float], aspecto_de_pie: float, angulo_de_pie: float,
    conf_piernas_min: float = 0.5,
) -> str:
    """Postura sin calibración, a partir de la geometría de la propia persona.

    Si las piernas son visibles, decide el ángulo de la rodilla (sentado ≈ 90°,
    de pie ≈ 150°). Si no —lo habitual en un aula, donde las carpetas tapan las
    piernas—, decide la proporción alto/ancho del recuadro: una persona de pie
    es una caja alta y estrecha (≈2.0), una sentada es casi cuadrada (≈1.0).

    Medido en los videos del proyecto: sentados p50 = 1.03, de pie p25 = 1.85.
    Es una regla, no el clasificador entrenado de la Etapa 4, pero a diferencia
    de la línea base no necesita calibrar la cámara antes.
    """
    if rasgos["confianza_piernas"] >= conf_piernas_min and rasgos["angulo_cadera_rodilla"] > 0:
        return "de_pie" if rasgos["angulo_cadera_rodilla"] >= angulo_de_pie else "sentado"
    aspecto = persona.alto_recuadro / max(persona.ancho_recuadro, 1.0)
    return "de_pie" if aspecto >= aspecto_de_pie else "sentado"


def filtrar_personas(
    personas: list[Persona], alto_frame: int, min_alto_fraccion: float, min_conf_media: float
) -> list[Persona]:
    """Descarta detecciones que no pueden ser una persona real en el aula.

    Dos filtros: recuadros demasiado pequeños respecto del cuadro (fotos en la
    pared, reflejos, gente muy lejos como para medirle nada) y detecciones cuyos
    keypoints tienen en promedio poca confianza (el detector "ve" una silueta
    pero no sabe dónde está el cuerpo). Cada uno se configura en config.yaml.
    """
    salida = []
    for p in personas:
        if p.alto_recuadro / max(alto_frame, 1) < min_alto_fraccion:
            continue
        if float(np.mean(p.confianzas)) < min_conf_media:
            continue
        salida.append(p)
    return salida


def munecas_de(persona: Persona) -> np.ndarray:
    """Coordenadas de las dos muñecas, para calcular la velocidad en el frame siguiente."""
    return np.array([persona.keypoints[9], persona.keypoints[10]])


def muñecas_previas(personas: list[Persona]) -> dict[int, np.ndarray]:
    """Mapa id -> muñecas, solo de las personas con identidad estable."""
    return {p.id: munecas_de(p) for p in personas if p.seguido}


if __name__ == "__main__":
    import argparse
    import cv2

    from src.vision.pose import DetectorPose, dibujar

    ap = argparse.ArgumentParser(description="Muestra las características en vivo")
    ap.add_argument("--camara", type=int, default=None)
    args = ap.parse_args()

    conf = cfg.cargar()
    indice = args.camara if args.camara is not None else int(conf["vision"]["camara_index"])
    umbral = float(conf["vision"]["postura"]["umbral_de_pie"])
    fps_est = float(conf["vision"]["fps_objetivo"])

    base = LineaBase.cargar()
    print("Línea base:", "calibrada" if base.ajustada else "SIN CALIBRAR (corre src.vision.calibrate)")

    detector = DetectorPose(conf).cargar()
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {indice}")

    previas: dict[int, np.ndarray] = {}
    print("Pulsa 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        personas = detector.detectar(frame)
        for p in personas:
            r = caracteristicas(p, personas, base, previas, fps_est)
            postura = postura_por_umbral(r, umbral) if base.ajustada else "?"
            x1, y1 = int(p.recuadro[0]), int(p.recuadro[1])
            cv2.putText(
                frame, f"{postura} d={r['desviacion_linea_base']:+.2f} v={r['velocidad_munecas']:.3f}",
                (x1, max(24, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )
        previas = {p.id: munecas_de(p) for p in personas}
        dibujar(frame, personas)
        cv2.imshow("Caracteristicas por persona", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
