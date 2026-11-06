"""Detección de personas dormidas en el pupitre, a partir del esqueleto.

No hay un dataset público de aulas con la clase "dormido" que se pueda
descargar sin cuenta (los que la tienen están en Roboflow Universe, piden
iniciar sesión, o no son públicos). Por eso es una regla sobre los puntos del
cuerpo que ya da YOLOv8-Pose, igual que la postura de pie / sentado:

  1. Cabeza baja: la cabeza (nariz, ojos, orejas) está a la altura de los
     hombros o por debajo. Se mide como
         (y_hombros - y_cabeza) / largo_del_torso
     De pie o sentado erguido vale ~0,4-0,6; con la cabeza apoyada en el
     pupitre, ~0 o negativo. Si los hombros se ven pero ningún punto de la
     cabeza (cara escondida entre los brazos), también cuenta como cabeza baja.
  1b. U ojos cerrados (src/vision/ojos.py), cuando la cara se ve lo bastante
     grande para decidirlo: alguien dormido sentado con la cabeza erguida.
  2. Quieto: las muñecas casi no se mueven (escribir o leer con la cabeza
     agachada mueve las manos; dormir no).
  3. Sostenido: se lleva una media móvil (constante de tiempo `tolerancia_s`)
     de "cumple 1 y 2". El reloj de dormido corre mientras esa media es >= 0,7
     (la mayoría de los últimos segundos con la cabeza baja) y vuelve a cero
     si cae por debajo de 0,4. Así un parpadeo del esqueleto no lo reinicia, y
     los ratos sueltos de cabeza baja de un alumno despierto (en el video del
     aula: 3-12 % de los cuadros, nunca más de 1,2 s seguidos) no se suman.

Se sigue a cada persona por su ID de ByteTrack. Si el seguidor le cambia el ID,
el contador vuelve a empezar.

Todo se ajusta en config.yaml, sección vision.dormido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CABEZA = (0, 1, 2, 3, 4)          # nariz, ojos, orejas (COCO)
HOMBROS = (5, 6)
CADERAS = (11, 12)


def altura_cabeza(p, conf_min: float = 0.3) -> float | None:
    """(y_hombros - y_cabeza) / torso. None si no se ven los hombros.

    Positivo = cabeza por encima de los hombros. Si la cabeza no se ve pero los
    hombros sí, devuelve 0 (cabeza escondida: probablemente apoyada).
    """
    c = p.confianzas
    hombros = [p.keypoints[i][1] for i in HOMBROS if c[i] > conf_min]
    if not hombros:
        return None
    y_hombros = float(np.mean(hombros))
    caderas = [p.keypoints[i][1] for i in CADERAS if c[i] > conf_min]
    torso = abs(float(np.mean(caderas)) - y_hombros) if caderas else 0.0
    if torso < 0.15 * p.alto_recuadro:            # caderas tapadas por el pupitre
        torso = 0.35 * p.alto_recuadro
    cabeza = [p.keypoints[i][1] for i in CABEZA if c[i] > conf_min]
    if not cabeza:
        return 0.0
    return (y_hombros - float(np.mean(cabeza))) / max(torso, 1.0)


@dataclass
class _Seguimiento:
    acumulado: float = 0.0       # segundos seguidos con cabeza baja y quieto
    ultimo_t: float = 0.0
    media: float = 0.0           # fracción reciente de cuadros con cabeza baja y quieto
    dormido: bool = False


class DetectorDormidos:
    def __init__(self, conf: dict) -> None:
        cd = conf["vision"].get("dormido", {})
        self.habilitado = bool(cd.get("habilitado", True))
        self.umbral_cabeza = float(cd.get("umbral_cabeza", 0.12))
        self.max_velocidad = float(cd.get("max_velocidad_munecas", 0.25))
        self.segundos_minimos = float(cd.get("segundos_minimos", 20))
        self.tolerancia_s = float(cd.get("tolerancia_s", 3))
        self._por_id: dict[int, _Seguimiento] = {}

    def actualizar(self, personas: list, velocidades: dict[int, float], t: float,
                   ojos_cerrados: dict[int, bool] | None = None) -> dict:
        """Procesa un cuadro. Devuelve {'dormidos': {id,...}, 'nuevos': [id,...], 'medidas': {id: (altura, s)}}."""
        if not self.habilitado:
            return {"dormidos": set(), "nuevos": [], "medidas": {}}
        nuevos, medidas, vistos = [], {}, set()
        for p in personas:
            vistos.add(p.id)
            s = self._por_id.setdefault(p.id, _Seguimiento(ultimo_t=t))
            dt = min(max(t - s.ultimo_t, 0.0), 1.0)
            s.ultimo_t = t
            alto = altura_cabeza(p)
            quieto = velocidades.get(p.id, 0.0) <= self.max_velocidad
            cabeza_baja = alto is not None and alto <= self.umbral_cabeza
            cumple = quieto and (cabeza_baja or bool((ojos_cerrados or {}).get(p.id)))
            alfa = dt / (self.tolerancia_s + dt) if dt > 0 else 0.0
            s.media += alfa * ((1.0 if cumple else 0.0) - s.media)
            if s.media >= 0.7:
                s.acumulado += dt
            elif s.media < 0.4:
                s.acumulado = 0.0
                s.dormido = False
            if s.acumulado >= self.segundos_minimos and not s.dormido:
                s.dormido = True
                nuevos.append(p.id)
            medidas[p.id] = (alto, s.acumulado)
        # Olvida a quien lleva mucho sin verse.
        for i in [i for i, s in self._por_id.items() if i not in vistos and t - s.ultimo_t > 30]:
            del self._por_id[i]
        dormidos = {i for i, s in self._por_id.items() if s.dormido and i in vistos}
        return {"dormidos": dormidos, "nuevos": nuevos, "medidas": medidas}
