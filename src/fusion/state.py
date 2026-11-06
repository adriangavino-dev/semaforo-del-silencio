"""Regla de decisión del semáforo — sección 7.

La regla completa es:

    SI clase_audio == "voz_docente"        -> VERDE
       (el docente hablando no es ruido disruptivo, aunque el nivel sea alto)
    SI clase_audio == "murmullo" Y nivel_db > umbral_amarillo  -> AMARILLO
    SI clase_audio == "murmullo" Y nivel_db > umbral_rojo      -> ROJO
    SI clase_audio == "ruido_brusco"       -> evento puntual, no cambia el estado
    SI evento_visual == "movimiento_anomalo" -> evento con zona y marca de tiempo

Un estado propuesto tiene que repetirse varios ciclos seguidos antes de aplicarse.
Sin esa histéresis el semáforo parpadearía con cada predicción aislada, y un
semáforo que parpadea deja de creerse. Un "ciclo" es un segundo: la máquina se
llama en cada cuadro de video, pero las repeticiones se cuentan como mucho una
vez por segundo, para que confirmaciones_estado = 3 signifique 3 s y no 3 cuadros.

Mientras no exista el clasificador de audio entrenado, clase_audio llega como
None y la regla de arriba daría siempre VERDE. En ese caso se decide solo por
nivel, igual que hace el ESP32 con su micrófono: los primeros segundos de la
sesión fijan el ruido de fondo y AMARILLO/ROJO se disparan al subir cierta
cantidad de dB sobre él (audio.db.sin_modelo en config.yaml). Es un respaldo
declarado, no la regla de la sección 7.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

VERDE = "VERDE"
AMARILLO = "AMARILLO"
ROJO = "ROJO"
ESTADOS = (VERDE, AMARILLO, ROJO)
RANGO = {VERDE: 0, AMARILLO: 1, ROJO: 2}


def estado_crudo(clase_audio: str | None, nivel_db: float, conf: dict) -> str:
    """Estado que corresponde a esta lectura, sin histéresis."""
    umbrales = conf["audio"]["db"]

    # La voz del docente no cuenta como ruido... salvo que sea un grito. Por
    # encima de voz_docente_max_db se trata como murmullo: medido en el aula,
    # el modelo llamaba "voz_docente" a segundos de ruido a 80 dB.
    if clase_audio == "voz_docente" and nivel_db > float(umbrales.get("voz_docente_max_db", 70)):
        clase_audio = "murmullo"
    if clase_audio == "voz_docente":
        return VERDE
    if clase_audio == "murmullo":
        # El orden importa: rojo es más exigente y se evalúa primero.
        if nivel_db > float(umbrales["umbral_rojo"]):
            return ROJO
        if nivel_db > float(umbrales["umbral_amarillo"]):
            return AMARILLO
        return VERDE
    # silencio, ruido_brusco o sin clasificar todavía
    return VERDE


def estado_por_nivel(nivel_db: float, base_db: float | None, actual: str, conf: dict) -> str:
    """Decisión solo por dB, con histéresis como en el firmware: se sube al
    superar el umbral y se baja solo al caer un margen por debajo.

    Con base_db (micrófono de la laptop, dB relativos) los umbrales van sobre el
    fondo del aula. Sin base_db (dB calibrados del ESP32) se usan los umbrales
    absolutos umbral_amarillo / umbral_rojo, los mismos del firmware.
    """
    db = conf["audio"]["db"]
    sm = db.get("sin_modelo", {})
    if base_db is None:
        amarillo = float(db["umbral_amarillo"])
        rojo = float(db["umbral_rojo"])
    else:
        amarillo = base_db + float(sm.get("amarillo_sobre_base", 8.0))
        rojo = base_db + float(sm.get("rojo_sobre_base", 15.0))
    margen = float(sm.get("margen_bajada", 2.0))
    if actual == ROJO:
        if nivel_db >= rojo - margen:
            return ROJO
        return AMARILLO if nivel_db >= amarillo - margen else VERDE
    if actual == AMARILLO:
        if nivel_db > rojo:
            return ROJO
        return AMARILLO if nivel_db >= amarillo - margen else VERDE
    if nivel_db > rojo:
        return ROJO
    return AMARILLO if nivel_db > amarillo else VERDE


class MaquinaEstado:
    """Aplica la regla de decisión y sostiene el estado con histéresis."""

    def __init__(self, configuracion: dict | None = None) -> None:
        self.conf = configuracion or cfg.cargar()
        self.confirmaciones = int(self.conf["fusion"]["confirmaciones_estado"])
        self.evento_visual_cambia_estado = bool(
            self.conf["fusion"].get("evento_visual_cambia_estado", False)
        )
        self.estado: str = VERDE
        self._candidato: str = VERDE
        self._repeticiones: int = 0
        self.cambios: list[dict] = []
        # Respaldo sin modelo: ruido de fondo = mediana de los últimos segundos,
        # así ni un ruido al arrancar ni un bajón del micrófono se vuelven "lo normal".
        sm = self.conf["audio"]["db"].get("sin_modelo", {})
        self._n_base = int(sm.get("segundos_base", 5))
        self._ventana_db: deque[float] = deque(maxlen=int(sm.get("segundos_ventana", 60)))
        self._ultima_muestra_t = 0.0   # la ventana se llena una vez por segundo, no por cuadro
        self._ultima_confirmacion_t = float("-inf")
        self.base_db: float | None = None
        # Con el ESP32 como fuente los dB ya vienen calibrados: umbrales absolutos.
        self.nivel_absoluto = str(self.conf["audio"].get("fuente", "laptop")).lower() == "esp32"

    def actualizar(
        self,
        clase_audio: str | None,
        nivel_db: float,
        evento_visual: str | None = None,
        n_de_pie: int | None = None,
        nivel_db_suave: float | None = None,
        t: float | None = None,
    ) -> dict:
        """Procesa una lectura. Devuelve el estado vigente y si acaba de cambiar.

        nivel_db_suave es el promedio de los últimos segundos; lo usa solo el
        respaldo sin modelo. Si no viene, se usa el nivel crudo.
        t es el reloj de la sesión en segundos (el del video cuando se analiza
        un archivo); si no viene, se usa el reloj de la máquina.
        """
        ahora = time.time() if t is None else float(t)
        if clase_audio is None and nivel_db <= 0:
            # Sin audio en este momento (micrófono parado o aún cargando): no
            # hay con qué decidir, así que se mantiene el estado vigente.
            propuesto = self.estado
            motivo = "sin_audio"
        elif clase_audio is None and self.nivel_absoluto:
            # dB calibrados del ESP32: umbrales fijos, como en el firmware.
            nivel = float(nivel_db if nivel_db_suave is None else nivel_db_suave)
            propuesto = estado_por_nivel(nivel, None, self.estado, self.conf)
            motivo = "nivel_db"
        elif clase_audio is None:
            # Sin clasificador: se compara con el fondo reciente del aula.
            nivel = float(nivel_db if nivel_db_suave is None else nivel_db_suave)
            if ahora - self._ultima_muestra_t >= 1.0:
                self._ultima_muestra_t = ahora
                self._ventana_db.append(nivel)
            if len(self._ventana_db) < self._n_base:
                propuesto = VERDE
            else:
                ordenados = sorted(self._ventana_db)
                self.base_db = ordenados[len(ordenados) // 2]   # mediana
                propuesto = estado_por_nivel(nivel, self.base_db, self.estado, self.conf)
            motivo = "nivel_db"
        else:
            nivel = float(nivel_db if nivel_db_suave is None else nivel_db_suave)
            propuesto = estado_crudo(clase_audio, nivel, self.conf)
            motivo = "audio"
            # Sube por clase + nivel, pero solo BAJA por nivel. Si el modelo
            # llama "voz_docente" o "silencio" a un segundo suelto en pleno
            # ruido, el color no cae: se mantiene hasta que el nivel baje de
            # verdad (umbral del color actual menos margen_bajada).
            if RANGO[propuesto] < RANGO[self.estado] and nivel > 0:
                por_nivel = estado_por_nivel(nivel, None if self.nivel_absoluto else self.base_db,
                                             self.estado, self.conf)
                if RANGO[por_nivel] > RANGO[propuesto]:
                    propuesto = por_nivel
                    motivo = "nivel_db"

        # Desviación configurable de la sección 7: un movimiento anómalo
        # confirmado sube el semáforo al menos a AMARILLO. Ver config.yaml.
        if self.evento_visual_cambia_estado and evento_visual == "movimiento_anomalo":
            if propuesto == VERDE:
                propuesto = AMARILLO
                motivo = "evento_visual"

        if propuesto != self._candidato:
            self._candidato = propuesto
            self._repeticiones = 1
            self._ultima_confirmacion_t = ahora
        elif ahora - self._ultima_confirmacion_t >= 1.0:
            self._repeticiones += 1
            self._ultima_confirmacion_t = ahora
        if motivo == "evento_visual":
            # El evento de la cámara ya viene confirmado por el detector
            # (confirmaciones_evento ventanas seguidas): no se espera otra vez.
            self._repeticiones = max(self._repeticiones, self.confirmaciones)

        cambio = False
        if self._candidato != self.estado and self._repeticiones >= self.confirmaciones:
            anterior = self.estado
            self.estado = self._candidato
            cambio = True
            # `motivo` dice qué provocó el cambio: "audio" (regla de la sección 7)
            # o "evento_visual" (movimiento anómalo confirmado). Cuando se vuelve a
            # VERDE, es porque cesó lo que lo había subido.
            self.cambios.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "de": anterior,
                "a": self.estado,
                "motivo": motivo if self.estado != VERDE else "normalizado",
                "clase_audio": clase_audio,
                "nivel_db": round(float(nivel_db), 1),
                "evento": evento_visual,
            })

        return {
            "estado": self.estado,
            "cambio": cambio,
            "estado_propuesto": propuesto,
            "clase_audio": clase_audio,
            "nivel_db": float(nivel_db),
            "evento": evento_visual,
            "n_de_pie": n_de_pie,
        }

    def payload_esp32(self, lectura: dict) -> dict:
        """Cuerpo JSON del POST al ESP32 — contrato de la sección 8."""
        return {
            "estado": lectura["estado"],
            "origen": "ia",
            "clase_audio": lectura.get("clase_audio"),
            "n_de_pie": lectura.get("n_de_pie"),
            "evento": lectura.get("evento"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


if __name__ == "__main__":
    conf = cfg.cargar()
    print("Prueba de la regla de decisión\n")
    casos = [
        ("voz_docente", 72.0, "el docente habla fuerte: no debe alertar"),
        ("silencio", 30.0, "aula en silencio"),
        ("murmullo", 50.0, "murmullo por debajo del umbral"),
        ("murmullo", 58.0, "murmullo sobre el umbral amarillo"),
        ("murmullo", 70.0, "murmullo sobre el umbral rojo"),
        ("ruido_brusco", 80.0, "evento puntual: no cambia el estado sostenido"),
    ]
    for clase, db, nota in casos:
        print(f"  {clase:<14} {db:5.1f} dB  ->  {estado_crudo(clase, db, conf):<9} {nota}")

    print(f"\nHistéresis: hacen falta {conf['fusion']['confirmaciones_estado']} lecturas seguidas")
    m = MaquinaEstado(conf)
    for i in range(5):
        r = m.actualizar("murmullo", 70.0)
        print(f"  lectura {i + 1}: propuesto={r['estado_propuesto']:<9} vigente={r['estado']:<9}"
              f"{'  <- cambio' if r['cambio'] else ''}")
