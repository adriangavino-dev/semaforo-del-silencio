"""Sesión completa: audio + video + fusión + reporte + ESP32.

Es el ejecutor del sistema: mira y escucha el aula, decide el color del semáforo,
registra los eventos y, al terminar, escribe el reporte de la sesión en reports/
sin que haya que hacer nada a mano.

A diferencia de demo_live.py, que solo muestra lo que reconocen los modelos
preentrenados, este script usa el clasificador de audio ya entrenado. Si ese
modelo todavía no existe avisa y sigue funcionando con el resto, para que se
pueda probar la parte de video y de reportes antes de terminar el entrenamiento.

Uso:
    python scripts/run_session.py --minutos 10
    python scripts/run_session.py --minutos 5 --sin-ventana
    python scripts/run_session.py --salon "Aula 302"
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.audio.capture import CapturaAudio
from src.audio.infer import ClasificadorAudio
from src.comm.esp32 import crear_cliente
from src.fusion.grabacion import GrabadorAudio, GrabadorVideo
from src.fusion.reporter import Reporte
from src.fusion.state import MaquinaEstado
from src.vision.events import DetectorEventos
from src.vision.features import (
    LineaBase, caracteristicas, filtrar_personas, muñecas_previas, postura_geometrica,
    postura_por_umbral,
)
from src.vision.pose import DetectorPose, dibujar
from src.vision.evidencias import Evidencias  # noqa: E402
from src.vision.ojos import ClasificadorOjos  # noqa: E402
from src.vision.sleep import DetectorDormidos  # noqa: E402

COLOR_ESTADO = {"VERDE": (60, 200, 60), "AMARILLO": (0, 200, 255), "ROJO": (60, 60, 235)}


def audio_del_esp32(cliente, caducidad_s: float = 3.0) -> dict:
    """Los dB del INMP441 del ESP32, en el mismo formato que devuelve HiloAudio.

    El firmware ya suaviza el nivel y lo calibra (DB_OFFSET), así que se usa tal
    cual. Si la última lectura tiene más de caducidad_s, se informa nivel 0 y
    la máquina de estados mantiene el estado en vez de decidir con datos viejos.
    """
    lectura = cliente.lectura() if hasattr(cliente, "lectura") else None
    base = {"clase": None, "confianza": 0.0, "nivel_db": 0.0, "nivel_db_suave": None,
            "clase_bloque": None, "fuente": "esp32"}
    if not lectura or lectura.get("nivel_db") is None:
        return {**base, "caducada": True}
    if time.time() - float(lectura.get("recibida_t", 0.0)) > caducidad_s:
        return {**base, "caducada": True}
    nivel = float(lectura["nivel_db"])
    return {**base, "nivel_db": nivel, "nivel_db_suave": nivel}


class HiloAudio(threading.Thread):
    """Clasifica el micrófono en un hilo aparte para no frenar el video."""

    def __init__(self, conf: dict, captura=None) -> None:
        super().__init__(daemon=True)
        self.conf = conf
        # Por defecto el micrófono de la laptop; con audio.fuente: esp32, el del aula.
        self.captura = captura or CapturaAudio(conf)
        self.clf = ClasificadorAudio(conf)
        self._detener = threading.Event()
        self._lock = threading.Lock()
        self.ultima: dict = {"clase": None, "confianza": 0.0, "nivel_db": 0.0, "clase_bloque": None}
        self._ultima_t = 0.0
        self.caducidad_s = 3.0   # más de esto sin bloque nuevo = el micrófono no está entregando
        self.listo = False
        self.con_modelo = False
        self.error: str | None = None
        self.grabador: GrabadorAudio | None = None   # audio.wav de la sesión

    def run(self) -> None:
        try:
            self.clf.cargar(obligatorio=False)
            self.captura.iniciar()
            with self._lock:
                self.listo = True
                self.con_modelo = self.clf.disponible
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.error = str(exc)
            print(f"[audio] error de inicialización: {exc}", file=sys.stderr)
            return

        while not self._detener.is_set():
            bloque = self.captura.leer(timeout=0.5)
            if bloque is None:
                continue
            if self.grabador:
                self.grabador.escribir(bloque)
            try:
                lectura = self.clf.clasificar_audio(bloque)
            except Exception as exc:  # noqa: BLE001
                print(f"[audio] error de inferencia: {exc}", file=sys.stderr)
                continue
            with self._lock:
                self.ultima = lectura
                self._ultima_t = time.time()

        self.captura.detener()

    def leer(self) -> dict:
        """Última lectura; si está caducada, se informa como "sin audio" (nivel 0).

        Así una parada del micrófono no deja congelado el último valor: la
        máquina de estados mantiene el estado en lugar de decidir con datos viejos.
        """
        with self._lock:
            lectura = dict(self.ultima)
            caducada = self._ultima_t > 0 and time.time() - self._ultima_t > self.caducidad_s
        if caducada:
            lectura.update({"clase": None, "confianza": 0.0, "nivel_db": 0.0, "nivel_db_suave": None,
                            "clase_bloque": None, "caducada": True})
        return lectura

    def detener(self) -> None:
        self._detener.set()


def analizar_frame(
    frame: np.ndarray, detector: DetectorPose, base: LineaBase, conf: dict,
    previas: dict[int, np.ndarray], eventos: DetectorEventos, fps: float = 30.0,
) -> dict:
    """Salida por frame — firma de la sección 6.4.

    La decisión de evento no se toma aquí cuadro a cuadro: los rasgos de cada
    persona se entregan a `eventos`, que decide por ventana de ~2 s y confirma
    por repetición. Esta función solo informa lo que el detector de eventos ya
    decidió.
    """
    cv = conf["vision"]
    cp = cv["postura"]
    umbral_postura = float(cp["umbral_de_pie"])
    aspecto_de_pie = float(cp["aspecto_de_pie"])
    angulo_de_pie = float(cp["angulo_rodilla_de_pie"])
    umbral_pico = float(cv["evento"]["umbral_velocidad_muneca_por_s"])
    max_fisica = float(cv["evento"]["max_velocidad_fisica_por_s"])
    filtro = cv["filtro_personas"]
    alto, ancho = frame.shape[:2]

    personas = filtrar_personas(
        detector.detectar(frame), alto,
        float(filtro["min_alto_fraccion"]), float(filtro["min_conf_media"]),
    )
    detalle, n_de_pie, rasgos_todos, zonas_rapidas = [], 0, [], []

    for p in personas:
        rasgos = caracteristicas(p, personas, base, previas, fps, max_fisica)
        zona = p.zona(ancho, alto, cv["zonas"])
        postura = (
            postura_por_umbral(rasgos, umbral_postura) if base.ajustada
            else postura_geometrica(p, rasgos, aspecto_de_pie, angulo_de_pie)
        )
        if postura == "de_pie":
            n_de_pie += 1
        if rasgos["velocidad_munecas"] > umbral_pico:
            zonas_rapidas.append(zona)
        rasgos_todos.append(rasgos)
        detalle.append({
            "id": p.id, "postura": postura, "zona": zona,
            "conf": round(p.conf_deteccion, 2),
            "velocidad": round(rasgos["velocidad_munecas"], 2),
        })

    decision = eventos.agregar_cuadro(rasgos_todos)

    # Zona del evento: la de quienes superan el pico en este cuadro; si en este
    # cuadro concreto nadie lo supera, la de la persona que más se mueve.
    if zonas_rapidas:
        zona_evento = max(set(zonas_rapidas), key=zonas_rapidas.count)
    elif detalle:
        zona_evento = max(detalle, key=lambda d: d["velocidad"])["zona"]
    else:
        zona_evento = None

    return {
        "personas": detalle,
        "n_total": len(personas),
        "n_de_pie": n_de_pie,
        "evento": decision["evento"],
        "evento_confirmado": decision["confirmado"],
        "evento_nuevo": decision["nuevo"],
        "puntaje_evento": decision["puntaje"],
        "zona_evento": zona_evento,
        "_decision": decision,
        "_objetos": personas,
    }


def panel(frame: np.ndarray, lineas: list[tuple[str, tuple[int, int, int], float]], estado: str) -> None:
    """Franja superior con el estado y las lecturas.

    El texto se achica si alguna línea no cabe en el ancho del cuadro (videos
    verticales o cámaras de baja resolución).
    """
    ancho = frame.shape[1]
    k = 1.0
    for texto, _, escala in lineas:
        w = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0]
        k = min(k, (ancho - 20) / max(w, 1))
    k = max(k, 0.45)
    paso = int(round(22 * max(k, 0.7)))
    alto_panel = paso * len(lineas) + 18
    capa = frame.copy()
    cv2.rectangle(capa, (0, 0), (ancho, alto_panel), (0, 0, 0), -1)
    cv2.addWeighted(capa, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (0, 0), (ancho, 6), COLOR_ESTADO.get(estado, (200, 200, 200)), -1)

    y = 6 + paso
    for texto, color, escala in lineas:
        cv2.putText(frame, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala * k, color, 1, cv2.LINE_AA)
        y += paso


GRIS = (170, 170, 170)
BLANCO = (255, 255, 255)
CORTO = {"silencio": "sil", "voz_docente": "voz", "murmullo": "mur", "ruido_brusco": "gol"}


def etiquetar_personas(frame: np.ndarray, vision: dict) -> None:
    """Debajo del "#ID conf" de cada recuadro: postura y velocidad de muñecas."""
    por_id = {d["id"]: d for d in vision["personas"]}
    for p in vision["_objetos"]:
        d = por_id.get(p.id)
        if not d:
            continue
        x1, _, _, y2 = (int(v) for v in p.recuadro)
        de_pie = d["postura"] == "de_pie"
        if d.get("dormido"):
            texto, color = "DORMIDO", (255, 80, 200)
        else:
            texto = f"{'DE PIE' if de_pie else 'sentado'}  v{d['velocidad']:.1f}"
            color = (0, 140, 255) if de_pie else (80, 220, 80)
            if d.get("ojos") is not None and d["ojos"] >= 0.5:
                texto += "  ojos cerrados"
            if d.get("seg_cabeza_baja", 0) >= 3:      # va camino de "dormido"
                texto += f"  zz {d['seg_cabeza_baja']:.0f}s"
        cv2.putText(frame, texto, (x1 + 2, max(14, y2 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)


def lineas_panel(lectura, maquina, audio, estado_audio, vision, de_pie, conf, *,
                 fps, reporte, cliente, servo, tiempo) -> list:
    """Todo lo que explica la decisión del semáforo, en 6 líneas."""
    db = conf["audio"]["db"]
    estado = lectura["estado"]

    # 1. Estado, candidato y cuántas confirmaciones lleva
    cand, rep_, nec = maquina._candidato, maquina._repeticiones, maquina.confirmaciones
    ultimo = maquina.cambios[-1]["motivo"] if maquina.cambios else "-"
    l1 = f"{estado}   motivo: {ultimo}"
    if cand != estado:
        l1 += f"   -> {cand}? {min(rep_, nec)}/{nec} s"

    # 2. Audio: clase, confianza y probabilidad de cada clase
    probs = audio.get("probabilidades") or {}
    txt_probs = "  ".join(f"{CORTO.get(c, c)} {p:.2f}" for c, p in probs.items())
    l2 = f"AUDIO   {estado_audio} ({audio.get('confianza', 0):.2f})   {txt_probs}"

    # 3. Nivel: dB frente a los umbrales
    nivel = audio.get("nivel_db_suave") or audio.get("nivel_db") or 0.0
    l3 = (f"NIVEL   {nivel:.0f} dB   umbral amarillo {db['umbral_amarillo']:.0f} / rojo {db['umbral_rojo']:.0f}"
          + ("   (sin lecturas del ESP32)" if audio.get("caducada") else ""))

    # 4. Cámara
    l4 = (f"CAMARA  personas {vision['n_total']}   de pie {de_pie}   ojos cerrados {vision.get('n_ojos_cerrados', 0)}"
          f"   dormidos {vision.get('n_dormidos', 0)}   {fps:.0f} FPS")
    if servo:
        l4 += f"   servo pos {servo[0]}/{servo[1] - 1}"

    # 5. Movimiento: cifras de la regla de eventos frente a sus umbrales
    d = vision.get("_decision") or {}
    l5 = (f"MOVIM.  vel media {d.get('velocidad_media', 0):.2f}/{d.get('umbral_media', 0):.2f}   "
          f"rapidos {100 * d.get('fraccion_rapidos', 0):.0f}%/{100 * d.get('umbral_fraccion', 0):.0f}%   "
          f"eventos {len(reporte.eventos)}")

    # 6. ESP32 y tiempo
    est = cliente.estadisticas() if hasattr(cliente, "estadisticas") else {}
    led = (cliente.lectura() or {}).get("estado_led", "?") if hasattr(cliente, "lectura") else "?"
    enlace = "ok" if est.get("conectado", est.get("habilitado")) else "SIN CONEXION"
    l6 = f"ESP32   {enlace}   LED {led}"
    if "audio_bloques" in est:
        l6 += f"   audio {est['audio_bloques']} bloques, {est['audio_bloques_perdidos']} perdidos"
    l6 += f"   {tiempo}"

    return [
        (l1, COLOR_ESTADO.get(estado, BLANCO), 0.7),
        (l2, BLANCO, 0.45),
        (l3, BLANCO, 0.45),
        (l4, BLANCO, 0.45),
        (l5, BLANCO, 0.45),
        (l6, GRIS, 0.45),
    ]


def cargar_linea_base(posicion: int) -> LineaBase:
    """Calibración por posición del servo: p0 usa el archivo base, las demás _pN."""
    ruta = cfg.ruta_de("calibracion")
    if posicion > 0:
        ruta = ruta.with_name(f"{ruta.stem}_p{posicion}{ruta.suffix}")
    return LineaBase.cargar(ruta)


def main() -> int:
    conf = cfg.cargar()
    ap = argparse.ArgumentParser(description="Sesión completa con reporte")
    ap.add_argument("--minutos", type=float, default=None,
                    help="duración de la sesión; sin este argumento corre hasta pulsar q o Ctrl+C")
    ap.add_argument("--camara", type=int, default=None)
    ap.add_argument("--video", help="analizar una grabación en vez de la cámara")
    ap.add_argument("--salon", default=None)
    ap.add_argument("--sin-ventana", action="store_true", help="correr sin mostrar video")
    args = ap.parse_args()

    cv = conf["vision"]
    indice = args.camara if args.camara is not None else int(cv["camara_index"])
    desde_archivo = args.video is not None

    base = LineaBase.cargar()
    if not base.ajustada:
        print("[vision] sin calibración de línea base: la postura se decide por geometría "
              "(alto/ancho del recuadro y ángulo de rodilla).")

    # Con un archivo no hay micrófono en vivo. OpenCV solo lee la imagen, así que
    # la sesión sale sin clasificación de audio y el reporte lo deja registrado.
    # Con audio.fuente: esp32 el micrófono de la laptop no se abre: los dB los
    # mide el INMP441 del ESP32 y llegan con cada lectura suya.
    fuente_audio = str(conf["audio"].get("fuente", "laptop")).lower()
    hilo = None
    if not desde_archivo and fuente_audio != "esp32":
        hilo = HiloAudio(conf)
        hilo.start()

    print("[vision] cargando YOLOv8-Pose...")
    detector = DetectorPose(conf).cargar()
    eventos = DetectorEventos(conf, fps=float(cv["fps_objetivo"])).cargar_modelo()
    print("[vision] eventos:", "modelo entrenado" if eventos.con_modelo else "regla provisional (sin models/vision_evento_clf.pkl)")

    if desde_archivo:
        ruta_video = Path(args.video)
        if not ruta_video.exists():
            print(f"No se encontró el archivo {ruta_video}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(ruta_video))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_cuadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[vision] {ruta_video.name}: {n_cuadros} cuadros a {fps_video:.1f} FPS "
              f"({n_cuadros / max(fps_video, 1):.1f} s)")
        eventos = DetectorEventos(conf, fps=fps_video).cargar_modelo()
    else:
        # Lector en segundo plano que se queda con el cuadro más reciente: con
        # cámaras por Wi-Fi (iVCam) el video ya no se va atrasando.
        from src.vision.camara import CamaraEnVivo

        cap = CamaraEnVivo(indice, int(cv["ancho"]), int(cv["alto"]))
        fps_video, n_cuadros = 0.0, 0

    if not cap.isOpened():
        origen = args.video if desde_archivo else f"la cámara {indice}"
        print(f"No se pudo abrir {origen}.", file=sys.stderr)
        if hilo:
            hilo.detener()
        return 1

    maquina = MaquinaEstado(conf)
    dormidos = DetectorDormidos(conf)
    ojos = ClasificadorOjos(conf).cargar()
    print("[vision] ojos cerrados:", "modelo CEW cargado" if ojos.disponible else "sin modelo (python -m src.vision.ojos --entrenar)")
    prob_ojos: dict[int, float] = {}
    ultimo_ojos = float("-inf")
    reporte = Reporte(conf, salon=args.salon)
    cg = conf.get("grabacion", {})
    evidencias = Evidencias(conf, reporte.carpeta)
    grabador_video = (GrabadorVideo(reporte.carpeta / "video.mp4", float(cv["fps_objetivo"]))
                      if cg.get("video") and not desde_archivo else None)
    print(f"[sesion] carpeta: sesiones/{reporte.sesion_id}  (" + ", ".join(
        n for n, si in (("video", grabador_video), ("audio", cg.get("audio") and not desde_archivo),
                        ("fotos", evidencias.habilitado)) if si) + ")")
    cliente = crear_cliente(conf)
    if fuente_audio == "esp32" and not desde_archivo:
        # Micrófono del aula: el ESP32 manda el audio por el cable y aquí corren
        # YAMNet + el clasificador. Los dB para la regla siguen siendo los del
        # ESP32 (calibrados con DB_OFFSET), no los que calcula la laptop.
        if bool(conf["esp32"].get("audio_remoto", True)) and hasattr(cliente, "iniciar_audio"):
            from src.audio.captura_esp32 import CapturaESP32

            hilo = HiloAudio(conf, captura=CapturaESP32(cliente, conf))
            if cg.get("audio"):
                hilo.grabador = GrabadorAudio(reporte.carpeta / "audio.wav", int(conf["audio"]["sample_rate"]))
            hilo.start()
            print(f"[audio] fuente: micrófono del ESP32 (INMP441) por {conf['esp32'].get('transporte')} "
                  "-> YAMNet + clasificador")
        else:
            print("[audio] fuente: micrófono del ESP32, solo dB (sin audio remoto)")

    previas: dict[int, np.ndarray] = {}
    ultimo_registro = 0.0
    # Un mismo disturbio no debe generar decenas de entradas: tras registrar un
    # evento se espera antes de admitir otro del mismo tipo.
    espera_evento_s = float(conf["vision"]["evento"]["espera_entre_eventos_s"])
    intervalo_publicacion = float(conf["dashboard"]["publicar_cada_s"])
    ultima_publicacion = 0.0

    def extra_reporte() -> dict:
        return {
            "modelo_audio_entrenado": bool(hilo and hilo.con_modelo),
            "linea_base_calibrada": base.ajustada,
            "fuente": f"video: {Path(args.video).name}" if desde_archivo else f"camara {indice}",
            "audio_disponible": hilo is not None or (fuente_audio == "esp32" and not desde_archivo),
            "fuente_audio": "archivo" if desde_archivo else fuente_audio,
            "modelo_evento_entrenado": eventos.con_modelo,
            "metodo_postura": "linea_base" if base.ajustada else "geometria",
            "servo": {"habilitado": servo_activo, "posiciones": n_posiciones, "posicion_final": posicion},
            "esp32": cliente.estadisticas(),
        }
    # -inf y no 0: con un video el reloj de la sesión arranca en cero, así que
    # un cero aquí bloquearía el primer evento durante toda la espera.
    ultimo_evento = float("-inf")
    fps_hist: deque[float] = deque(maxlen=30)
    fin = time.time() + args.minutos * 60.0 if args.minutos else float("inf")
    n_frame = 0

    def reloj() -> float:
        """Tiempo transcurrido de la sesión.

        Con un archivo es el tiempo del propio video: se analiza más rápido que
        en tiempo real, así que el reloj de pared daría una duración equivocada.
        """
        return (n_frame / fps_video) if desde_archivo else time.time()

    # ---- servo: la cámara recorre posiciones fijas y el análisis se pausa en
    # cada giro. Durante el giro todo lo que se ve se desplaza, y sin la pausa
    # el sistema creería que todas las personas se movieron a la vez.
    cs = conf["servo"]
    servo_activo = bool(cs["habilitado"]) and not desde_archivo
    n_posiciones = max(int(cs["posiciones"]), 1)
    permanencia_s = float(cs["permanencia_s"])
    tiempo_giro_s = float(cs["tiempo_giro_s"])
    zonas_por_pos = cs.get("zonas_por_posicion") or {}
    posicion = 0
    proximo_giro = reloj() + permanencia_s
    sentido_giro = 1   # +1 hacia la última posición, -1 de vuelta hacia la 0
    pausa_hasta = float("-inf")
    conf_pos = conf  # copia de la configuración con las zonas de la posición actual

    def cambiar_posicion(nueva: int) -> None:
        """Ordena el giro y deja el análisis listo para la nueva posición."""
        nonlocal posicion, pausa_hasta, base, previas, conf_pos
        if not cliente.girar(nueva):
            print(f"  servo: no se pudo girar a {nueva}; se sigue en la posición {posicion}")
            return
        posicion = nueva
        pausa_hasta = reloj() + tiempo_giro_s
        # Lo aprendido en la posición anterior no vale para la nueva.
        detector.reiniciar_seguimiento()
        eventos.reiniciar()
        previas = {}
        base = cargar_linea_base(nueva)
        zonas = zonas_por_pos.get(nueva) or zonas_por_pos.get(str(nueva))
        conf_pos = {**conf, "vision": {**conf["vision"], "zonas": zonas}} if zonas else conf
        print(f"  servo -> posición {nueva}  (análisis en pausa {tiempo_giro_s:.1f} s; "
              f"línea base {'calibrada' if base.ajustada else 'no calibrada'})")

    if desde_archivo:
        print(f"\nAnalizando '{Path(args.video).name}' como sesión de '{reporte.salon}'.\n")
    else:
        duracion = f"por {args.minutos:g} min" if args.minutos else "sin límite (q o Ctrl+C para terminar)"
        print(f"\nSesión en marcha en '{reporte.salon}' {duracion}.")
        print("Pulsa 'q' sobre la ventana o Ctrl+C para terminar antes.\n")

    try:
        while desde_archivo or time.time() < fin:
            ok, frame = cap.read()
            if not ok:
                if desde_archivo:
                    print("Fin del video.")
                else:
                    print("La cámara dejó de entregar cuadros.", file=sys.stderr)
                break
            n_frame += 1
            if grabador_video:
                grabador_video.escribir(frame)   # el cuadro tal cual, antes de dibujar encima

            t0 = time.time()
            # Con archivo se usan los FPS del video; con cámara, los medidos.
            fps_actual = fps_video if desde_archivo else (
                sum(fps_hist) / len(fps_hist) if fps_hist else float(cv["fps_objetivo"])
            )
            eventos.ajustar_fps(fps_actual)

            # Giro programado: solo cuando toca y nunca con un evento en curso.
            if servo_activo and n_posiciones > 1 and reloj() >= proximo_giro:
                proximo_giro = reloj() + permanencia_s
                if eventos.ultima_prediccion["evento"] != "movimiento_anomalo":
                    # Ida y vuelta: 0 -> 1 -> 2 -> 1 -> 0 -> 1 ... (sin saltar de 2 a 0).
                    if not 0 <= posicion + sentido_giro < n_posiciones:
                        sentido_giro = -sentido_giro
                    cambiar_posicion(posicion + sentido_giro)
                    proximo_giro += tiempo_giro_s

            if reloj() < pausa_hasta:
                # Cámara girando: se consume el cuadro pero no se analiza.
                if not args.sin_ventana:
                    cv2.putText(frame, f"CAMARA GIRANDO A POSICION {posicion}", (12, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(f"Semaforo del Silencio - {reporte.salon}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            vision = analizar_frame(frame, detector, base, conf_pos, previas, eventos, fps_actual)
            # Ojos cerrados: cada medio segundo, todas las caras visibles en un lote.
            if ojos.disponible and reloj() - ultimo_ojos >= 0.5:
                ultimo_ojos = reloj()
                prob_ojos = ojos.probabilidades(frame, vision["_objetos"])
            cerrados = {pid: pr >= ojos.umbral for pid, pr in prob_ojos.items()}
            # Dormidos: (cabeza baja u ojos cerrados) y quieto, sostenido (src/vision/sleep.py).
            r_dorm = dormidos.actualizar(
                vision["_objetos"], {d["id"]: d["velocidad"] for d in vision["personas"]}, reloj(), cerrados
            )
            vision["n_dormidos"] = len(r_dorm["dormidos"])
            vision["n_ojos_cerrados"] = sum(1 for d in vision["personas"] if cerrados.get(d["id"]))
            for d in vision["personas"]:
                d["dormido"] = d["id"] in r_dorm["dormidos"]
                d["ojos"] = prob_ojos.get(d["id"])
                d["cabeza"], d["seg_cabeza_baja"] = r_dorm["medidas"].get(d["id"], (None, 0.0))
            for pid in r_dorm["nuevos"]:
                zona = next((d["zona"] for d in vision["personas"] if d["id"] == pid), None)
                foto = evidencias.guardar(frame, vision["_objetos"], maquina.estado, f"persona dormida #{pid}", "evento")
                reporte.registrar_evento("persona_dormida", zona=zona, detalle={"id": pid, "foto": foto},
                                         segundo=reloj() if desde_archivo else None, posicion=posicion)
                print(f"  persona dormida: #{pid} en zona {zona}")
            previas = muñecas_previas(vision["_objetos"])
            fps_hist.append(1.0 / max(time.time() - t0, 1e-6))

            if hilo and fuente_audio == "esp32":
                # Clase del audio del aula + dB calibrados que mide el propio ESP32.
                audio = hilo.leer()
                medido = audio_del_esp32(cliente)
                audio.update({"nivel_db": medido["nivel_db"], "nivel_db_suave": medido["nivel_db_suave"],
                              "fuente": "esp32"})
                if medido.get("caducada"):
                    audio["caducada"] = True
            elif hilo:
                audio = hilo.leer()
            elif fuente_audio == "esp32" and not desde_archivo:
                audio = audio_del_esp32(cliente)
            else:
                audio = {"clase": None, "confianza": 0.0, "nivel_db": 0.0, "clase_bloque": None}

            # El detector de eventos ya decidió por ventana y confirmó por
            # repetición; aquí solo se registra una vez por disturbio.
            evento_confirmado = "movimiento_anomalo" if vision["evento_confirmado"] else None
            if vision["evento_nuevo"] and reloj() - ultimo_evento > espera_evento_s:
                ultimo_evento = reloj()
                foto = evidencias.guardar(frame, vision["_objetos"], maquina.estado,
                                          f"movimiento anomalo ({vision['zona_evento']})", "evento")
                reporte.registrar_evento(
                    "movimiento_anomalo", vision["zona_evento"],
                    detalle={"puntaje": round(vision["puntaje_evento"], 2), "foto": foto},
                    segundo=reloj() if desde_archivo else None,
                    posicion=posicion,
                )
                print(f"  evento: movimiento anómalo en zona {vision['zona_evento']} "
                      f"(puntaje {vision['puntaje_evento']:.2f}, segundo {reloj() if desde_archivo else n_frame / max(fps_actual, 1):.1f})")

            lectura = maquina.actualizar(
                audio["clase"], audio["nivel_db"], evento_confirmado, vision["n_de_pie"],
                nivel_db_suave=audio.get("nivel_db_suave"),
                t=reloj(),
            )

            if audio["clase_bloque"] == "ruido_brusco":
                reporte.registrar_evento("ruido_brusco", detalle={"nivel_db": round(audio["nivel_db"], 1)})

            # Se llama en cada vuelta: el cliente manda solo si cambió el estado o
            # si toca refrescarlo (el ESP32 lo mantiene mientras la laptop hable).
            cliente.enviar(maquina.payload_esp32(lectura))
            if lectura["cambio"]:
                ultimo = maquina.cambios[-1]
                ultimo["foto"] = evidencias.guardar(
                    frame, vision["_objetos"], lectura["estado"],
                    f"{ultimo['de']} -> {ultimo['a']} ({ultimo['motivo']}, {audio['nivel_db']:.0f} dB)", "cambio")
                reporte.cambios_estado = maquina.cambios
                print(f"  estado -> {lectura['estado']}  ({audio['clase']}, {audio['nivel_db']:.0f} dB)"
                      + (f"  foto: {ultimo['foto']}" if ultimo.get("foto") else ""))

            # Se guarda una fila por segundo, no una por cuadro.
            ahora = reloj()
            if ahora - ultimo_registro >= 1.0:
                ultimo_registro = ahora
                reporte.registrar(
                    estado=lectura["estado"],
                    clase_audio=audio["clase"],
                    confianza_audio=audio["confianza"],
                    nivel_db=audio["nivel_db"],
                    n_personas=vision["n_total"],
                    n_de_pie=vision["n_de_pie"],
                    evento=evento_confirmado,
                    zona_evento=vision["zona_evento"],
                    segundo=ahora if desde_archivo else None,
                    posicion=posicion,
                    n_dormidos=vision.get("n_dormidos", 0),
                )

            # El tablero lee sesiones/, así que la sesión se publica cada pocos
            # segundos mientras corre. Sin esto no aparecería nada en la web
            # hasta terminar la sesión.
            if ahora - ultima_publicacion >= intervalo_publicacion:
                ultima_publicacion = ahora
                reporte.cambios_estado = maquina.cambios
                reporte.guardar_parcial(lectura["estado"], extra=extra_reporte())

            if not args.sin_ventana:
                dibujar(frame, vision["_objetos"])
                etiquetar_personas(frame, vision)
                de_pie = "n/d" if vision["n_de_pie"] is None else str(vision["n_de_pie"])
                if hilo:
                    estado_audio = (
                        audio["clase"] if hilo.con_modelo
                        else ("sin modelo entrenado" if hilo.listo else "cargando...")
                    )
                else:
                    estado_audio = "ESP32" if audio.get("fuente") == "esp32" else "sin audio"
                    if audio.get("caducada"):
                        estado_audio = "ESP32 sin lecturas"
                panel(frame, lineas_panel(
                    lectura, maquina, audio, estado_audio, vision, de_pie, conf,
                    fps=sum(fps_hist) / len(fps_hist), reporte=reporte, cliente=cliente,
                    servo=(posicion, n_posiciones) if servo_activo else None,
                    tiempo=(f"video {reloj():.1f}s / {n_cuadros / max(fps_video, 1):.1f}s" if desde_archivo
                            else f"sesion {reporte.duracion_s / 60:.1f} min" + (
                                f"  (restan {max(0, fin - time.time()) / 60:.1f})" if fin != float("inf") else "")),
                ), lectura["estado"])
                cv2.imshow(f"Semaforo del Silencio - {reporte.salon}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nSesión cortada por el usuario.")
    finally:
        if hilo:
            hilo.detener()
            hilo.join(timeout=2.0)
            if hilo.grabador:
                hilo.grabador.cerrar()
        cap.release()
        if grabador_video:
            grabador_video.cerrar()
        cv2.destroyAllWindows()
        # Al terminar la sesión el semáforo vuelve a VERDE: el ESP32 mantiene
        # el último color hasta que se le mande otro, y sin sesión no hay quién.
        cliente.finalizar("VERDE")
        cliente.cerrar()

    if desde_archivo:
        reporte.duracion_forzada = n_frame / max(fps_video, 1e-6)

    reporte.cambios_estado = maquina.cambios
    ruta_csv, ruta_json = reporte.guardar(
        maquina.estado,
        extra=extra_reporte(),
    )

    resumen = reporte.resumen(maquina.estado)
    print("\n" + "=" * 52)
    print(f"Sesión de {resumen['duracion_s'] / 60:.1f} min en {resumen['salon']}")
    print(f"Estado final: {resumen['estado_final']}")
    print(f"Tiempo por estado: {resumen['porcentaje_estados']}")
    print(f"Clases de audio: {resumen['audio']['distribucion_clases']}")
    print(f"Personas promedio: {resumen['camara']['personas_promedio']}")
    print(f"Eventos: {resumen['eventos']['total']}")
    print(f"\nTodo quedó en: {reporte.carpeta}")
    for archivo in sorted(reporte.carpeta.iterdir()):
        extra = f"  ({len(list(archivo.iterdir()))} fotos)" if archivo.is_dir() else f"  ({archivo.stat().st_size / 1e6:.1f} MB)"
        print(f"  {archivo.name}{extra}")
    print("\nVer el tablero:\n    python scripts/serve_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
