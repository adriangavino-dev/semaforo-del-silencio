"""Verificación de todos los targets del sistema con datasets públicos.

Cada parte mide con qué porcentaje acierta un target y guarda el resultado en
reports/verificacion_targets.json. Se pueden correr por separado.

    python scripts/verificar_targets.py audio      # 4 clases de sonido (FSD50K, ya entrenado)
    python scripts/verificar_targets.py conteo     # personas detectadas (SCUT-HEAD Part A, aulas)
    python scripts/verificar_targets.py sentado    # postura sentado (SCB, alumnos en el pupitre)
    python scripts/verificar_targets.py ojos       # ojos cerrados vs abiertos según distancia (CEW)
    python scripts/verificar_targets.py kth        # de pie + correr vs caminar + boxeo (KTH Actions)
    python scripts/verificar_targets.py peleas     # peleas vs normal (Surveillance Camera Fight)
    python scripts/verificar_targets.py tabla      # imprime la tabla resumen
    python scripts/verificar_targets.py todo

Los datasets se esperan en C:/Users/Adrian/datasets/ (ver README).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.pose import Persona  # noqa: E402

DATASETS = Path(r"C:\Users\Adrian\datasets")
SALIDA = cfg.ruta_de("reports") / "verificacion_targets.json"


def guardar(parte: str, datos: dict) -> None:
    todo = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    todo[parte] = datos
    SALIDA.write_text(json.dumps(todo, indent=2, ensure_ascii=False), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def modelo_pose():
    from ultralytics import YOLO

    conf = cfg.cargar()
    return YOLO(str(cfg.RAIZ / "models" / conf["vision"]["modelo_pose"]))


def personas_de(resultado) -> list[Persona]:
    salida = []
    r = resultado
    if r.keypoints is None or not len(r.boxes):
        return salida
    confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones((len(r.boxes), 17))
    for i in range(len(r.boxes)):
        salida.append(Persona(id=i, recuadro=tuple(r.boxes.xyxy[i].tolist()),
                              keypoints=r.keypoints.xy[i].cpu().numpy(), confianzas=confs[i],
                              conf_deteccion=float(r.boxes.conf[i])))
    return salida


# ================================================================ audio
def audio() -> dict:
    ruta = sorted(cfg.ruta_de("reports").glob("entrenamiento_audio_*.json"))[-1]
    r = json.loads(ruta.read_text(encoding="utf-8"))
    cv = r.get("validacion_cruzada") or r.get("resultados", {}).get("validacion_cruzada")
    prueba = r.get("prueba") or r.get("resultados", {}).get("prueba")
    datos = {"fuente": ruta.name}
    for nombre, bloque in (("prueba", prueba), ("validacion_cruzada", cv)):
        if not bloque:
            continue
        rep = bloque.get("reporte")
        mc = np.array(bloque["matriz_confusion"])
        clases = bloque["clases"]
        recall = {c: float(mc[i, i] / max(mc[i].sum(), 1)) for i, c in enumerate(clases)}
        datos[nombre] = {"exactitud_balanceada": bloque["exactitud_balanceada"], "acierto_por_clase": recall}
        if rep is None:
            pass
    print(json.dumps(datos, indent=2, ensure_ascii=False))
    guardar("audio", datos)
    return datos


# ================================================================ conteo de personas
def conteo(max_fotos: int = 500) -> dict:
    """SCUT-HEAD Part A (prueba): cabezas marcadas a mano en fotos de cámaras de aulas."""
    import cv2

    from src.vision.features import filtrar_personas

    base = DATASETS / "scut_head" / "SCUT_HEAD_Part_A"
    ids = (base / "ImageSets" / "Main" / "test.txt").read_text().split()[:max_fotos]
    conf = cfg.cargar()
    filtro = conf["vision"]["filtro_personas"]
    yolo = modelo_pose()
    total_cabezas = encontradas = encontradas_filtro = 0
    errores_conteo, errores_filtro = [], []
    for k, i in enumerate(ids, 1):
        raiz = ET.parse(base / "Annotations" / f"{i}.xml").getroot()
        cabezas = [tuple(float(o.find("bndbox").find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                   for o in raiz.iter("object")]
        img = cv2.imread(str(base / "JPEGImages" / f"{i}.jpg"))
        if img is None or not cabezas:
            continue
        personas = personas_de(yolo.predict(img, verbose=False, classes=[0],
                                            conf=float(conf["vision"].get("conf_minima", 0.25)))[0])
        filtradas = filtrar_personas(personas, img.shape[0], float(filtro["min_alto_fraccion"]),
                                     float(filtro["min_conf_media"]))
        for lista, es_filtro in ((personas, False), (filtradas, True)):
            libres = list(range(len(cabezas)))
            emparejadas = 0
            for p in lista:
                cab = [p.keypoints[j] for j in range(5) if p.confianzas[j] > 0.3]
                punto = np.mean(cab, axis=0) if cab else np.array([(p.recuadro[0] + p.recuadro[2]) / 2,
                                                                     p.recuadro[1] + 0.1 * p.alto_recuadro])
                for j in libres:
                    x1, y1, x2, y2 = cabezas[j]
                    mx, my = 0.5 * (x2 - x1), 0.5 * (y2 - y1)
                    if x1 - mx <= punto[0] <= x2 + mx and y1 - my <= punto[1] <= y2 + my:
                        libres.remove(j)
                        emparejadas += 1
                        break
            if es_filtro:
                encontradas_filtro += emparejadas
                errores_filtro.append(abs(len(lista) - len(cabezas)) / len(cabezas))
            else:
                encontradas += emparejadas
                errores_conteo.append(abs(len(lista) - len(cabezas)) / len(cabezas))
        total_cabezas += len(cabezas)
        if k % 100 == 0:
            print(f"  {k}/{len(ids)} fotos", flush=True)
    datos = {
        "fotos": len(ids), "personas_marcadas": total_cabezas,
        "detectadas_yolo": encontradas / total_cabezas,
        "detectadas_tras_filtro_del_proyecto": encontradas_filtro / total_cabezas,
        "error_medio_del_conteo_yolo": float(np.mean(errores_conteo)),
        "error_medio_del_conteo_tras_filtro": float(np.mean(errores_filtro)),
    }
    print(f"Personas marcadas: {total_cabezas} en {len(ids)} fotos de aulas")
    print(f"  detectadas por YOLOv8n-pose: {pct(datos['detectadas_yolo'])} "
          f"(error medio del conteo {pct(datos['error_medio_del_conteo_yolo'])})")
    print(f"  tras el filtro del proyecto (alto >= {filtro['min_alto_fraccion']} del cuadro): "
          f"{pct(datos['detectadas_tras_filtro_del_proyecto'])}")
    guardar("conteo", datos)
    return datos


# ================================================================ sentado
def sentado(max_fotos: int = 800) -> dict:
    """SCB BowTurnHead: todos los alumnos etiquetados están sentados en su pupitre."""
    import cv2

    from src.vision.features import LineaBase, caracteristicas, postura_geometrica

    conf = cfg.cargar()
    cp = conf["vision"]["postura"]
    yolo = modelo_pose()
    fotos = sorted((DATASETS / "scb_bowturnhead").rglob("images/**/*.jpg"))
    random.Random(1).shuffle(fotos)
    fotos = fotos[:max_fotos]
    base = LineaBase()
    n = sent = 0
    for k, foto in enumerate(fotos, 1):
        etiquetas = Path(str(foto).replace("images", "labels")).with_suffix(".txt")
        img = cv2.imread(str(foto))
        if img is None or not etiquetas.exists():
            continue
        alto, ancho = img.shape[:2]
        cajas = []
        for linea in etiquetas.read_text().split("\n"):
            partes = linea.split()
            if len(partes) == 5:
                _, cx, cy, w, h = map(float, partes)
                cajas.append(((cx - w / 2) * ancho, (cy - h / 2) * alto, (cx + w / 2) * ancho, (cy + h / 2) * alto))
        personas = personas_de(yolo.predict(img, verbose=False, classes=[0], conf=0.25)[0])
        for caja in cajas:
            cx, cy = (caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2
            p = next((p for p in personas if p.recuadro[0] <= cx <= p.recuadro[2] and p.recuadro[1] <= cy <= p.recuadro[3]), None)
            if p is None:
                continue
            rasgos = caracteristicas(p, personas, base, {}, 25.0, 8.0)
            n += 1
            sent += postura_geometrica(p, rasgos, float(cp["aspecto_de_pie"]), float(cp["angulo_rodilla_de_pie"])) == "sentado"
        if k % 200 == 0:
            print(f"  {k}/{len(fotos)} fotos", flush=True)
    datos = {"fotos": len(fotos), "alumnos_sentados_medidos": n, "clasificados_sentado": sent / max(n, 1)}
    print(f"Alumnos sentados (SCB): {n} | clasificados como sentado: {pct(datos['clasificados_sentado'])}")
    guardar("sentado", datos)
    return datos


# ================================================================ ojos
def ojos() -> dict:
    """CEW, caras alineadas: abiertos/cerrados, achicando la cara para simular distancia.

    Validación cruzada por cara: el bosque nunca ve recortes de la cara que evalúa.
    """
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    from src.vision.ojos import _cargar_cew, recortes_ojos, rasgos_ojo

    X, y, g = _cargar_cew()
    caras = []
    raiz = DATASETS / "cew" / "dataset_B_FacialImages"
    for archivo, carpeta, clase in (("EyeCoordinatesInfo_OpenFace.txt", "OpenFace", "abierto"),
                                    ("EyeCoordinatesInfo_ClosedFace.txt", "ClosedFace", "cerrado")):
        for linea in (raiz / archivo).read_text().split("\n"):
            partes = linea.split()
            if len(partes) == 5:
                nombre = partes[0]
                grupo = re.sub(r"\.jpg$", "", nombre) if clase == "abierto" else nombre
                caras.append((raiz / carpeta / nombre, clase, grupo, np.array(list(map(float, partes[1:])))))
    grupos_caras = np.array([c[2] for c in caras])
    distancias = [40, 25, 15, 10]
    aciertos = {d: [] for d in distancias}
    sin_dato = {d: 0 for d in distancias}
    todos_grupos = np.unique(np.concatenate([g, grupos_caras]))
    pliegue = {grp: i % 5 for i, grp in enumerate(todos_grupos)}
    for f in range(5):
        tr = np.array([pliegue[x] != f for x in g])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(X[tr], y[tr])
        i_cerr = list(rf.classes_).index("cerrado")
        for ruta, clase, grupo, ojos_xy in (c for c in caras if pliegue[c[2]] == f):
            img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            d0 = float(np.hypot(ojos_xy[2] - ojos_xy[0], ojos_xy[3] - ojos_xy[1]))
            for d in distancias:
                esc = d / d0
                peq = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
                kp = np.zeros((17, 2)); cf = np.zeros(17)
                kp[1], kp[2] = ojos_xy[:2] * esc, ojos_xy[2:] * esc
                cf[1] = cf[2] = 1.0
                p = Persona(id=0, recuadro=(0, 0, peq.shape[1], peq.shape[0]), keypoints=kp, confianzas=cf)
                recortes = recortes_ojos(peq, p)
                if not recortes:
                    sin_dato[d] += 1
                    continue
                prob = rf.predict_proba(np.stack([rasgos_ojo(r) for r in recortes]))[:, i_cerr].mean()
                aciertos[d].append((("cerrado" if prob >= 0.5 else "abierto") == clase, clase))
    datos = {}
    print("Ojos abiertos/cerrados (CEW, validación cruzada por cara), según distancia entre ojos en la imagen:")
    for d in distancias:
        a = aciertos[d]
        ab = [ok for ok, c in a if c == "abierto"]; ce = [ok for ok, c in a if c == "cerrado"]
        eb = 0.5 * (np.mean(ab) + np.mean(ce))
        datos[f"{d}px"] = {"exactitud_balanceada": float(eb), "abiertos_bien": float(np.mean(ab)),
                           "cerrados_bien": float(np.mean(ce)), "caras": len(a)}
        print(f"  {d:>3} px entre ojos: {pct(eb)}  (abiertos {pct(np.mean(ab))}, cerrados {pct(np.mean(ce))}, {len(a)} caras)")
    guardar("ojos", datos)
    return datos


# ================================================================ video (KTH y peleas)
def procesar_video(ruta: Path, detector, conf: dict, paso: int = 1) -> dict:
    """Postura y regla de movimiento sobre un video, cuadro a cuadro.

    Usa el mismo DetectorPose (con seguimiento ByteTrack) que run_session.py: la
    velocidad de muñecas solo se calcula para personas seguidas por ID.
    """
    import cv2

    from src.vision.events import DetectorEventos
    from src.vision.features import LineaBase, caracteristicas, filtrar_personas, muñecas_previas, postura_geometrica

    cv = conf["vision"]
    cp = cv["postura"]
    cap = cv2.VideoCapture(str(ruta))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / paso
    eventos = DetectorEventos(conf, fps)
    base = LineaBase()
    previas: dict[int, np.ndarray] = {}
    n_cuadro = ventanas = anomalas = 0
    confirmado = False
    de_pie = medidas = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_cuadro += 1
        if (n_cuadro - 1) % paso:
            continue
        personas = filtrar_personas(detector.detectar(frame), frame.shape[0], float(cv["filtro_personas"]["min_alto_fraccion"]),
                                    float(cv["filtro_personas"]["min_conf_media"]))
        rasgos = [caracteristicas(p, personas, base, previas, fps, float(cv["evento"]["max_velocidad_fisica_por_s"]))
                  for p in personas]
        for p, ra in zip(personas, rasgos):
            medidas += 1
            de_pie += postura_geometrica(p, ra, float(cp["aspecto_de_pie"]), float(cp["angulo_rodilla_de_pie"])) == "de_pie"
        previas = muñecas_previas(personas)
        d = eventos.agregar_cuadro(rasgos)
        if eventos._desde_ultima == 0 and len(eventos._cuadros) == eventos.tam_ventana:   # se evaluó una ventana
            ventanas += 1
            anomalas += d["evento"] == "movimiento_anomalo"
        confirmado = confirmado or d["confirmado"]
    cap.release()
    detector.reiniciar_seguimiento()   # cada video empieza con IDs nuevos
    return {"ventanas": ventanas, "ventanas_anomalas": anomalas, "evento_confirmado": confirmado,
            "fraccion_de_pie": de_pie / medidas if medidas else None, "medidas_postura": medidas}


def detector_sesion():
    from src.vision.pose import DetectorPose

    return DetectorPose(cfg.cargar()).cargar()


def kth(por_accion: int = 24) -> dict:
    conf = cfg.cargar()
    yolo = detector_sesion()
    datos = {}
    for accion in ("running", "jogging", "walking", "boxing"):
        videos = sorted((DATASETS / "kth" / accion).glob("*.avi"))
        videos = videos[:: max(1, len(videos) // por_accion)][:por_accion]
        res = [procesar_video(v, yolo, conf, paso=2) for v in videos]
        con_ventanas = [r for r in res if r["ventanas"]]
        alguna = float(np.mean([r["ventanas_anomalas"] > 0 for r in con_ventanas])) if con_ventanas else 0.0
        evento = float(np.mean([r["evento_confirmado"] for r in res]))
        fr = [r["fraccion_de_pie"] for r in res if r["fraccion_de_pie"] is not None]
        datos[accion] = {"videos": len(res), "con_alguna_ventana_anomala": alguna,
                         "con_evento_confirmado": evento, "de_pie": float(np.mean(fr)) if fr else None}
        print(f"  {accion:<8} {len(res)} videos | alguna ventana anómala {pct(alguna)} | evento confirmado {pct(evento)}"
              f" | personas clasificadas de pie {pct(datos[accion]['de_pie'] or 0)}", flush=True)
    guardar("kth", datos)
    return datos


def peleas() -> dict:
    conf = cfg.cargar()
    yolo = detector_sesion()
    datos = {}
    # Los clips ya están copiados en el proyecto (scripts/importar_videos_peleas.py).
    carpeta = cfg.ruta_de("data_video_raw") / "surv"
    for patron, clase in (("fi*.mp4", "pelea"), ("nofi*.mp4", "normal")):
        videos = sorted(carpeta.glob(patron))
        res = [procesar_video(v, yolo, conf) for v in videos]
        con_ventanas = [r for r in res if r["ventanas"]]
        alguna = float(np.mean([r["ventanas_anomalas"] > 0 for r in con_ventanas])) if con_ventanas else 0.0
        datos[clase] = {"videos": len(res), "con_ventanas": len(con_ventanas), "con_alguna_ventana_anomala": alguna}
        print(f"  {clase:<7} {len(res)} clips ({len(con_ventanas)} con personas) | marcados como movimiento anómalo {pct(alguna)}",
              flush=True)
    guardar("peleas", datos)
    return datos


# ================================================================ tabla
def tabla() -> None:
    t = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    filas = []
    if "audio" in t:
        a = t["audio"]
        cv = a.get("validacion_cruzada", {})
        filas.append(("Sonido: 4 clases", "FSD50K", pct(cv.get("exactitud_balanceada", 0)) + " (exactitud balanceada)"))
        for c, v in cv.get("acierto_por_clase", {}).items():
            filas.append((f"  {c}", "FSD50K", pct(v)))
    if "conteo" in t:
        c = t["conteo"]
        filas.append(("Personas detectadas", "SCUT-HEAD (aulas)", pct(c["detectadas_yolo"])))
    if "sentado" in t:
        filas.append(("Postura: sentado", "SCB (aulas)", pct(t["sentado"]["clasificados_sentado"])))
    if "kth" in t:
        k = t["kth"]
        fr = [v["de_pie"] for v in k.values() if v.get("de_pie") is not None]
        if fr:
            filas.append(("Postura: de pie", "KTH", pct(float(np.mean(fr)))))
        # Lo que cuenta es el evento confirmado: es lo que cambia el semáforo.
        for accion, nombre in (("running", "Correr detectado"), ("jogging", "Trotar detectado"),
                               ("boxing", "Golpes (boxeo) detectados")):
            if accion in k:
                filas.append((nombre, "KTH", pct(k[accion]["con_evento_confirmado"])))
        if "walking" in k:
            filas.append(("Caminar SIN falsa alarma", "KTH", pct(1 - k["walking"]["con_evento_confirmado"])))
    if "peleas" in t:
        p = t["peleas"]
        filas.append(("Pelea detectada (clip de 2 s)", "Fight CCTV", pct(p["pelea"]["con_alguna_ventana_anomala"])))
        filas.append(("Normal SIN falsa alarma (clip 2 s)", "Fight CCTV", pct(1 - p["normal"]["con_alguna_ventana_anomala"])))
    if "ojos" in t:
        for d, v in t["ojos"].items():
            filas.append((f"Ojos cerrados ({d} entre ojos)", "CEW", pct(v["exactitud_balanceada"])))
    if "dormido" in t or True:
        scb = cfg.ruta_de("reports") / "evaluacion_cabeza_scb.json"
        if scb.exists():
            s = json.loads(scb.read_text(encoding="utf-8"))
            erg = s["clases"].get("TurnHead", {}).get("fraccion_bajo_umbral")
            if erg is not None:
                filas.append(("Dormido: erguido NO marcado", "SCB", pct(1 - erg)))
    ancho = max(len(f[0]) for f in filas) if filas else 10
    print(f"\n{'Target':<{ancho}}  {'Dataset':<18}  Resultado")
    print("-" * (ancho + 40))
    for nombre, ds, valor in filas:
        print(f"{nombre:<{ancho}}  {ds:<18}  {valor}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verifica cada target con datasets públicos")
    ap.add_argument("parte", choices=["audio", "conteo", "sentado", "ojos", "kth", "peleas", "tabla", "todo"])
    a = ap.parse_args()
    partes = ["audio", "conteo", "sentado", "ojos", "kth", "peleas"] if a.parte == "todo" else [a.parte]
    for parte in partes:
        if parte != "tabla":
            print(f"\n===== {parte} =====", flush=True)
            globals()[parte]()
    tabla()
