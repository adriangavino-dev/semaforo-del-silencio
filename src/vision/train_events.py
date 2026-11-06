"""Entrenamiento del clasificador de eventos — sección 6.4.

Cada video registrado en data/video/labels.csv se recorre con YOLOv8-Pose, se
calculan los rasgos por persona y se resumen en ventanas de ~2 s (media, máximo
y desviación estándar de cada rasgo). Todas las ventanas de un video heredan su
clase: `normal` o `movimiento_anomalo`. Sobre esos vectores se entrena un
RandomForest.

Los vectores se guardan en caché (data/video/ventanas.npz) porque recorrer los
videos con el detector de pose es la parte lenta.

Evaluación honesta: nunca se evalúa un video con otro de la misma sesión en el
entrenamiento.
  - Pocas sesiones (< 10): "dejar un video fuera": se entrena sin cada video y
    se predice sobre él.
  - Muchas sesiones (p. ej. el dataset de vigilancia, 46 videos de origen):
    validación cruzada de 10 particiones agrupada por sesión (GroupKFold). Los
    clips cortados del mismo video de YouTube son casi idénticos; si unos fueran
    a entrenamiento y otros a prueba, el acierto saldría inflado.

Uso:
    python -m src.vision.train_events
    python -m src.vision.train_events --recalcular
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.vision.events import DetectorEventos, nombres_vector, vector_de_ventana  # noqa: E402
from src.vision.features import (  # noqa: E402
    LineaBase, caracteristicas, filtrar_personas, muñecas_previas,
)
from src.vision.pose import DetectorPose  # noqa: E402


def leer_registro() -> list[dict[str, str]]:
    ruta = cfg.ruta_de("data_video_labels")
    if not ruta.exists():
        raise SystemExit(
            f"No existe {ruta}.\n"
            "Registra videos con: python scripts/record_video.py --registrar ARCHIVO --clase CLASE"
        )
    with ruta.open("r", encoding="utf-8", newline="") as fh:
        filas = [f for f in csv.DictReader(fh) if f.get("clase")]
    if not filas:
        raise SystemExit("labels.csv no tiene videos con clase asignada.")
    return filas


def ventanas_de_video(ruta: Path, conf: dict, detector: DetectorPose) -> tuple[np.ndarray, float]:
    """Recorre un video y devuelve (matriz de ventanas, fps)."""
    import cv2

    cv = conf["vision"]
    filtro = cv["filtro_personas"]
    max_fisica = float(cv["evento"]["max_velocidad_fisica_por_s"])

    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det_eventos = DetectorEventos(conf, fps)
    detector.reiniciar_seguimiento()
    base = LineaBase.cargar()

    previas: dict[int, np.ndarray] = {}
    vectores: list[np.ndarray] = []
    ventana: list[list[dict]] = []
    desde_ultima = 0
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        personas = filtrar_personas(
            detector.detectar(frame), frame.shape[0],
            float(filtro["min_alto_fraccion"]), float(filtro["min_conf_media"]),
        )
        rasgos = [caracteristicas(p, personas, base, previas, fps, max_fisica) for p in personas]
        previas = muñecas_previas(personas)

        ventana.append(rasgos)
        if len(ventana) > det_eventos.tam_ventana:
            ventana.pop(0)
        desde_ultima += 1
        if len(ventana) == det_eventos.tam_ventana and desde_ultima >= det_eventos.paso:
            desde_ultima = 0
            vectores.append(vector_de_ventana(ventana))

    cap.release()
    matriz = np.array(vectores) if vectores else np.zeros((0, len(nombres_vector())))
    print(f"  {ruta.name}: {n} cuadros, {fps:.1f} FPS, {len(matriz)} ventanas")
    return matriz, fps


def cargar_ventanas(filas: list[dict[str, str]], recalcular: bool) -> dict[str, np.ndarray]:
    ruta_cache = cfg.ruta_de("ventanas_video")
    cache: dict[str, np.ndarray] = {}
    if ruta_cache.exists() and not recalcular:
        with np.load(ruta_cache) as d:
            cache = {k: d[k] for k in d.files}
        print(f"Ventanas en caché: {len(cache)} videos")

    faltan = [f for f in filas if f["archivo"] not in cache]
    if faltan:
        conf = cfg.cargar()
        print(f"Recorriendo {len(faltan)} videos con YOLOv8-Pose...")
        detector = DetectorPose(conf).cargar()
        carpeta = cfg.ruta_de("data_video_raw")
        for f in faltan:
            ruta = carpeta / f["archivo"]
            if not ruta.exists():
                print(f"  falta {ruta}, se omite")
                continue
            cache[f["archivo"]], _ = ventanas_de_video(ruta, conf, detector)
        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta_cache, **cache)
        print(f"Guardadas en {ruta_cache}")
    return cache


def resumen_por_video(filas: list[dict[str, str]], cache: dict[str, np.ndarray]) -> None:
    """Muestra las estadísticas que más separan las clases, para entender los datos."""
    nombres = nombres_vector()
    i_media = nombres.index("velocidad_munecas_media")
    i_max = nombres.index("velocidad_munecas_max")
    i_np = nombres.index("n_personas_media")
    print("\nVelocidad de muñecas por video (alturas de recuadro / s)")
    print(f"  {'video':<28} {'clase':<20} {'vent':>5} {'vel media':>10} {'vel max':>9} {'personas':>9}")
    for f in filas:
        m = cache.get(f["archivo"])
        if m is None or not len(m):
            continue
        print(f"  {f['archivo']:<28} {f['clase']:<20} {len(m):>5} "
              f"{m[:, i_media].mean():>10.2f} {m[:, i_max].mean():>9.2f} {m[:, i_np].mean():>9.2f}")


def entrenar(recalcular: bool = False) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    import joblib

    conf = cfg.cargar()
    clases = list(conf["vision"]["clases_evento"])
    filas = leer_registro()
    cache = cargar_ventanas(filas, recalcular)
    filas = [f for f in filas if f["archivo"] in cache and len(cache[f["archivo"]])]
    resumen_por_video(filas, cache)

    # Un video sin personas no enseña nada sobre el movimiento: solo enseñaría
    # "si hay alguien es anómalo". Se excluye y se avisa. Este caso ocurrió de
    # verdad con una grabación de webcam apuntando a una escena vacía.
    i_np = nombres_vector().index("n_personas_media")
    vacios = [f for f in filas if cache[f["archivo"]][:, i_np].mean() < 0.3]
    for f in vacios:
        print(f"\n  EXCLUIDO {f['archivo']}: casi no se detectan personas "
              f"({cache[f['archivo']][:, i_np].mean():.2f} por ventana). Vuelve a grabarlo con gente en cuadro.")
    filas = [f for f in filas if f not in vacios]
    if not filas:
        raise SystemExit("No queda ningún video utilizable.")

    X = np.vstack([cache[f["archivo"]] for f in filas])
    y = np.concatenate([[f["clase"]] * len(cache[f["archivo"]]) for f in filas])
    video = np.concatenate([[f["archivo"]] * len(cache[f["archivo"]]) for f in filas])
    sesion = np.concatenate([[f["sesion"]] * len(cache[f["archivo"]]) for f in filas])

    presentes = sorted(set(y))
    print(f"\n{len(X)} ventanas de {len(filas)} videos")
    for c in clases:
        print(f"  {c:<20} {int((y == c).sum()):>5} ventanas")
    if len(presentes) < 2:
        raise SystemExit(
            "\nHace falta al menos un video de cada clase. Graba actividad normal con:\n"
            "  python scripts/record_video.py --nombre aula_normal_01 --clase normal --segundos 120"
        )

    ce = conf["audio"]["entrenamiento"]  # se reutilizan árboles y semilla
    def nuevo_modelo():
        return RandomForestClassifier(
            n_estimators=int(ce["n_arboles"]), random_state=int(ce["semilla"]),
            class_weight="balanced", min_samples_leaf=2, n_jobs=-1,
        )

    por_video = {}
    aciertos_total, total = 0, 0
    validacion_cruzada = None
    n_sesiones = len(set(sesion))
    if n_sesiones >= 10:
        # ------------------------------------------ validación cruzada agrupada
        from sklearn.model_selection import GroupKFold

        k = min(10, n_sesiones)
        print(f"\n=== VALIDACIÓN CRUZADA AGRUPADA POR SESIÓN ({k} particiones, {n_sesiones} sesiones) ===")
        pred_oof = np.empty_like(y)
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y, groups=sesion), 1):
            pred_oof[te] = nuevo_modelo().fit(X[tr], y[tr]).predict(X[te])
        eb_oof = float(balanced_accuracy_score(y, pred_oof))
        mc_oof = confusion_matrix(y, pred_oof, labels=presentes)
        aciertos_total, total = int((pred_oof == y).sum()), len(y)
        for f in filas:
            m = video == f["archivo"]
            por_video[f["archivo"]] = {"clase": f["clase"], "n": int(m.sum()),
                                       "acierto": float((pred_oof[m] == y[m]).mean())}
        print(f"  exactitud balanceada fuera de muestra: {eb_oof:.3f}")
        print(f"  matriz de confusión (filas = real, columnas = predicho) {presentes}:")
        for c, fila in zip(presentes, mc_oof):
            print(f"    {c:<20} {fila.tolist()}")
        for c in presentes:
            vids = [v for v in por_video.values() if v["clase"] == c]
            bien = sum(v["acierto"] >= 0.5 for v in vids)
            print(f"  {c:<20} {bien}/{len(vids)} videos clasificados bien (mayoría de sus ventanas)")
        propios = [f for f in filas if "/" not in f["archivo"]]
        for f in propios:
            v = por_video[f["archivo"]]
            print(f"  video propio {f['archivo']:<26} {f['clase']:<20} {v['acierto'] * 100:5.1f}% de {v['n']} ventanas")
        validacion_cruzada = {"particiones": k, "sesiones": n_sesiones, "exactitud_balanceada": eb_oof,
                              "matriz_confusion": mc_oof.tolist()}

    # ---------------------------------------------- dejar un video fuera
    if validacion_cruzada is None:
        print("\n=== DEJAR UN VIDEO FUERA ===")
        print("(se entrena sin ese video y se predice sobre él)")
    for f in (filas if validacion_cruzada is None else []):
        m_test = video == f["archivo"]
        if len(set(y[~m_test])) < 2:
            print(f"  {f['archivo']:<28} sin evaluar: al quitarlo queda una sola clase")
            continue
        modelo = nuevo_modelo().fit(X[~m_test], y[~m_test])
        pred = modelo.predict(X[m_test])
        acierto = float((pred == y[m_test]).mean())
        por_video[f["archivo"]] = {"clase": f["clase"], "n": int(m_test.sum()), "acierto": acierto}
        aciertos_total += int((pred == y[m_test]).sum())
        total += int(m_test.sum())
        marca = "OK " if acierto >= 0.7 else "?? "
        print(f"  {marca}{f['archivo']:<28} {f['clase']:<20} {acierto * 100:5.1f}% de sus {m_test.sum()} ventanas")
    if total:
        print(f"  acierto global fuera de muestra: {100 * aciertos_total / total:.1f}%")

    # -------------------------------------------------- modelo final
    modelo = nuevo_modelo().fit(X, y)
    pred = modelo.predict(X)
    mc = confusion_matrix(y, pred, labels=presentes)
    eb = float(balanced_accuracy_score(y, pred))
    print(f"\nModelo final (sobre todo, solo referencia): exactitud balanceada {eb:.3f}")

    importancias = sorted(zip(modelo.feature_importances_, nombres_vector()), reverse=True)[:6]
    print("Rasgos más usados:")
    for imp, nombre in importancias:
        print(f"  {imp:.3f}  {nombre}")

    # Trampa real que ocurrió en este proyecto: si los videos normales y los
    # anómalos se grabaron en escenas distintas (aula lejana con mucha gente vs.
    # una persona en primer plano), el modelo aprende a distinguir la ESCENA
    # —cuántas personas hay, a qué distancia— y no el MOVIMIENTO. Sale con 100 %
    # y no sirve para nada. La señal es que los rasgos de escena dominan y los de
    # velocidad no aparecen.
    escena = ("n_personas", "distancia_minima", "relacion_torso_altura")
    peso_escena = sum(i for i, n in zip(modelo.feature_importances_, nombres_vector()) if n.startswith(escena))
    peso_movimiento = sum(i for i, n in zip(modelo.feature_importances_, nombres_vector()) if n.startswith("velocidad"))
    if peso_escena > 0.5 and peso_movimiento < 0.2:
        print(
            "\n  AVISO: el modelo se apoya en rasgos de ESCENA (personas, distancias, proporciones)"
            f" [{peso_escena:.0%}] y casi nada en la velocidad [{peso_movimiento:.0%}].\n"
            "  Probablemente los videos normales y anómalos se grabaron en sitios o encuadres\n"
            "  distintos, y el modelo aprendió a reconocer el sitio. Graba ambas clases con la\n"
            "  MISMA cámara, en el MISMO lugar y con la MISMA cantidad de gente, y vuelve a entrenar.",
            file=sys.stderr,
        )

    ruta_modelo = cfg.ruta_de("modelo_evento")
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "modelo": modelo, "clases": list(modelo.classes_),
        "columnas": nombres_vector(),
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "videos": [f["archivo"] for f in filas],
    }, ruta_modelo)
    print(f"\nModelo guardado en {ruta_modelo}")

    marca = datetime.now().strftime("%Y%m%d_%H%M")
    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "n_ventanas": int(len(X)),
        "por_clase": {c: int((y == c).sum()) for c in presentes},
        "videos": [{"archivo": f["archivo"], "clase": f["clase"], "sesion": f["sesion"]} for f in filas],
        "dejar_un_video_fuera": por_video,
        "acierto_fuera_de_muestra": (aciertos_total / total) if total else None,
        "validacion_cruzada_agrupada": validacion_cruzada,
        "matriz_confusion_entrenamiento": mc.tolist(),
        "clases": presentes,
        "importancias": [{"rasgo": n, "importancia": float(i)} for i, n in importancias],
    }
    ruta_reporte = cfg.ruta_de("reports") / f"entrenamiento_eventos_{marca}.json"
    ruta_reporte.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte guardado en {ruta_reporte}")
    return resumen


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Entrena el clasificador de eventos")
    ap.add_argument("--recalcular", action="store_true")
    args = ap.parse_args()
    entrenar(args.recalcular)
