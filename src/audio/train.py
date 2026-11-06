"""Entrenamiento del clasificador de audio — Etapa 3.

Entrena un RandomForest sobre los vectores de 1024 dimensiones que produce
YAMNet. YAMNet no se reentrena: aporta las características y el bosque aprende a
separar las cuatro clases del proyecto.

La división se hace por sesión de grabación, no por fragmento. Es la decisión más
importante de este archivo. Los fragmentos de una misma grabación se parecen
muchísimo entre sí —misma sala, mismo micrófono, mismo ruido de fondo—, así que
si unos van a entrenamiento y otros a prueba, el modelo reconoce la grabación en
lugar de la clase y la exactitud reportada sale falsamente alta.

Los vectores de YAMNet se guardan en un .npz para no recalcularlos en cada
entrenamiento, que es la parte lenta.

Uso:
    python -m src.audio.train
    python -m src.audio.train --recalcular
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.audio.dataset import etiquetados, leer_etiquetas, ruta_clip  # noqa: E402


# --------------------------------------------------------------- embeddings
def calcular_embeddings(filas: list[dict[str, str]], recalcular: bool = False) -> dict[str, np.ndarray]:
    """Devuelve {clip_id: vector de 1024}. Reutiliza lo ya calculado."""
    from src.audio.features import Yamnet

    ruta_cache = cfg.ruta_de("embeddings_audio")
    cache: dict[str, np.ndarray] = {}
    if ruta_cache.exists() and not recalcular:
        with np.load(ruta_cache) as datos:
            cache = {k: datos[k] for k in datos.files}
        print(f"Vectores en caché: {len(cache)}")

    faltantes = [f for f in filas if f["clip_id"] not in cache]
    if faltantes:
        import librosa

        print(f"Calculando {len(faltantes)} vectores con YAMNet...")
        yam = Yamnet().cargar()
        sr = int(cfg.cargar()["audio"]["sample_rate"])
        for i, fila in enumerate(faltantes, 1):
            ruta = ruta_clip(fila)
            if not ruta.exists():
                print(f"  falta el archivo {ruta}, se omite")
                continue
            onda, _ = librosa.load(str(ruta), sr=sr, mono=True)
            cache[fila["clip_id"]] = yam.embedding(onda)
            if i % 50 == 0 or i == len(faltantes):
                print(f"  {i}/{len(faltantes)}")

        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta_cache, **cache)
        print(f"Guardados en {ruta_cache}")

    return cache


# --------------------------------------------------------------- división
def dividir_por_sesion(
    sesiones: list[str], n_prueba: int, n_validacion: int
) -> tuple[list[str], list[str], list[str]]:
    """Reparte sesiones completas entre entrenamiento, validación y prueba."""
    unicas = sorted(set(sesiones))
    if len(unicas) < n_prueba + n_validacion + 1:
        raise ValueError(
            f"Hacen falta al menos {n_prueba + n_validacion + 1} sesiones distintas para "
            f"separar por sesión, y solo hay {len(unicas)}: {unicas}.\n"
            "Graba las cuatro clases en más días o momentos distintos."
        )
    # Las últimas sesiones grabadas se reservan para evaluar.
    prueba = unicas[-n_prueba:]
    validacion = unicas[-(n_prueba + n_validacion) : -n_prueba]
    entrenamiento = unicas[: -(n_prueba + n_validacion)]
    return entrenamiento, validacion, prueba


# --------------------------------------------------------------- reportes
def guardar_matriz(matriz: np.ndarray, clases: list[str], ruta: Path, titulo: str) -> None:
    """Guarda la matriz de confusión como PNG, normalizada por fila."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = matriz.sum(axis=1, keepdims=True)
    normal = np.divide(matriz, total, out=np.zeros_like(matriz, dtype=float), where=total > 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(normal, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(clases)), clases, rotation=35, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)

    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(
                j, i, f"{normal[i, j]:.2f}\n({int(matriz[i, j])})",
                ha="center", va="center", fontsize=9,
                color="white" if normal[i, j] > 0.55 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, label="proporción de la fila")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def tasa_confusion(matriz: np.ndarray, clases: list[str], real: str, predicha: str) -> float | None:
    """Proporción de muestras de `real` que el modelo llamó `predicha`."""
    if real not in clases or predicha not in clases:
        return None
    i, j = clases.index(real), clases.index(predicha)
    total = matriz[i].sum()
    return float(matriz[i, j] / total) if total else None


# --------------------------------------------------------------- principal
def entrenar(recalcular: bool = False) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    import joblib

    conf = cfg.cargar()
    ce = conf["audio"]["entrenamiento"]
    clases: list[str] = conf["audio"]["clases"]

    filas = etiquetados(leer_etiquetas())
    if not filas:
        raise SystemExit(
            "No hay fragmentos etiquetados todavía.\n"
            "Graba con scripts/record_session.py, corta con scripts/make_clips.py\n"
            "y revisa el estado con: python scripts/label_tool.py --resumen"
        )

    cache = calcular_embeddings(filas, recalcular)
    filas = [f for f in filas if f["clip_id"] in cache]

    X = np.stack([cache[f["clip_id"]] for f in filas])
    y = np.array([f["clase"] for f in filas])
    sesiones = [f["sesion"] for f in filas]

    print(f"\n{len(filas)} fragmentos etiquetados")
    for clase in clases:
        print(f"  {clase:<15} {int((y == clase).sum()):>5}")

    tr, val, te = dividir_por_sesion(
        sesiones, int(ce["sesiones_prueba"]), int(ce["sesiones_validacion"])
    )
    print(f"\nEntrenamiento: {tr}\nValidación:    {val}\nPrueba:        {te}")

    ses = np.array(sesiones)
    m_tr, m_val, m_te = np.isin(ses, tr), np.isin(ses, val), np.isin(ses, te)

    for nombre, mascara in (("prueba", m_te), ("validación", m_val)):
        presentes = set(y[mascara])
        if len(presentes) < 2:
            print(
                f"\nAviso: el conjunto de {nombre} solo contiene {presentes}. "
                "Graba las cuatro clases dentro de cada sesión.",
                file=sys.stderr,
            )

    modelo = RandomForestClassifier(
        n_estimators=int(ce["n_arboles"]),
        random_state=int(ce["semilla"]),
        class_weight="balanced",   # las clases no van a estar equilibradas
        n_jobs=-1,
    )
    modelo.fit(X[m_tr], y[m_tr])

    presentes = [c for c in clases if c in set(y)]
    resultados: dict[str, dict] = {}
    ruta_reports = cfg.ruta_de("reports")
    marca = datetime.now().strftime("%Y%m%d_%H%M")

    for nombre, mascara in (("validacion", m_val), ("prueba", m_te)):
        if not mascara.any():
            continue
        pred = modelo.predict(X[mascara])
        real = y[mascara]
        eb = float(balanced_accuracy_score(real, pred))
        mc = confusion_matrix(real, pred, labels=presentes)

        resultados[nombre] = {
            "exactitud_balanceada": eb,
            "n_muestras": int(mascara.sum()),
            "sesiones": sorted(set(ses[mascara])),
            "matriz_confusion": mc.tolist(),
            "clases": presentes,
            "reporte": classification_report(
                real, pred, labels=presentes, output_dict=True, zero_division=0
            ),
        }

        print(f"\n=== {nombre.upper()} ===")
        print(f"Exactitud balanceada: {eb:.3f}")
        print(classification_report(real, pred, labels=presentes, zero_division=0))

        guardar_matriz(
            mc, presentes,
            ruta_reports / f"matriz_confusion_{nombre}_{marca}.png",
            f"Clasificador de audio — {nombre}",
        )

    # ------------------------------------- validación cruzada por sesión
    # Una sola sesión de prueba da una cifra que depende de qué sesión tocó. Con
    # muchas sesiones se añade "dejar una sesión fuera" sobre todas: cada sesión
    # se predice con un modelo que no la vio. Es la estimación más estable; la de
    # prueba sigue siendo la que decide los criterios.
    unicas = sorted(set(sesiones))
    if len(unicas) >= 5:
        from sklearn.model_selection import LeaveOneGroupOut

        pred_cv = np.empty_like(y)
        for tr_i, te_i in LeaveOneGroupOut().split(X, y, ses):
            pred_cv[te_i] = RandomForestClassifier(
                n_estimators=int(ce["n_arboles"]), random_state=int(ce["semilla"]),
                class_weight="balanced", n_jobs=-1,
            ).fit(X[tr_i], y[tr_i]).predict(X[te_i])
        mc_cv = confusion_matrix(y, pred_cv, labels=presentes)
        eb_cv = float(balanced_accuracy_score(y, pred_cv))
        dm_cv = tasa_confusion(mc_cv, presentes, "voz_docente", "murmullo")
        resultados["validacion_cruzada"] = {
            "exactitud_balanceada": eb_cv, "sesiones": len(unicas),
            "matriz_confusion": mc_cv.tolist(), "clases": presentes,
            "voz_docente_confundida_con_murmullo": dm_cv,
        }
        print(f"\n=== VALIDACIÓN CRUZADA (dejar una sesión fuera, {len(unicas)} sesiones) ===")
        print(f"Exactitud balanceada: {eb_cv:.3f}   voz_docente -> murmullo: "
              f"{'n/d' if dm_cv is None else f'{dm_cv:.3f}'}")
        print(classification_report(y, pred_cv, labels=presentes, zero_division=0))
        guardar_matriz(mc_cv, presentes, ruta_reports / f"matriz_confusion_validacion_cruzada_{marca}.png",
                       "Clasificador de audio — validación cruzada por sesión")

    # ------------------------------------------------ criterios de aceptación
    prueba = resultados.get("prueba", {})
    eb = prueba.get("exactitud_balanceada")
    mc = np.array(prueba.get("matriz_confusion", [])) if prueba else np.array([])
    conf_dm = (
        tasa_confusion(mc, presentes, "voz_docente", "murmullo") if mc.size else None
    )

    minimo = float(ce["exactitud_balanceada_minima"])
    maximo_dm = float(ce["confusion_max_docente_murmullo"])
    criterios = {
        "exactitud_balanceada": {
            "valor": eb, "minimo": minimo,
            "cumple": bool(eb is not None and eb >= minimo),
        },
        "voz_docente_confundida_con_murmullo": {
            "valor": conf_dm, "maximo": maximo_dm,
            "cumple": bool(conf_dm is not None and conf_dm <= maximo_dm),
        },
    }

    print("\n=== CRITERIOS DE ACEPTACIÓN (sección 10) ===")
    for nombre, c in criterios.items():
        valor = "n/d" if c["valor"] is None else f"{c['valor']:.3f}"
        print(f"  {'CUMPLE' if c['cumple'] else 'NO CUMPLE'}  {nombre}: {valor}")

    # ------------------------------------------------------------- guardado
    ruta_modelo = cfg.ruta_de("modelo_audio")
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "clases": list(modelo.classes_),
            "entrenado": datetime.now().isoformat(timespec="seconds"),
            "sesiones_entrenamiento": tr,
            "n_muestras": int(m_tr.sum()),
        },
        ruta_modelo,
    )
    print(f"\nModelo guardado en {ruta_modelo}")

    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "n_fragmentos": len(filas),
        "por_clase": {c: int((y == c).sum()) for c in presentes},
        "sesiones": {"entrenamiento": tr, "validacion": val, "prueba": te},
        "resultados": resultados,
        "criterios": criterios,
    }
    ruta_resumen = ruta_reports / f"entrenamiento_audio_{marca}.json"
    ruta_resumen.parent.mkdir(parents=True, exist_ok=True)
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte guardado en {ruta_resumen}")

    return resumen


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Entrena el clasificador de audio")
    ap.add_argument("--recalcular", action="store_true", help="ignorar los vectores en caché")
    args = ap.parse_args()
    entrenar(recalcular=args.recalcular)
