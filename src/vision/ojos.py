"""Ojos abiertos / cerrados, con un RandomForest sobre recortes del ojo.

Entrenamiento: CEW, "Closed Eyes in the Wild" (Song, Tan, Liu y Chen, Pattern
Recognition 2014; NUAA; solo investigación, no redistribuir). 4 854 recortes de
ojo de 24x24: 2 388 cerrados y 2 466 abiertos, de 2 423 caras.

En vivo: YOLOv8-Pose ya da el centro de cada ojo. Se recorta un cuadrado de
lado ~0,7 x la distancia entre los ojos (el mismo encuadre que CEW), se pasa a
24x24 en gris y el bosque decide. Rasgos: HOG (bordes del párpado y la
pupila) + los píxeles normalizados.

Límite físico declarado: con la cámara al frente del aula, el ojo de un alumno
del fondo mide pocos píxeles. Si la distancia entre ojos es menor que
`min_px_entre_ojos`, no se decide ("sin dato") en vez de inventar.

Uso:
    python -m src.vision.ojos --entrenar     # entrena con CEW y mide por validación cruzada
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

LADO = 24
OJO_IZQ, OJO_DER = 1, 2   # COCO
CEW = Path(r"C:\Users\Adrian\datasets\cew")
_HOG = None


def _hog():
    global _HOG
    if _HOG is None:
        import cv2

        _HOG = cv2.HOGDescriptor((LADO, LADO), (8, 8), (4, 4), (4, 4), 9)
    return _HOG


def rasgos_ojo(gris24: np.ndarray) -> np.ndarray:
    """HOG + píxeles normalizados (contraste estirado) de un recorte 24x24 en gris."""
    import cv2

    g = cv2.equalizeHist(gris24.astype(np.uint8))
    hog = _hog().compute(g).ravel()
    pix = cv2.resize(g, (12, 12), interpolation=cv2.INTER_AREA).ravel() / 255.0
    return np.concatenate([hog, pix]).astype(np.float32)


def recortes_ojos(imagen: np.ndarray, persona, conf_min: float = 0.5, min_px: float = 0.0):
    """Recortes 24x24 en gris de los dos ojos de una persona, o [] si no se puede."""
    import cv2

    c = persona.confianzas
    if c[OJO_IZQ] < conf_min or c[OJO_DER] < conf_min:
        return []
    izq, der = persona.keypoints[OJO_IZQ], persona.keypoints[OJO_DER]
    d = float(np.linalg.norm(izq - der))
    if d < max(min_px, 2.0):
        return []
    lado = 0.7 * d
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    alto, ancho = gris.shape
    salida = []
    for centro in (izq, der):
        x1, y1 = int(round(centro[0] - lado / 2)), int(round(centro[1] - lado / 2))
        x2, y2 = int(round(centro[0] + lado / 2)), int(round(centro[1] + lado / 2))
        if x1 < 0 or y1 < 0 or x2 > ancho or y2 > alto or x2 - x1 < 2:
            continue
        salida.append(cv2.resize(gris[y1:y2, x1:x2], (LADO, LADO), interpolation=cv2.INTER_AREA))
    return salida


class ClasificadorOjos:
    def __init__(self, conf: dict | None = None) -> None:
        co = (conf or cfg.cargar())["vision"].get("ojos", {})
        self.habilitado = bool(co.get("habilitado", True))
        self.min_px = float(co.get("min_px_entre_ojos", 20))
        self.umbral = float(co.get("umbral_cerrado", 0.5))
        self._modelo = None

    def cargar(self) -> "ClasificadorOjos":
        import joblib

        ruta = cfg.ruta_de("modelo_ojos")
        if ruta.exists():
            self._modelo = joblib.load(ruta)["modelo"]
            self._modelo.n_jobs = 1     # pocas muestras por llamada: repartir en hilos cuesta más
        return self

    def probabilidades(self, imagen: np.ndarray, personas: list) -> dict[int, float]:
        """{id: probabilidad de ojos cerrados} de todas las personas con la cara visible, en un solo lote."""
        if not self.disponible:
            return {}
        filas, ids = [], []
        for p in personas:
            for r in recortes_ojos(imagen, p, min_px=self.min_px):
                filas.append(rasgos_ojo(r)); ids.append(p.id)
        if not filas:
            return {}
        i = list(self._modelo.classes_).index("cerrado")
        probs = self._modelo.predict_proba(np.stack(filas))[:, i]
        salida: dict[int, list[float]] = {}
        for pid, pr in zip(ids, probs):
            salida.setdefault(pid, []).append(float(pr))
        return {pid: float(np.mean(v)) for pid, v in salida.items()}

    @property
    def disponible(self) -> bool:
        return self._modelo is not None and self.habilitado

    def probabilidad_cerrados(self, imagen: np.ndarray, persona) -> float | None:
        """Probabilidad de ojos cerrados (media de los dos), o None si no se puede ver."""
        if not self.disponible:
            return None
        recortes = recortes_ojos(imagen, persona, min_px=self.min_px)
        if not recortes:
            return None
        X = np.stack([rasgos_ojo(r) for r in recortes])
        i = list(self._modelo.classes_).index("cerrado")
        return float(self._modelo.predict_proba(X)[:, i].mean())


# ------------------------------------------------------------------ entrenamiento
def _cargar_cew() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    X, y, grupos = [], [], []
    for carpeta, clase in (("closedLeftEyes", "cerrado"), ("closedRightEyes", "cerrado"),
                           ("openLeftEyes", "abierto"), ("openRightEyes", "abierto")):
        for f in sorted((CEW / "dataset_B_Eye_Images" / carpeta).glob("*.jpg")):
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (LADO, LADO))
            X.append(rasgos_ojo(g)); y.append(clase)
            # Mismo grupo para el ojo izquierdo y el derecho de la misma cara.
            grupos.append(re.sub(r"_[LR]\.jpg$", "", f.name))
    return np.array(X), np.array(y), np.array(grupos)


def entrenar() -> dict:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold

    X, y, g = _cargar_cew()
    if not len(X):
        raise SystemExit(f"No se encontró CEW en {CEW}")
    print(f"{len(X)} recortes: {int((y == 'cerrado').sum())} cerrados, {int((y == 'abierto').sum())} abiertos, "
          f"{len(set(g))} caras")

    def modelo():
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                      random_state=42, n_jobs=-1)

    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = modelo().fit(X[tr], y[tr]).predict(X[te])
    eb = float(balanced_accuracy_score(y, pred))
    mc = confusion_matrix(y, pred, labels=["abierto", "cerrado"])
    print(f"Validación cruzada por cara (5 particiones): exactitud balanceada {eb:.3f}")
    print(f"  abiertos bien {mc[0, 0] / mc[0].sum():.3f} | cerrados bien {mc[1, 1] / mc[1].sum():.3f}")

    final = modelo().fit(X, y)
    ruta = cfg.ruta_de("modelo_ojos")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "entrenado": datetime.now().isoformat(timespec="seconds")}, ruta)
    resumen = {"n": int(len(X)), "exactitud_balanceada_cv": eb, "matriz_confusion": mc.tolist(),
               "clases": ["abierto", "cerrado"]}
    (cfg.ruta_de("reports") / "entrenamiento_ojos.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo guardado en {ruta}")
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrenar", action="store_true")
    if ap.parse_args().entrenar:
        entrenar()
    else:
        ap.print_help()
