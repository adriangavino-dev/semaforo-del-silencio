"""Plano 2D, vista explosionada y STL a partir de los 3MF de la carcasa.

Trabaja sobre la geometría exportada del CAD del equipo: no modela nada. Las cotas
salen de la malla; las tolerancias se definen en el CAD. Los 3MF van en
entrega_semana13/cad/ con los nombres "caja final.3mf" y "Tapa final.3mf".

    python docs/entrega/planos.py
"""
from __future__ import annotations

import re
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from diagramas import GRIS, LINEA, NAVY, TINTA  # noqa: E402

RAIZ = AQUI.parent.parent
CAD = RAIZ / "entrega_semana13" / "cad"
PLANOS = RAIZ / "entrega_semana13" / "planos"
ANGULO_ARISTA = 20.0


def leer_3mf(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    x = zipfile.ZipFile(ruta).read("3D/3dmodel.model").decode("utf-8")
    v = np.array(re.findall(r'<vertex x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"', x), dtype=float)
    t = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', x), dtype=int)
    return v, t


def guardar_stl(ruta: Path, v: np.ndarray, t: np.ndarray) -> None:
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with ruta.open("wb") as fh:
        fh.write(b"Semaforo del Silencio".ljust(80, b" "))
        fh.write(struct.pack("<I", len(t)))
        for ni, ti in zip(n, tri):
            fh.write(struct.pack("<12fH", *ni, *ti.ravel(), 0))


def aristas(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pares de vértices donde la superficie dobla: lo que dibujaría un plano."""
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    caras = defaultdict(list)
    for i, (a, b, c) in enumerate(t):
        for p, q in ((a, b), (b, c), (c, a)):
            caras[(min(p, q), max(p, q))].append(i)
    limite = np.cos(np.radians(ANGULO_ARISTA))
    return np.array([e for e, f in caras.items() if len(f) != 2 or n[f[0]] @ n[f[1]] < limite])


def componentes(ar: np.ndarray) -> list[np.ndarray]:
    padre: dict[int, int] = {}

    def raiz(a):
        padre.setdefault(a, a)
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    for a, b in ar:
        padre[raiz(int(a))] = raiz(int(b))
    grupos = defaultdict(set)
    for a, b in ar:
        grupos[raiz(int(a))].update((int(a), int(b)))
    return [np.array(sorted(g)) for g in grupos.values()]


def perforaciones(v: np.ndarray, ar: np.ndarray) -> list[dict]:
    """Bordes pequeños sobre una cara: agujeros y ranuras de la caja."""
    medidas = v.max(0) - v.min(0)
    salida = {}
    for idx in componentes(ar):
        p = v[idx]
        tam = p.max(0) - p.min(0)
        if (tam >= medidas * 0.8).sum() >= 2:
            continue               # contorno exterior o cavidad
        eje = int(np.argmin(tam))
        otros = [i for i in range(3) if i != eje]
        clave = (eje, *np.round(p[:, otros].min(0), 1), *np.round(tam[otros], 1))
        redondo = len(idx) > 12
        salida[clave] = {"eje": eje, "otros": otros, "origen": p[:, otros].min(0), "tam": tam[otros], "redondo": redondo}
    return list(salida.values())


def vista(ax, v, ar, ejes, titulo):
    i, j = ejes
    for a, b in ar:
        ax.plot(v[[a, b], i], v[[a, b], j], color=TINTA, lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=9.5, fontweight="bold", color=NAVY, pad=14)


def cota(ax, a, b, offset, texto, vertical=False):
    if vertical:
        x = a[0] + offset
        ax.annotate("", (x, a[1]), (x, b[1]), arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=0.8))
        ax.text(x + (1.5 if offset > 0 else -1.5), (a[1] + b[1]) / 2, texto, rotation=90, va="center",
                ha="left" if offset > 0 else "right", fontsize=8, color=TINTA)
    else:
        y = a[1] + offset
        ax.annotate("", (a[0], y), (b[0], y), arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=0.8))
        ax.text((a[0] + b[0]) / 2, y + (1.5 if offset > 0 else -1.5), texto, ha="center",
                va="bottom" if offset > 0 else "top", fontsize=8, color=TINTA)


def plano(caja, tapa) -> list[dict]:
    (vc, tc), (vt, tt) = caja, tapa
    ac, at = aristas(vc, tc), aristas(vt, tt)
    fig = plt.figure(figsize=(13, 8.2))
    fig.text(0.04, 0.955, "Plano de la envolvente", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(0.04, 0.925, "Medidas en mm, tomadas del modelo. Las tolerancias se definen en el CAD del equipo.",
             fontsize=9, color=GRIS)
    ax1 = fig.add_axes([0.03, 0.40, 0.30, 0.47])
    vista(ax1, vc, ac, (0, 1), "Caja · planta (X-Y)")
    cota(ax1, (0, 0), (vc[:, 0].max(), 0), -7, f"{vc[:, 0].max():.1f}")
    cota(ax1, (0, 0), (0, vc[:, 1].max()), -7, f"{vc[:, 1].max():.1f}", vertical=True)
    ax2 = fig.add_axes([0.36, 0.40, 0.21, 0.47])
    vista(ax2, vc, ac, (2, 1), "Caja · lateral (Z-Y)")
    cota(ax2, (0, 0), (vc[:, 2].max(), 0), -7, f"{vc[:, 2].max():.1f}")
    ax3 = fig.add_axes([0.03, 0.05, 0.30, 0.28])
    vista(ax3, vc, ac, (0, 2), "Caja · superior (X-Z)")
    vt2 = vt.copy()
    vt2[:, 2] += 129.5
    ax4 = fig.add_axes([0.36, 0.05, 0.21, 0.28])
    vista(ax4, vt2, at, (0, 1), "Tapa · canto (X-Y)")
    cota(ax4, (0, 0), (0, vt2[:, 1].max()), -5, f"{vt2[:, 1].max():.1f}", vertical=True)

    filas = perforaciones(vc, ac)
    nombres = "XYZ"
    ax = fig.add_axes([0.60, 0.40, 0.38, 0.47])
    ax.axis("off")
    ax.set_title("Aberturas de la caja", fontsize=9.5, fontweight="bold", color=NAVY, pad=14, loc="left")
    celdas = []
    for k, f in enumerate(sorted(filas, key=lambda r: (r["eje"], *r["origen"])), 1):
        a, b = f["otros"]
        forma = f"ø {f['tam'][0]:.1f}" if f["redondo"] else f"{f['tam'][0]:.1f} × {f['tam'][1]:.1f}"
        cx, cy = f["origen"] + f["tam"] / 2
        celdas.append([str(k), f"Plano {nombres[a]}{nombres[b]}", forma, f"{nombres[a]} {cx:.1f} · {nombres[b]} {cy:.1f}"])
    t = ax.table(cellText=celdas, colLabels=["#", "Cara", "Tamaño", "Centro"], loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for (r, _), c in t.get_celld().items():
        c.set_edgecolor(LINEA)
        if r == 0:
            c.set_facecolor(NAVY)
            c.set_text_props(color="white", fontweight="bold")

    grosor = float(np.unique(vc[:, 0].round(2))[1])
    interior = vc[:, 0].max() - 2 * grosor, vc[:, 1].max() - 2 * grosor
    borde = vt2[:, 0], vt2[:, 2]
    labio = np.unique(vt[:, 0].round(2)), np.unique(vt[:, 2].round(2))
    ax5 = fig.add_axes([0.60, 0.05, 0.38, 0.30])
    ax5.axis("off")
    notas = [f"Pared de la caja: {grosor:.1f} mm",
             f"Cavidad: {interior[0]:.1f} × {interior[1]:.1f} mm",
             f"Labio de la tapa: {labio[0][2] - labio[0][1]:.1f} × {labio[1][2] - labio[1][1]:.1f} mm, 3,0 mm de alto",
             "Holgura nominal labio-cavidad: 0,0 mm",
             "Cierre por encastre del labio; el modelo no tiene agujeros de tornillo"]
    ax5.text(0, 1, "\n".join(notas), va="top", fontsize=9, color=TINTA, linespacing=1.7)
    fig.savefig(PLANOS / "ensamble_2d.png", dpi=200, facecolor="white")
    plt.close(fig)
    return filas


def rasterizar(mallas: list, tam: int = 1500) -> tuple[np.ndarray, tuple]:
    """Proyección ortogonal con z-buffer: matplotlib 3D ordena mal las caras grandes."""
    az, el = np.radians(38), np.radians(26)
    der = np.array([np.cos(az), np.sin(az), 0.0])
    frente = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    arriba = np.cross(frente, der)
    todos = np.vstack([m[0] for m in mallas])
    plano_xy = np.column_stack([todos @ der, todos @ arriba])
    minimo, maximo = plano_xy.min(0), plano_xy.max(0)
    escala = (tam - 120) / (maximo - minimo).max()
    centro = (tam - (maximo - minimo) * escala) / 2

    def proyectar(p):
        return np.column_stack([(p @ der - minimo[0]) * escala + centro[0],
                                tam - ((p @ arriba - minimo[1]) * escala + centro[1]), p @ frente])

    img = np.ones((tam, tam, 3))
    zbuf = np.full((tam, tam), -np.inf)
    luz = np.array([0.35, -0.55, 0.76])
    luz /= np.linalg.norm(luz)
    for v, tri, color in mallas:
        base = np.array(matplotlib.colors.to_rgb(color))
        for a, b, c in v[tri]:
            n = np.cross(b - a, c - a)
            n /= max(np.linalg.norm(n), 1e-12)
            brillo = 0.62 + 0.38 * max(n @ luz, 0.0) if n @ frente > 0 else 0.5
            p = proyectar(np.array([a, b, c]))
            x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), tam - 1))
            y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), tam - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
            if abs(d) < 1e-9:
                continue
            l0 = ((p[1, 1] - p[2, 1]) * (gx - p[2, 0]) + (p[2, 0] - p[1, 0]) * (gy - p[2, 1])) / d
            l1 = ((p[2, 1] - p[0, 1]) * (gx - p[2, 0]) + (p[0, 0] - p[2, 0]) * (gy - p[2, 1])) / d
            l2 = 1 - l0 - l1
            dentro = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = l0 * p[0, 2] + l1 * p[1, 2] + l2 * p[2, 2]
            vista_z = zbuf[y0:y1 + 1, x0:x1 + 1]
            pinta = dentro & (z > vista_z)
            vista_z[pinta] = z[pinta]
            img[y0:y1 + 1, x0:x1 + 1][pinta] = base * brillo
    # Aristas de la pieza, solo donde son visibles.
    for v, tri, _ in mallas:
        for a, b in aristas(v, tri):
            p = proyectar(np.linspace(v[a], v[b], 900))
            x, y = p[:, 0].astype(int), p[:, 1].astype(int)
            ok = (x >= 0) & (x < tam) & (y >= 0) & (y < tam)
            x, y, z = x[ok], y[ok], p[ok, 2]
            visible = z >= zbuf[y, x] - 1.2 * 1
            img[y[visible], x[visible]] = np.array([0.06, 0.11, 0.18])
    return img, (proyectar,)


def explosionada(caja, tapa) -> None:
    (vc, tc), (vt, tt) = caja, tapa
    altura = float(vc[:, 2].max())
    separacion = 42.0
    # La tapa viene acostada en su archivo: se gira para cerrar la cara abierta. El giro
    # invierte la orientación de los triángulos, por eso se intercambian dos vértices.
    vt2 = np.column_stack([vt[:, 0], -vt[:, 2], -vt[:, 1] + 5.5 + altura + separacion])
    img, (proyectar,) = rasterizar([(vc, tc, "#3E7CC4"), (vt2, tt[:, [0, 2, 1]], "#6FA8DC")])
    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.imshow(np.clip(img, 0, 1), interpolation="lanczos")
    ax.axis("off")
    cx, cy = vc[:, 0].mean(), vc[:, 1].mean()
    p = proyectar(np.array([[cx, cy, altura + 1], [cx, cy, altura + separacion - 3]]))
    ax.plot(p[:, 0], p[:, 1], color=GRIS, lw=1, ls=(0, (4, 3)))
    for texto, punto in (("1", [vc[:, 0].max(), 0, 12]), ("2", [vt2[:, 0].max(), 0, vt2[:, 2].max()])):
        q = proyectar(np.array([punto]))[0]
        ax.text(q[0] + 70, q[1] + 10, texto, color="white", fontsize=13, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle", fc=NAVY, ec="none"))
    fig.text(0.05, 0.965, "Vista explosionada de la envolvente", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.93, "1. Caja        2. Tapa con labio de encastre", fontsize=10, color=TINTA)
    fig.savefig(PLANOS / "vista_explosionada.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def generar() -> dict | None:
    rutas = CAD / "caja final.3mf", CAD / "Tapa final.3mf"
    if not all(r.exists() for r in rutas):
        return None
    PLANOS.mkdir(parents=True, exist_ok=True)
    caja, tapa = leer_3mf(rutas[0]), leer_3mf(rutas[1])
    guardar_stl(CAD / "caja_final.stl", *caja)
    guardar_stl(CAD / "tapa_final.stl", *tapa)
    filas = plano(caja, tapa)
    explosionada(caja, tapa)
    medida = lambda v: v.max(0) - v.min(0)  # noqa: E731
    return {"caja": medida(caja[0]), "tapa": medida(tapa[0]), "aberturas": filas}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = generar()
    print(r)
