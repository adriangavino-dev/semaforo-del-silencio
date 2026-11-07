"""Diagramas del expediente técnico, dibujados desde los datos reales del proyecto.

Pines y umbrales salen de firmware/semaforo_esp32/config.example.h y de
config/config.yaml, así que si cambian, el diagrama cambia con ellos.

    python docs/entrega/diagramas.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))
from src import config as cfg  # noqa: E402

NAVY = "#0F1C2E"
TINTA = "#1C2733"
GRIS = "#5B6775"
LINEA = "#C9D1DA"
PANEL = "#F3F5F8"
AZUL = "#1F6FB2"
AZUL_C = "#E4EFF8"
VERDE = "#2E9E5B"
AMBAR = "#D99A00"
ROJO = "#D6453D"
AVISO = "#FFF3CD"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TINTA})


def pines() -> dict:
    t = (RAIZ / "firmware/semaforo_esp32/config.example.h").read_text(encoding="utf-8")
    return {k: int(v) for k, v in re.findall(r"#define (PIN_\w+)\s+(\d+)", t)}


def lienzo(ancho, alto):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    return fig, ax


def caja(ax, x, y, w, h, titulo=None, cuerpo=None, fondo="white", borde=LINEA, lw=1.2, tam=10, color=TINTA):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fondo, ec=borde, lw=lw))
    cx = x + w / 2
    if titulo and cuerpo:
        ax.text(cx, y + h - 0.2, titulo, ha="center", va="top", fontsize=tam, fontweight="bold", color=color)
        ax.text(cx, y + h - 0.47, cuerpo, ha="center", va="top", fontsize=tam - 1.8, color=GRIS, linespacing=1.3)
    elif titulo:
        ax.text(cx, y + h / 2, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color=color)


def flecha(ax, a, b, color=AZUL, texto=None, lado=0.0, estilo="-|>", rad=0.0, lw=1.5, tam=8.2, ls="-", ptexto=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if texto:
        x, y = ptexto or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + lado)
        ax.text(x, y, texto, ha="center", va="center", fontsize=tam, color=TINTA,
                bbox=dict(fc="white", ec="none", pad=1.5), linespacing=1.25)


def guardar(fig, carpeta: Path, nombre: str) -> None:
    fig.savefig(carpeta / nombre, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def arquitectura(carpeta: Path) -> None:
    fig, ax = lienzo(13, 7.2)
    # Unidad monolítica
    ax.add_patch(Rectangle((0.15, 0.3), 5.55, 6.55, fc=PANEL, ec=NAVY, lw=1.8))
    ax.text(2.92, 6.62, "UNIDAD MONOLÍTICA", ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(2.92, 6.34, "Con 5 V funciona sola: mide, decide y muestra", ha="center", fontsize=8.8, color=GRIS)
    caja(ax, 1.75, 2.75, 2.35, 2.0, "ESP32", "Tarea audio/LED/servo\nTarea red (núcleo 0)\nControl propio por dB\nServidor HTTP :80",
         fondo=NAVY, borde=NAVY, color="white", tam=11.5)
    ax.texts[-1].set_color("#C9D6E3")
    caja(ax, 0.4, 5.15, 1.75, 0.95, "INMP441", "Micrófono I²S")
    caja(ax, 3.75, 5.15, 1.75, 0.95, "Servo MG90S", "3 posiciones")
    caja(ax, 0.4, 1.0, 1.75, 1.1, "Módulo 5 V", "ELEGOO, con\ninterruptor; entra USB")
    caja(ax, 3.75, 1.0, 1.75, 1.1, "Semáforo", "LED V / A / R")
    flecha(ax, (1.45, 5.15), (2.2, 4.75), texto="I²S", ptexto=(1.45, 4.86))
    flecha(ax, (3.65, 4.75), (4.45, 5.15), texto="PWM", ptexto=(4.45, 4.86))
    flecha(ax, (1.45, 2.1), (2.2, 2.75), texto="5 V", color=ROJO, ptexto=(1.45, 2.5))
    flecha(ax, (3.65, 2.75), (4.45, 2.1), texto="GPIO", ptexto=(4.45, 2.5))
    # Hotspot
    caja(ax, 5.87, 5.42, 2.31, 1.15, "Hotspot Wi-Fi", "iPhone · 2,4 GHz · red /28\nTodo el tráfico pasa por aquí",
         fondo=AZUL_C, borde=AZUL, tam=10)
    # Estación
    ax.add_patch(Rectangle((8.35, 0.3), 4.5, 6.55, fc="white", ec=GRIS, lw=1.4, ls=(0, (5, 3))))
    ax.text(10.6, 6.62, "ESTACIÓN DE ANÁLISIS (opcional)", ha="center", fontsize=11, fontweight="bold", color=GRIS)
    ax.text(10.6, 6.34, "Laptop: fuera de la unidad; suma visión y clasificación", ha="center", fontsize=8.8, color=GRIS)
    caja(ax, 8.6, 4.4, 4.0, 1.55, "run_session.py", "YOLOv8-Pose · ojos · dormidos\nYAMNet + RandomForest\nFusión: máquina de estados")
    caja(ax, 8.6, 2.55, 4.0, 1.4, "serve_dashboard.py", "Recibe telemetría · descubre al ESP32\nTablero web :8000")
    caja(ax, 8.6, 0.75, 1.9, 1.15, "SQLite", "Lecturas,\neventos")
    caja(ax, 10.7, 0.75, 1.9, 1.15, "sesiones/", "Reporte, video,\naudio, fotos")
    flecha(ax, (10.6, 4.4), (10.6, 3.95), color=GRIS)
    flecha(ax, (9.55, 2.55), (9.55, 1.9), color=GRIS)
    flecha(ax, (12.45, 4.4), (12.3, 1.9), color=GRIS, rad=-0.28)
    # Enlaces por Wi-Fi: todos cruzan el hueco entre la unidad y la estación
    enlaces = [(4.95, "UDP :8766 · audio 16 kHz", AZUL, "-|>", (5.7, 8.6), "-"),
               (4.48, "HTTP :80 · color, servo", VERDE, "-|>", (8.6, 5.7), "-"),
               (3.5, "HTTP :8000 · telemetría 1 s", AZUL, "-|>", (5.7, 8.6), "-"),
               (3.0, "UDP :8765 · descubrimiento", GRIS, "<|-|>", (5.7, 8.6), "--")]
    for y, texto, color, estilo, (xa, xb), ls in enlaces:
        flecha(ax, (xa, y), (xb, y), color=color, estilo=estilo, ls=ls)
        ax.text(7.02, y + 0.17, texto, ha="center", va="center", fontsize=8.4, color=TINTA)
    guardar(fig, carpeta, "arquitectura.png")


def maquina_estados(carpeta: Path) -> None:
    conf = cfg.cargar()
    db = conf["audio"]["db"]
    am, ro = float(db["umbral_amarillo"]), float(db["umbral_rojo"])
    m = float(db["sin_modelo"]["margen_bajada"])
    vmax = float(db["voz_docente_max_db"])
    n = int(conf["fusion"]["confirmaciones_estado"])

    fig, ax = lienzo(13, 6.8)
    pos = {"VERDE": (2.1, 2.75), "AMARILLO": (6.5, 2.75), "ROJO": (10.9, 2.75)}
    color = {"VERDE": VERDE, "AMARILLO": AMBAR, "ROJO": ROJO}
    r = 0.92
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc=color[k], ec="none"))
        ax.text(x, y, k, ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")

    v, a, ro_ = pos["VERDE"], pos["AMARILLO"], pos["ROJO"]
    flecha(ax, (v[0] + r - 0.05, v[1] + 0.32), (a[0] - r + 0.05, a[1] + 0.32),
           texto=f"murmullo y nivel > {am:.0f} dB\no movimiento anómalo (inmediato)", ptexto=(4.3, 3.68))
    flecha(ax, (a[0] - r + 0.05, a[1] - 0.32), (v[0] + r - 0.05, v[1] - 0.32),
           color=GRIS, texto=f"nivel < {am - m:.0f} dB", ptexto=(4.3, 2.08))
    flecha(ax, (a[0] + r - 0.05, a[1] + 0.32), (ro_[0] - r + 0.05, ro_[1] + 0.32),
           texto=f"murmullo y nivel > {ro:.0f} dB", ptexto=(8.7, 3.45))
    flecha(ax, (ro_[0] - r + 0.05, ro_[1] - 0.32), (a[0] + r - 0.05, a[1] - 0.32),
           color=GRIS, texto=f"nivel < {ro - m:.0f} dB", ptexto=(8.7, 2.08))
    flecha(ax, (v[0] + 0.3, v[1] + r - 0.02), (ro_[0] - 0.3, ro_[1] + r - 0.02), rad=-0.3,
           texto=f"murmullo y nivel > {ro:.0f} dB", ptexto=(6.5, 5.2))
    flecha(ax, (ro_[0] - 0.3, ro_[1] - r + 0.02), (v[0] + 0.3, v[1] - r + 0.02), rad=-0.3,
           color=GRIS, texto=f"nivel < {am - m:.0f} dB", ptexto=(6.5, 0.72))
    flecha(ax, (0.2, 2.75), (v[0] - r, 2.75), color=TINTA, texto="inicio", ptexto=(0.5, 3.0))

    ax.text(0.1, 6.45, "Máquina de estados de la sesión (src/fusion/state.py)", fontsize=12.5,
            fontweight="bold", color=NAVY)
    ax.text(0.1, 6.13, f"Suben por clase de audio + nivel; bajan solo por nivel. Todo cambio pide {n} s seguidos "
            "del mismo color,", fontsize=9.2, color=GRIS)
    ax.text(0.1, 5.88, f"salvo el movimiento anómalo, que ya viene confirmado. Sin audio se mantiene el color. "
            f"Voz del docente sobre {vmax:.0f} dB cuenta como murmullo.", fontsize=9.2, color=GRIS)
    ax.text(0.1, 0.12, f"Sin laptop, el firmware usa histéresis de 2 dB: sube a {am + 2:.0f} / {ro + 2:.0f} dB y baja "
            f"bajo {am - 2:.0f} / {ro - 2:.0f} dB. Sin orden de la laptop por 15 s, retoma este control.",
            fontsize=9.2, color=TINTA)
    ax.text(0.1, -0.16, "Al cerrar la sesión, la laptop deja el semáforo en VERDE.", fontsize=9.2, color=TINTA)
    guardar(fig, carpeta, "maquina_estados.png")


def esquematico(carpeta: Path) -> None:
    p = pines()
    fig, ax = lienzo(13, 7.6)
    ax.text(0.1, 7.35, "Diagrama esquemático", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.1, 7.07, "Pines tomados del firmware; módulos según la unidad fabricada.",
            fontsize=9.2, color=GRIS)
    # ESP32
    x0, y0, w, h = 5.0, 1.25, 3.0, 4.75
    ax.add_patch(Rectangle((x0, y0), w, h, fc=NAVY, ec=NAVY))
    ax.text(x0 + w / 2, y0 + h - 0.35, "U1 · ESP32 DevKit", ha="center", color="white", fontweight="bold", fontsize=11)
    ax.text(x0 + w / 2, y0 + h - 0.62, "módulo ESP32-WROOM-32", ha="center", color="#C9D6E3", fontsize=8.5)

    def pin_izq(y, nombre):
        ax.plot([x0 - 0.3, x0], [y, y], color=TINTA, lw=1.3)
        ax.text(x0 + 0.1, y, nombre, va="center", color="white", fontsize=9)

    def pin_der(y, nombre):
        ax.plot([x0 + w, x0 + w + 0.3], [y, y], color=TINTA, lw=1.3)
        ax.text(x0 + w - 0.1, y, nombre, va="center", ha="right", color="white", fontsize=9)

    # INMP441
    mic = Rectangle((1.0, 2.45), 1.95, 2.7, fc="white", ec=TINTA, lw=1.3)
    ax.add_patch(mic)
    ax.text(1.97, 5.3, "U2 · INMP441", ha="center", fontweight="bold", fontsize=10)
    filas = [("SD", f"D{p['PIN_I2S_SD']}", 4.75, AZUL), ("SCK", f"D{p['PIN_I2S_SCK']}", 4.3, AZUL),
             ("WS", f"D{p['PIN_I2S_WS']}", 3.85, AZUL), ("VDD", "3V3", 3.4, ROJO), ("GND", "GND", 2.95, TINTA)]
    for señal, pin, y, c in filas:
        ax.text(1.1, y, señal, va="center", fontsize=9)
        ax.plot([2.95, x0 - 0.3], [y, y], color=c, lw=1.4)
        pin_izq(y, pin)
    ax.text(1.1, 2.62, "L/R → GND", va="center", fontsize=8.5, color=GRIS)

    def nodo(x, y):
        ax.add_patch(Circle((x, y), 0.055, fc=TINTA, ec=TINTA, zorder=5))

    xr = x0 + w + 0.3
    xret = xr + 2.6    # retorno de los LED y del servo hacia GND

    # Servo: arriba, para que su línea de 5 V no cruce los LED
    y_s = 5.1
    pin_der(y_s, f"D{p['PIN_SERVO']}")
    ax.add_patch(Rectangle((xr + 0.75, 4.72), 1.35, 0.8, fc="white", ec=TINTA, lw=1.3, zorder=3))
    ax.text(xr + 1.42, 5.24, "M1 · MG90S", ha="center", fontweight="bold", fontsize=9.3, zorder=4)
    ax.text(xr + 1.42, 4.94, "S / + / GND", ha="center", fontsize=8.3, color=GRIS, zorder=4)
    ax.plot([xr, xr + 0.75], [y_s, y_s], color=AZUL, lw=1.4)
    ax.plot([xr + 2.1, xret], [4.95, 4.95], color=TINTA, lw=1.3)

    # LEDs
    leds = [("VERDE", p["PIN_LED_VERDE"], 4.05, VERDE, "D1"), ("AMARILLO", p["PIN_LED_AMARILLO"], 3.2, AMBAR, "D2"),
            ("ROJO", p["PIN_LED_ROJO"], 2.35, ROJO, "D3")]
    for i, (nom, gp, y, c, ref) in enumerate(leds):
        pin_der(y, f"D{gp}")
        ax.plot([xr, xr + 0.35], [y, y], color=TINTA, lw=1.3)
        ax.add_patch(Rectangle((xr + 0.35, y - 0.12), 0.85, 0.24, fc="white", ec=TINTA, lw=1.3))
        ax.text(xr + 0.78, y + 0.27, f"R{i + 1} 220 Ω", ha="center", fontsize=8.3)
        ax.plot([xr + 1.2, xr + 1.75], [y, y], color=TINTA, lw=1.3)
        ax.plot([xr + 1.75, xr + 1.75, xr + 2.05, xr + 1.75], [y - 0.17, y + 0.17, y, y - 0.17], color=c, lw=1.8)
        ax.plot([xr + 2.05, xr + 2.05], [y - 0.17, y + 0.17], color=c, lw=1.8)
        ax.text(xr + 1.9, y + 0.33, f"{ref} {nom.lower()}", ha="center", fontsize=8.3)
        ax.plot([xr + 2.05, xret], [y, y], color=TINTA, lw=1.3)
        nodo(xret, y)
    ax.plot([xret, xret], [4.95, 0.75], color=TINTA, lw=1.3)
    nodo(xret, 4.95)
    nodo(xret, 0.75)

    # Riel de potencia
    yr = 6.55
    ax.plot([3.5, 12.2], [yr, yr], color=ROJO, lw=2)
    ax.text(12.25, yr, "5 V", va="center", color=ROJO, fontweight="bold")
    ax.add_patch(Rectangle((0.5, yr - 0.34), 3.0, 0.68, fc="white", ec=TINTA, lw=1.3, zorder=3))
    ax.text(2.0, yr + 0.1, "U3 · Módulo ELEGOO", ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=4)
    ax.text(2.0, yr - 0.15, "entrada USB, interruptor, salidas 5 V y 3,3 V", ha="center", va="center", fontsize=7.2,
            color=GRIS, zorder=4)
    # VIN
    xv = x0 + w / 2 - 0.5
    ax.plot([xv, xv], [yr, y0 + h], color=ROJO, lw=1.6)
    nodo(xv, yr)
    ax.text(xv + 0.08, 6.25, "VIN", fontsize=8.5, color=ROJO)
    # Servo a 5 V y C1
    ax.plot([xr + 1.42] * 2, [yr, 5.52], color=ROJO, lw=1.5)
    nodo(xr + 1.42, yr)
    xc = xret
    # Tierra común
    yg = 0.75
    ax.plot([0.5, xc], [yg, yg], color=TINTA, lw=2)
    ax.text(0.5, yg - 0.25, "GND común", fontsize=9, fontweight="bold")
    ax.plot([x0 + w / 2] * 2, [y0, yg], color=TINTA, lw=1.5)
    nodo(x0 + w / 2, yg)
    ax.plot([0.5, 0.5], [yr - 0.32, yg], color=TINTA, lw=1.5)
    guardar(fig, carpeta, "esquematico.png")


def arnes(carpeta: Path) -> None:
    p = pines()
    fig, ax = lienzo(13, 5.8)
    ax.text(0.1, 5.55, "Diagrama de arneses y conectores", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.1, 5.27, "Conexiones de la unidad; los colores son de referencia.", fontsize=9.2, color=GRIS)
    ax.add_patch(Rectangle((4.9, 0.55), 3.2, 4.3, fc=PANEL, ec=NAVY, lw=1.6))
    ax.text(6.5, 4.5, "Placa principal", ha="center", fontweight="bold", color=NAVY, fontsize=11)
    ax.text(6.5, 4.2, "con ESP32 DevKit", ha="center", color=GRIS, fontsize=9)
    izq = [("J1", "Módulo ELEGOO", "2 vías · salida de 5 V", 3.3, [("5 V", ROJO), ("GND", TINTA)]),
           ("J2", "INMP441", "5 vías · cable plano", 1.45, [("SD", AZUL), ("SCK", "#7A5AF8"), ("WS", "#0E9384"),
                                                                 ("3V3", ROJO), ("GND", TINTA)])]
    der = [("J3", "Semáforo", "4 vías · LED + retorno", 3.3, [(f"D{p['PIN_LED_VERDE']}", VERDE), (f"D{p['PIN_LED_AMARILLO']}", AMBAR),
                                                              (f"D{p['PIN_LED_ROJO']}", ROJO), ("GND", TINTA)]),
           ("J4", "Servo MG90S", "3 vías · propio del servo", 1.45, [(f"D{p['PIN_SERVO']}", "#E67700"), ("5 V", ROJO), ("GND", "#7A4A1E")])]
    for ref, nom, nota, y, hilos in izq:
        caja(ax, 0.3, y - 0.45, 2.35, 1.2, nom, nota, tam=10)
        ax.add_patch(Rectangle((4.65, y - 0.42), 0.25, 1.1, fc=NAVY, ec=NAVY))
        ax.text(4.42, y + 0.87, ref, ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
        for i, (t, c) in enumerate(hilos):
            yy = y + 0.55 - i * 0.2
            ax.plot([2.65, 4.65], [yy, yy], color=c, lw=1.8)
            ax.text(3.45, yy + 0.02, t, fontsize=7, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    for ref, nom, nota, y, hilos in der:
        caja(ax, 10.35, y - 0.45, 2.35, 1.2, nom, nota, tam=10)
        ax.add_patch(Rectangle((8.1, y - 0.42), 0.25, 1.1, fc=NAVY, ec=NAVY))
        ax.text(8.6, y + 0.87, ref, ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
        for i, (t, c) in enumerate(hilos):
            yy = y + 0.55 - i * 0.22
            ax.plot([8.35, 10.35], [yy, yy], color=c, lw=1.8)
            ax.text(9.55, yy + 0.02, t, fontsize=7, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    guardar(fig, carpeta, "arnes.png")


def generar(carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    arquitectura(carpeta)
    maquina_estados(carpeta)
    esquematico(carpeta)
    arnes(carpeta)


if __name__ == "__main__":
    salida = RAIZ / "entrega_semana13" / "diagramas"
    generar(salida)
    print("Diagramas en", salida)
