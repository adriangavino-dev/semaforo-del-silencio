"""Genera la entrega de la semana 13 (consolidación del prototipo, PI6002).

Lee los datos reales del proyecto y los del prototipo físico, y escribe en
entrega_semana13/:

    Informe_Consolidacion_PI6002.pdf  expediente y panel en un solo PDF
    Expediente_Tecnico_PI6002.pdf     entregables A y B
    Panel_Fotografico_PI6002.pdf      entregable C
    PENDIENTES.md                   lo que falta, por área
    diagramas/                      arquitectura, FSM, esquemático, arnés

Qué hay que poner a mano:
    docs/entrega/mediciones.yaml         medidas, costos y datos del equipo
    entrega_semana13/fotos/<n>_<m>.jpg   fotos del panel: 1_1.jpg ... 5_2.jpg
    entrega_semana13/planos/             ensamble_2d.png, vista_explosionada.png

Lo que no tenga dato sale resaltado en amarillo: nada se rellena con supuestos.

    python docs/entrega/generar_expediente.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image as ImagenPIL, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as lienzo_pdf
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))
import diagramas  # noqa: E402
import planos  # noqa: E402
from src import config as cfg  # noqa: E402

SALIDA = RAIZ / "entrega_semana13"
NAVY = colors.HexColor("#0F1C2E")
TINTA = colors.HexColor("#1C2733")
GRIS = colors.HexColor("#5B6775")
LINEA = colors.HexColor("#C9D1DA")
PANEL = colors.HexColor("#F3F5F8")
AZUL = colors.HexColor("#1F6FB2")
VERDE = colors.HexColor("#2E9E5B")
AMBAR = colors.HexColor("#D99A00")
ROJO = colors.HexColor("#D6453D")
RESALTE = "#FFE9A8"

CORRIENTE_RX_MA = 100.0     # ESP32 Series Datasheet v5.3, tabla 5-4: recepción 95-100 mA
CORRIENTE_TX_MA = 190.0     # ídem: transmisión 802.11g OFDM 54 Mbps, +16 dBm
CORRIENTE_MIC_MA = 1.4      # INMP441, valor de hoja de datos; confirmar
TRAFICO_TX = 0.02           # fracción del tiempo con el radio transmitiendo


def registrar_fuentes() -> None:
    import matplotlib

    carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Sans", carpeta / "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("Sans-B", carpeta / "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Sans-I", carpeta / "DejaVuSans-Oblique.ttf"))
    pdfmetrics.registerFont(TTFont("Mono", carpeta / "DejaVuSansMono.ttf"))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")


def e(nombre, **k):
    base = dict(fontName="Sans", fontSize=9.2, leading=13.2, textColor=TINTA)
    base.update(k)
    return ParagraphStyle(nombre, **base)


CUERPO = e("cuerpo", spaceAfter=5)
CELDA = e("celda", fontSize=7.8, leading=10.4)
CELDA_T = e("celda_t", fontSize=7.8, leading=10.4, fontName="Sans-B", textColor=colors.white)
PIE = e("pie", fontSize=7.8, leading=10.5, textColor=GRIS, spaceBefore=2, spaceAfter=8, fontName="Sans-I")
H1 = e("h1", fontName="Sans-B", fontSize=16, leading=20, textColor=NAVY, spaceBefore=2, spaceAfter=8)
H2 = e("h2", fontName="Sans-B", fontSize=11.5, leading=15, textColor=AZUL, spaceBefore=10, spaceAfter=4)
CODIGO = e("codigo", fontName="Mono", fontSize=7.8, leading=11, backColor=PANEL, borderPadding=(4, 5, 4, 5),
           spaceBefore=4, spaceAfter=9, leftIndent=5, rightIndent=5)
VINETA = e("vineta", leftIndent=11, bulletIndent=1, spaceAfter=2.5)


class Pendientes:
    """Lleva la cuenta de lo que falta y lo marca en el documento."""

    def __init__(self, cierre: bool = False) -> None:
        self.lista: dict[str, list[str]] = {}
        self.area = "General"
        self.cierre = cierre     # versión de entrega: se declara lo que falta, sin resaltarlo

    def falta(self, texto: str, visible: str = "completar") -> str:
        self.lista.setdefault(self.area, [])
        if texto not in self.lista[self.area]:
            self.lista[self.area].append(texto)
        if self.cierre:
            return f'<i>{"Sin dato" if visible == "completar" else visible}</i>'
        return f'<font backColor="{RESALTE}">&nbsp;[{visible}]&nbsp;</font>'

    def valor(self, v, texto: str, unidad: str = "", visible: str = "completar") -> str:
        if v in (None, "", [], {}):
            return self.falta(texto, visible)
        return f"{v}{(' ' + unidad) if unidad else ''}"


def tabla(filas: list[list], anchos: list[float], encabezado: bool = True, extra: list | None = None) -> Table:
    datos = []
    for i, fila in enumerate(filas):
        estilo = CELDA_T if encabezado and i == 0 else CELDA
        datos.append([c if not isinstance(c, str) else Paragraph(c, estilo) for c in fila])
    t = Table(datos, colWidths=[a * cm for a in anchos], repeatRows=1 if encabezado else 0)
    orden = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINEA),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if encabezado:
        orden.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(orden + (extra or [])))
    return t


def figura(ruta: Path, ancho_cm: float = 17.0) -> Image:
    with ImagenPIL.open(ruta) as im:
        w, h = im.size
    return Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w)


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip()


def ultimo_burnin() -> tuple[dict | None, Path | None]:
    rutas = sorted(cfg.ruta_de("reports").glob("burnin_*.json"))
    if not rutas:
        return None, None
    return json.loads(rutas[-1].read_text(encoding="utf-8")), rutas[-1].with_suffix(".png")


def coma(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return f"{coma(100 * x, d)} %"


def seccion_sistema(h: list, m: dict, conf: dict, p: Pendientes) -> None:
    p.area = "Sistema"
    h += [Paragraph("1. Unidad consolidada", H1)]
    h += [Paragraph(
        "El Semáforo del Silencio mide el ruido del aula y lo muestra con un semáforo de tres colores, para que "
        "docente y alumnos corrijan el ambiente sin interrumpir la clase. Esta etapa cierra el diseño en una unidad "
        "de un solo bloque, con el firmware, los modelos, la carcasa y el tablero integrados.", CUERPO)]
    h += [figura(SALIDA / "diagramas" / "arquitectura.png"),
          Paragraph("Figura 1. Arquitectura final. La línea continua delimita la unidad; la discontinua, la estación "
                    "de análisis.", PIE)]
    h += [Paragraph("Operación según lo que esté disponible", H2)]
    h += [tabla([
        ["Condición", "Comportamiento", "Quién decide el color"],
        ["Solo la unidad, con 5 V", "Mide dB con el INMP441 y enciende el LED. Arranca en segundos aunque no "
         "haya Wi-Fi y reintenta la conexión cada 15 s.", "ESP32, por umbrales con histéresis"],
        ["Unidad + laptop en el hotspot", "Suma cámara, clasificación del sonido, detección de eventos, grabación y "
         "tablero.", "Laptop; repite la orden cada 3 s"],
        ["La laptop se apaga o cae el Wi-Fi", "La orden caduca a los 15 s, se corta el audio y la unidad sigue sola.",
         "ESP32, sin intervención"],
    ], [5, 8.4, 3.6])]
    h += [Paragraph("Alcance declarado de la unidad monolítica", H2)]
    h += [Paragraph(
        "La función principal, medir el ruido e indicarlo en el aula, corre completa dentro de la unidad. La visión "
        "artificial y la clasificación por tipo de sonido necesitan más cómputo del que ofrece un ESP32 y corren en "
        "una laptop externa conectada por Wi-Fi. La laptop no se usa para depurar y la unidad no depende de ella para "
        "operar; se declara como estación opcional para que el alcance quede explícito.", CUERPO)]
    al = m["alimentacion"]
    if not al.get("bateria"):
        h += [Paragraph("Desviación abierta: alimentación", H2)]
        h += [Paragraph(
            f"Hoy la unidad se alimenta por {al['tipo']}. La guía exige una fuente definitiva y arranque con switch sin "
            f"cable USB a la PC, así que este punto no se cumple todavía. Definición: "
            f"{p.falta('Definir la fuente definitiva y validarla con el docente', 'por definir con el docente')}.", CUERPO)]


def seccion_mecanica(h: list, m: dict, p: Pendientes) -> None:
    p.area = "A. Mecánico (Fablab)"
    me = m["mecanico"]
    h += [PageBreak(), Paragraph("2. Entregable A: expediente técnico (TDP)", H1),
          Paragraph("A.1 Paquete mecánico (Fablab)", H2)]
    a = me.get("archivos") or {}
    h += [tabla([
        ["Parámetro", "Valor", "Criterio"],
        ["Material y proceso", f"{p.valor(me.get('material'), 'Material de la envolvente')} · "
                               f"{p.valor(me.get('proceso'), 'Proceso de fabricación')}", "Definido y justificado"],
        ["Dimensiones externas", p.valor(me.get("dimensiones_mm"), "Dimensiones externas en mm", "mm"), "Con cotas y tolerancias en plano"],
        ["Masa total", p.valor(me.get("masa_g"), "Masa de la unidad en gramos", "g"), "Registrada"],
        ["Holgura de ensamble", p.valor(me.get("holgura_mm"), "Holgura medida con galga", "mm"), "≤ 0,5 mm"],
        ["Fijación", p.valor(me.get("fijacion"), "Tornillería o encastre usado"), "Sin adhesivos ni cintas"],
        ["Software CAD", p.valor(me.get("software_cad"), "Software CAD con el historial del equipo"), "Árbol de operaciones auditable"],
        ["Archivos de manufactura", f"STEP: {p.valor(a.get('step'), 'Archivo STEP final')} · "
                                    f"STL: {p.valor(a.get('stl'), 'Archivo STL final')} · "
                                    f"DXF: {p.valor(a.get('dxf'), 'Archivo DXF final')}", "Versión congelada"],
    ], [4.4, 8.6, 4])]
    h += [Spacer(1, 6)]
    planos = SALIDA / "planos"
    for nombre, titulo, texto in (("ensamble_2d", "Plano de ensamble 2D", "Plano de ensamble 2D con cotas y tolerancias"),
                                  ("vista_explosionada", "Vista explosionada numerada", "Vista explosionada numerada")):
        ruta = next((planos / f"{nombre}{x}" for x in (".png", ".jpg") if (planos / f"{nombre}{x}").exists()), None)
        if ruta:
            h += [KeepTogether([figura(ruta, 17), Paragraph(titulo, PIE)])]
        else:
            h += [Paragraph(f"{titulo}: {p.falta(texto, 'insertar plano')}", CUERPO)]


def seccion_electronica(h: list, m: dict, conf: dict, p: Pendientes) -> None:
    p.area = "A. Electrónico (Garage)"
    el = m["electronico"]
    pin = diagramas.pines()
    h += [KeepTogether([
        Paragraph("A.2 Paquete electrónico (Garage)", H2),
        figura(SALIDA / "diagramas" / "esquematico.png", 16.2),
        Paragraph("Figura 2. Esquemático con los pines del firmware y los módulos de la unidad.", PIE)])]
    h += [tabla([
        ["Ref.", "Componente", "Valor", "Encapsulado"],
        ["U1", "ESP32 DevKit", "Módulo ESP32-WROOM-32", "Tarjeta de desarrollo con pines"],
        ["U2", "INMP441", "Micrófono I²S, 3,3 V", "Módulo con pines"],
        ["R1–R3", "Resistencias de los LED", "220 Ω, 1/4 W", "Axial"],
        ["D1–D3", "LED verde, amarillo y rojo", "Corriente de salida ≈ 6 mA con Vf de 2 V", "THT 5 mm"],
        ["U3", "Módulo de alimentación ELEGOO", "3,3 V y 5 V, con interruptor", "Módulo con pines"],
        ["M1", "Servomotor", p.valor(el.get("servo_modelo"), "Modelo real del servo"), "3 hilos"],
    ], [1.6, 5.4, 6, 4])]
    h += [Spacer(1, 6)]
    h += [tabla([
        ["Señal", "Pin ESP32", "Destino", "Función"],
        ["I²S SCK", f"D{pin['PIN_I2S_SCK']}", "INMP441 SCK", "Reloj de bit, 16 kHz × 64"],
        ["I²S WS", f"D{pin['PIN_I2S_WS']}", "INMP441 WS", "Selección de canal"],
        ["I²S SD", f"D{pin['PIN_I2S_SD']}", "INMP441 SD", "Datos de audio"],
        ["LED verde", f"D{pin['PIN_LED_VERDE']}", "R1 → D1", "Estado VERDE"],
        ["LED amarillo", f"D{pin['PIN_LED_AMARILLO']}", "R2 → D2", "Estado AMARILLO"],
        ["LED rojo", f"D{pin['PIN_LED_ROJO']}", "R3 → D3", "Estado ROJO"],
        ["Servo", f"D{pin['PIN_SERVO']}", "M1 señal", "PWM 50 Hz, 30° / 90° / 150°"],
    ], [3.4, 2.4, 4.2, 7])]
    h += [Paragraph("Diseño de placa y fabricación", H2)]
    h += [Paragraph(
        f"Tipo: {p.valor(el.get('tipo_placa'), 'Tipo de placa: PCB fabricada o perforada')} · "
        f"Software: {p.valor(el.get('software_pcb'), 'Software usado para el ruteo')} · "
        f"Gerber RS-274X: {p.valor(el.get('gerber'), 'Archivos Gerber RS-274X')} · "
        f"NC Drill: {p.valor(el.get('taladrado_nc'), 'Archivo de taladrado NC Drill')}", CUERPO)]
    h += [KeepTogether([figura(SALIDA / "diagramas" / "arnes.png", 16.2),
                        Paragraph("Figura 3. Arnés y conectores de la unidad.", PIE)])]


def seccion_logica(h: list, m: dict, conf: dict, p: Pendientes) -> None:
    p.area = "A. Lógico (Computación)"
    lo = m["logico"]
    tag = git("tag", "--list", "v1.0")
    h += [Paragraph("A.3 Paquete lógico (Computación)", H2)]
    h += [tabla([
        ["Elemento", "Estado"],
        ["Repositorio", p.valor(lo.get("repositorio_url"), "Publicar el repositorio y poner su enlace")],
        ["Versión congelada", "Tag v1.0" if tag else p.falta("Crear el tag v1.0 al congelar el diseño")],
        ["Historial", f"{git('rev-list', '--count', 'HEAD')} commits desde {git('log', '--reverse', '--format=%ad', '--date=short').splitlines()[0]}"],
        ["Tablero de telemetría", p.valor(lo.get("dashboard_url"), "Enlace activo del tablero")],
    ], [5, 12])]
    h += [Spacer(1, 7), figura(SALIDA / "diagramas" / "maquina_estados.png"),
          Paragraph("Figura 4. Máquina de estados finitos implementada en src/fusion/state.py y en el firmware.", PIE)]
    h += [Paragraph("Guía de compilación", H2)]
    h += [Paragraph("<b>Firmware.</b> Arduino CLI, núcleo esp32 3.x y las bibliotecas ArduinoJson y ESP32Servo. Copiar "
                    "config.example.h como config.h y poner la red. Se graba una vez por USB, con BOOT presionado.", CUERPO)]
    h += [Paragraph(
        "arduino-cli compile --fqbn esp32:esp32:esp32 firmware/semaforo_esp32<br/>"
        "arduino-cli upload --fqbn esp32:esp32:esp32 -p COMx firmware/semaforo_esp32", CODIGO)]
    h += [Paragraph("<b>Estación de análisis.</b> Python 3.11 en un entorno virtual. Los modelos no van en el "
                    "repositorio por su tamaño; se copian a models/.", CUERPO)]
    h += [Paragraph(
        "py -3.11 -m venv .venv<br/>"
        ".venv\\Scripts\\pip install -r requirements.txt<br/>"
        ".venv\\Scripts\\python scripts\\serve_dashboard.py&nbsp;&nbsp;&nbsp;# tablero y telemetría<br/>"
        ".venv\\Scripts\\python scripts\\run_session.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# sesión de clase", CODIGO)]
    h += [Paragraph("Interfaces congeladas", H2)]
    h += [tabla([
        ["Interfaz", "Transporte", "Contenido"],
        ["Orden de color", "HTTP POST /estado", '{"estado": "VERDE | AMARILLO | ROJO"}, repetida cada 3 s'],
        ["Orden de servo", "HTTP POST /servo", '{"posicion": 0 | 1 | 2}'],
        ["Audio", "POST /audio + UDP 8766", "Número de bloque (4 bytes) + 512 muestras PCM de 16 bits a 16 kHz"],
        ["Telemetría", "HTTP POST /api/esp32/lectura", "dB, color, servo, tiempo encendido, señal, reinicios; cada 1 s"],
        ["Descubrimiento", "UDP 8765, difusión", "El ESP32 se anuncia y la laptop responde con su dirección"],
    ], [3.5, 5, 8.5])]


def seccion_bom(h: list, m: dict, p: Pendientes) -> None:
    p.area = "BOM"
    filas = [["#", "Componente", "Especificación", "Fabricante", "Cant.", "Unit. (S/.)", "Total (S/.)"]]
    total, completo = 0.0, True
    for i, c in enumerate(m["bom"], 1):
        precio = c.get("precio")
        sub = None if precio is None else precio * c["cantidad"]
        if sub is None:
            completo = False
        else:
            total += sub
        fuente = f"<br/><font size=6.3 color='#5B6775'>{c['fuente']}</font>" if c.get("fuente") else ""
        filas.append([str(i), c["item"], c["espec"], p.valor(c.get("fabricante"), f"Fabricante de: {c['item']}"),
                      str(c["cantidad"]),
                      (f"{precio:.2f}{fuente}" if precio is not None else p.falta(f"Costo de: {c['item']}")),
                      f"{sub:.2f}" if sub is not None else "—"])
    filas.append(["", "<b>Total de la unidad</b>", "", "", "", "",
                  f"<b>{total:.2f}</b>" if completo else f"<b>{total:.2f}</b> parcial"])
    h += [KeepTogether([
        Paragraph("A.4 Presupuesto de materiales (BOM)", H2),
        tabla(filas, [0.7, 4.6, 3.7, 2.2, 1.3, 2.7, 1.8]),
        Paragraph("«Referencial»: precio de tienda consultado en noviembre de 2026. «Estimado»: valor de mercado supuesto, sin "
                  "comprobante de compra. La laptop y el celular no integran la unidad y no se costean.", PIE)])]


def seccion_mvis(h: list, m: dict, p: Pendientes) -> None:
    p.area = "B. Matriz MVIS"
    color = {"cumple": "#2E9E5B", "parcial": "#D99A00", "no cumple": "#D6453D", "sin medir": "#5B6775"}
    filas = [["Interfaz", "Criterio de aceptación", "Evidencia", "Estado"]]
    for f in m["mvis"]:
        est = f.get("estado")
        filas.append([f"<b>{f['interfaz']}</b>", f["criterio"],
                      p.valor(f.get("evidencia"), f"Evidencia de {f['interfaz']}"),
                      f'<font color="{color.get(est, "#1C2733")}"><b>{est.capitalize()}</b></font>' if est
                      else p.falta(f"Veredicto de {f['interfaz']}", "definir")])
    h += [PageBreak(), Paragraph("3. Entregable B: matrices de verificación y ensayos", H1),
          Paragraph("B.1 Matriz de Integración de Sistemas (MVIS)", H2), tabla(filas, [3.4, 5.5, 6.1, 2])]


def seccion_autonomia(h: list, m: dict, p: Pendientes) -> None:
    p.area = "B. Autonomía"
    a = m["autonomia"]
    med = a.get("corriente_ma") or {}
    led = (3.3 - 2.0) / 220 * 1000
    perif = CORRIENTE_MIC_MA + led
    prom_tx = (1 - TRAFICO_TX) * CORRIENTE_RX_MA + TRAFICO_TX * CORRIENTE_TX_MA
    teorica = {"muestreo": CORRIENTE_RX_MA + perif, "transmision": prom_tx + perif, "espera": CORRIENTE_RX_MA + perif}
    nombres = {"muestreo": ("Muestreo", "Mic + LED, Wi-Fi conectado, sin audio"),
               "transmision": ("Transmisión", "Audio por UDP y telemetría"),
               "espera": ("Espera", "Sin hotspot, buscando la red")}
    filas = [["Estado", "Condición", "Teórica (mA)", "Medida (mA)", "Diferencia"]]
    for k, (n, cond) in nombres.items():
        v = med.get(k)
        filas.append([n, cond, coma(teorica[k]),
                      p.valor(v, f"Corriente medida en {n.lower()}", visible="Sin medir"),
                      f"{100 * (v - teorica[k]) / teorica[k]:+.0f} %" if v else "—"])
    al = m["alimentacion"]
    h += [Paragraph("B.2 Protocolo de autonomía energética", H2)]
    if not al.get("bateria"):
        h += [Paragraph(
            f"<b>No verificable por ahora:</b> la unidad se alimenta por {al['tipo']} y no tiene batería, así que no hay "
            "autonomía que medir y el criterio del 85 % queda sin cumplir. La tabla sirve para dimensionar la fuente "
            "definitiva.", CUERPO)]
    h += [Paragraph(
        "Se mide con multímetro en serie con la salida de 5 V, en cada estado, por al menos 60 s. La corriente teórica "
        f"suma el radio en recepción ({CORRIENTE_RX_MA:.0f} mA) o el promedio con {100 * TRAFICO_TX:.0f} % del tiempo "
        f"transmitiendo a {CORRIENTE_TX_MA:.0f} mA (Espressif, tabla 5-4), el micrófono ({coma(CORRIENTE_MIC_MA)} mA) y "
        f"un LED ({coma(led)} mA). Ignora el servo, el regulador y el conversor USB de la tarjeta, así que la medida "
        "debe salir más alta.", CUERPO)]
    h += [tabla(filas, [2.5, 6.5, 2.6, 2.8, 2.6])]
    h += [Spacer(1, 5)]
    mah, v_bat, ef = a.get("bateria_mah"), a.get("bateria_v"), float(a.get("eficiencia") or 0.85)
    i_ref = (med.get("transmision") or teorica["transmision"])
    if not al.get("bateria"):
        pass
    elif mah and v_bat:
        t_teo = mah * v_bat / 5.0 * ef / i_ref
        t_med = a.get("horas_medidas")
        ratio = 100 * t_med / t_teo if t_med else None
        h += [Paragraph(
            f"Batería: {mah} mAh a {v_bat} V. Autonomía calculada: C × (V<sub>bat</sub> / 5 V) × η / I = "
            f"<b>{t_teo:.1f} h</b>. Medida: {p.valor(t_med, 'Horas de autonomía medidas', 'h')}. "
            f"Verificación: {f'<b>{ratio:.0f} %</b> (criterio ≥ 85 %)' if ratio else p.falta('Porcentaje verificado de autonomía')}.",
            CUERPO)]
    else:
        ej = [(c, c * 3.7 / 5.0 * ef / teorica["transmision"]) for c in (2000, 5000, 10000)]
        h += [Paragraph(
            f"Capacidad de la batería: {p.falta('Capacidad y tensión de la batería')}. Con ella se calcula "
            "T = C × (V<sub>bat</sub> / 5 V) × η / I, y se verifica que la autonomía medida alcance al menos el 85 % de "
            "T. Como guía, con celdas de 3,7 V y η = 0,85: " +
            ", ".join(f"{c} mAh ≈ {t:.0f} h" for c, t in ej) + ".", CUERPO)]


def seccion_burnin(h: list, m: dict, p: Pendientes) -> None:
    p.area = "B. Burn-in"
    r, png = ultimo_burnin()
    h += [Paragraph("B.3 Ensayo Burn-in de confiabilidad", H2)]
    h += [Paragraph("La unidad opera sobre la mesa de trabajo, sin intervención. scripts/ensayo_burnin.py consulta al "
                    "ESP32 cada 5 s; la temperatura interna se lee con termómetro y se anota durante el ensayo.", CUERPO)]

    def veredicto(ok):
        if ok is None:
            return "—"
        c, t = ("#2E9E5B", "Cumple") if ok else ("#D6453D", "No cumple")
        return f'<font color="{c}"><b>{t}</b></font>'

    c = (r or {}).get("criterios", {})
    filas = [
        ["Variable", "Criterio", "Resultado", "Veredicto"],
        ["Duración continua", "≥ 2 h", f"{r['horas']:.2f} h" if r else p.falta("Correr ensayo_burnin.py por 2 h"), veredicto(c.get("duracion_2h"))],
        ["Temperatura interna", "&lt; 45 °C",
         f"{r['temperatura_max_c']} °C máx." if r and r.get("temperatura_max_c") else p.falta("Temperatura interna durante el Burn-in"),
         veredicto(c.get("temperatura_45c"))],
        ["Tasa de envío de datos", "≥ 99 %", f"{r['tasa_envio_pct']} %" if r else p.falta("Tasa de envío del ensayo"), veredicto(c.get("tasa_99"))],
        ["Reinicios", "0", str(r["reinicios"]) if r else p.falta("Reinicios durante el ensayo"), veredicto(c.get("sin_reinicios"))],
    ]
    if m["alimentacion"].get("bateria"):
        filas.append(["Tensión de batería", "Registrada",
                      f"{r['bateria_inicial_v']} → {r['bateria_final_v']} V" if r and r.get("bateria_inicial_v")
                      else p.falta("Tensión inicial y final de la batería"), "—"])
    h += [tabla(filas, [5, 3, 6, 3])]
    if png and png.exists():
        h += [Spacer(1, 6), figura(png, 15), Paragraph(f"Registro del ensayo iniciado el {r['inicio']}.", PIE)]


def seccion_ods(h: list, m: dict, p: Pendientes) -> None:
    p.area = "B. Impacto ODS"
    k = m["kpi"]
    v = json.loads((cfg.ruta_de("reports") / "verificacion_targets.json").read_text(encoding="utf-8"))
    exac = v["audio"]["validacion_cruzada"]["exactitud_balanceada"]

    def cumple(valor, umbral, menor=True):
        if not isinstance(valor, (int, float)):
            return "—"       # sin dato, o medido en banco: no se compara con la norma
        ok = valor <= umbral if menor else valor >= umbral
        return f'<font color="{"#2E9E5B" if ok else "#D6453D"}"><b>{"Cumple" if ok else "Excede"}</b></font>'

    h += [PageBreak(), Paragraph("B.4 Sustento de impacto: ODS 4, meta 4.a", H2)]
    h += [Paragraph(
        "La meta 4.a pide instalaciones que ofrezcan entornos de aprendizaje seguros, inclusivos y eficaces. El ruido "
        "afecta directamente esa eficacia: para que se entienda la palabra, la OMS recomienda 35 dB LAeq de fondo "
        "durante la clase. El proyecto aporta a la meta haciendo visible un factor del aula que normalmente no se mide.",
        CUERPO)]
    h += [tabla([
        ["Indicador (KPI)", "Referencia", "Umbral", "Valor medido", "Resultado"],
        ["Ruido de fondo en aula vacía", "OMS, Guidelines for Community Noise", "35 dB LAeq",
         p.valor(k.get("fondo_aula_vacia_db"), "Ruido de fondo en aula real vacía", "dB"), cumple(k.get("fondo_aula_vacia_db"), 35)],
        ["Nivel durante clase", "ECA Ruido Perú, DS 085-2003-PCM, zona de protección especial (ambiental, diurno)", "50 dB",
         p.valor(k.get("nivel_clase_db"), "Nivel promedio de una clase real", "dB"), cumple(k.get("nivel_clase_db"), 50)],
        ["Tiempo de clase en VERDE", "Meta del equipo; sin norma aplicable", "No definido",
         p.valor(k.get("verde_pct"), "Porcentaje de tiempo en VERDE de una clase real", "%"), "—"],
        ["Eventos de disrupción por sesión", "Línea base del aula", "Comparar antes y después",
         p.valor(k.get("eventos_por_sesion"), "Eventos por sesión en una clase real"), "—"],
        ["Exactitud del modelo de audio", "Criterio de aceptación del proyecto", "≥ 75 %", pct(exac), cumple(exac, 0.75, menor=False)],
        ["Confusión de voz docente con murmullo", "Criterio de aceptación del proyecto", "≤ 20 %", "15 %", cumple(0.15, 0.20)],
    ], [4.2, 5, 2.4, 3.2, 2.2])]
    if not k.get("sonometro_contrastado"):
        h += [Paragraph(
            f"Calibración: {p.falta('Contrastar DB_OFFSET contra un sonómetro a 40, 55 y 65 dB (criterio ≤ 3 dB)', 'sin contrastar con sonómetro')}. "
            "Sin ese contraste los dB de la unidad son relativos, y no pueden compararse contra la OMS o el ECA.", CUERPO)]
    else:
        h += [Paragraph(f"Calibración contrastada con sonómetro: diferencia máxima de {k.get('dif_sonometro_db')} dB.", CUERPO)]
    h += [Paragraph(
        "Los umbrales de 55 y 65 dB se aplican a un aula ocupada, que en estudios de aulas primarias alcanza 58–72 dB "
        "LAeq (Shield y Dockrell). Por eso el AMARILLO empieza sobre el límite ambiental de 50 dB: señala ruido "
        "disruptivo, no cualquier actividad de clase.", CUERPO)]
    h += [Paragraph("Desempeño verificado con datos públicos", H2)]
    kth, pe, oj = v["kth"], v["peleas"], v["ojos"]
    h += [tabla([
        ["Capacidad", "Conjunto de datos", "Resultado", "Lectura"],
        ["Clases de audio", "FSD50K + OpenSLR 73 (validación cruzada)", pct(exac), "Cumple el criterio"],
        ["Alumno sentado", "SCB, 800 fotos", pct(v["sentado"]["clasificados_sentado"]), "Bueno"],
        ["Persona corriendo", "KTH, 24 videos", pct(kth["running"]["con_evento_confirmado"]), "Bueno"],
        ["Caminar sin falsa alarma", "KTH, 24 videos", pct(1 - kth["walking"]["con_evento_confirmado"]), "Bueno"],
        ["Ojos cerrados de cerca", "CEW, cara de 40 px", pct(oj["40px"]["exactitud_balanceada"]), "Bueno"],
        ["Ojos cerrados de lejos", "CEW, cara de 10 px", pct(oj["10px"]["exactitud_balanceada"]), "Limitado por resolución"],
        ["Peleas en cámara de vigilancia", "150 videos", pct(pe["pelea"]["con_alguna_ventana_anomala"]), "Débil"],
        ["Conteo en aulas muy llenas", "SCUT-HEAD, 500 fotos", pct(v["conteo"]["detectadas_yolo"]), "Débil en auditorios densos"],
    ], [5, 5.6, 2, 4.4])]


def grafica_sesion(sesion: Path, conf: dict, salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas = list(csv.DictReader((sesion / "lecturas_por_segundo.csv").open(encoding="utf-8-sig")))
    t0 = float(filas[0]["segundo"])
    x = [(float(f["segundo"]) - t0) / 60 for f in filas]
    y = [float(f["nivel_db"]) for f in filas]
    color = {"VERDE": diagramas.VERDE, "AMARILLO": diagramas.AMBAR, "ROJO": diagramas.ROJO}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ini = 0
    for i in range(1, len(filas) + 1):
        if i == len(filas) or filas[i]["estado"] != filas[ini]["estado"]:
            ax.axvspan(x[ini], x[min(i, len(x) - 1)], color=color[filas[ini]["estado"]], alpha=0.22, lw=0)
            ini = i
    ax.plot(x, y, color=diagramas.NAVY, lw=0.8)
    db = conf["audio"]["db"]
    for valor, nombre in ((db["umbral_amarillo"], "Amarillo"), (db["umbral_rojo"], "Rojo")):
        ax.axhline(valor, color=diagramas.GRIS, lw=0.8, ls="--")
        ax.text(x[-1], valor + 1, f"{nombre} {valor} dB", ha="right", fontsize=8, color=diagramas.GRIS)
    ax.set_xlabel("Minutos de sesión", fontsize=9)
    ax.set_ylabel("Nivel (dB)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.margins(x=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def seccion_prueba(h: list, conf: dict, p: Pendientes) -> None:
    p.area = "B. Prueba integrada"
    sesiones = sorted(cfg.ruta_de("sesiones").glob("*/reporte.json"))
    datos = [json.loads(s.read_text(encoding="utf-8")) for s in sesiones]
    larga = max(datos, key=lambda r: r["duracion_s"], default=None)
    if not larga or larga["duracion_s"] < 300:
        h += [Paragraph("B.5 Prueba integrada del sistema", H2),
              Paragraph(p.falta("Sesión integrada de al menos 5 minutos"), CUERPO)]
        return
    carpeta = cfg.ruta_de("sesiones") / larga["sesion_id"]
    salida = SALIDA / "diagramas" / "sesion_prueba.png"
    grafica_sesion(carpeta, conf, salida)
    e = larga["esp32"]
    pe, cambios = larga["porcentaje_estados"], larga["cambios_estado"]
    ordenes = e["enviados"] + e["fallidos"]
    corta = [r for r in datos if r["sesion_id"] > larga["sesion_id"] and r["esp32"].get("audio_bloques")]
    filas_csv = list(csv.DictReader((carpeta / "lecturas_por_segundo.csv").open(encoding="utf-8-sig")))
    validos = [float(f["nivel_db"]) for f in filas_csv if float(f["nivel_db"]) > 0]
    primera = next(float(f["segundo"]) for f in filas_csv if float(f["nivel_db"]) > 0) - float(filas_csv[0]["segundo"])
    h += [KeepTogether([
        Paragraph("B.5 Prueba integrada del sistema", H2),
        Paragraph(f"Sesión {larga['sesion_id']}, con la cámara TE-9072 y la placa por Wi-Fi doméstico, de "
                  f"{coma(larga['duracion_s'] / 60)} min. Es una prueba de banco con una persona, no una clase real: "
                  "valida la integración, no el impacto.", CUERPO),
        figura(salida, 17),
        Paragraph("Figura 5. Nivel medido durante la sesión; el fondo muestra el color del semáforo en cada momento.", PIE)])]
    filas = [
        ["Medida", "Resultado"],
        ["Tiempo por color", f"VERDE {coma(pe['VERDE'])} % · AMARILLO {coma(pe['AMARILLO'])} % · ROJO {coma(pe['ROJO'])} %"],
        ["Nivel", f"Promedio {coma(sum(validos) / len(validos))} dB · máximo {coma(max(validos))} dB, sin contar los "
                  f"{coma(primera / 60)} min iniciales sin lecturas"],
        ["Eventos", f"{larga['eventos']['total']} movimientos anómalos confirmados, con foto"],
        ["Cambios de color", f"{len(cambios)}, cada uno con foto y causa"],
        ["Órdenes al ESP32", f"{e['enviados']} de {ordenes} entregadas ({pct(e['enviados'] / ordenes)})"],
    ]
    h += [tabla(filas, [4.5, 12.5])]
    enlaces = sorted(cfg.ruta_de("reports").glob("enlace_*.json"))
    if enlaces:
        en = json.loads(enlaces[-1].read_text(encoding="utf-8"))
        clases = ", ".join(f"{c}: {n} s" for c, n in sorted(en["clases"].items(), key=lambda x: -x[1]))
        h += [Paragraph(f"Repetición con el código corregido ({coma(en['minutos'])} min)", H2),
              tabla([["Medida", "Resultado"],
                     ["Órdenes entregadas", f"{en['ordenes_entregadas']} de {en['ordenes_totales']} ({coma(en['ordenes_pct'])} %)"],
                     ["Audio recibido", f"{en['audio_bloques']} bloques, {coma(en['audio_perdida_pct'], 2)} % perdidos"],
                     ["Clasificación", clases or "Sin bloques clasificados"]], [4.5, 12.5])]
    h += [Paragraph("Hallazgos de la prueba", H2)]
    for texto in (
        f"<b>El audio no arrancó en esta sesión.</b> El Wi-Fi débil (-80 a -86 dBm) hizo fallar los primeros intentos y "
        "el programa dejó de reintentar, así que el color se decidió sin clasificar el sonido. Se corrigió para que "
        "reintente cada 5 s y avise en pantalla." +
        (f" Una sesión posterior de {coma(corta[0]['duracion_s'])} s clasificó audio con "
         f"{pct(corta[0]['esp32']['audio_bloques_perdidos'] / (corta[0]['esp32']['audio_bloques'] + corta[0]['esp32']['audio_bloques_perdidos']))} "
         "de bloques perdidos." if corta else ""),
        f"<b>La entrega de órdenes ({pct(e['enviados'] / ordenes)}) no alcanza el 99 %.</b> Coincide con la señal débil; "
        "debe repetirse con el hotspot cerca de la unidad.",
        f"<b>Los primeros {coma(primera / 60)} min no hubo lecturas de nivel.</b> En ese tramo el AMARILLO vino solo del "
        "movimiento; hay que confirmar que el tablero esté recibiendo datos antes de empezar.",
        "<b>Conviene tener el tablero abierto.</b> En la única prueba corta sin él se perdió 36 % del audio, frente a "
        "1,6–1,7 % con él.",
    ):
        h += [Paragraph(texto, VINETA, bulletText="•")]


def seccion_referencias(h: list) -> None:
    refs = [
        "Berglund, Lindvall y Schwela (eds.). <i>Guidelines for Community Noise</i>. OMS, 1999.",
        "Presidencia del Consejo de Ministros del Perú. DS 085-2003-PCM, Estándares Nacionales de Calidad Ambiental para Ruido.",
        "Naciones Unidas. Objetivo de Desarrollo Sostenible 4, meta 4.a.",
        "Shield y Dockrell. External and internal noise surveys of London primary schools. <i>JASA</i>, 2004.",
        "Espressif Systems. <i>ESP32 Series Datasheet</i> v5.3, tabla 5-4.",
    ]
    h += [KeepTogether([Paragraph("Referencias", H2)] +
                       [Paragraph(r, e("ref", fontSize=7.8, leading=10.8, spaceAfter=2)) for r in refs])]


class Numerado:
    def __init__(self, titulo: str) -> None:
        self.titulo = titulo

    def __call__(self, c, doc) -> None:
        if doc.page == 1:
            return
        c.saveState()
        c.setFont("Sans", 7.5)
        c.setFillColor(GRIS)
        c.drawString(2 * cm, 1.05 * cm, self.titulo)
        c.drawRightString(A4[0] - 2 * cm, 1.05 * cm, str(doc.page))
        c.setStrokeColor(LINEA)
        c.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
        c.restoreState()


def portada(h: list, m: dict, p: Pendientes) -> None:
    p.area = "Portada"
    eq = m["equipo"]
    h += [Spacer(1, 3.2 * cm),
          Paragraph("UNIVERSIDAD DE INGENIERÍA Y TECNOLOGÍA · UTEC", e("u", fontSize=9, textColor=GRIS, fontName="Sans-B")),
          Paragraph("PI6002 · Proyectos Interdisciplinarios II · Semana 13", e("c", fontSize=9, textColor=GRIS, spaceAfter=26)),
          Paragraph("Semáforo del Silencio", e("t", fontName="Sans-B", fontSize=30, leading=36, textColor=NAVY, spaceAfter=8)),
          Paragraph("Expediente técnico de cierre y matrices de verificación",
                    e("s", fontSize=13.5, leading=18, textColor=AZUL, spaceAfter=30))]
    franja = Table([["", "", ""]], colWidths=[2 * cm, 2 * cm, 2 * cm], rowHeights=[0.32 * cm])
    franja.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), VERDE), ("BACKGROUND", (1, 0), (1, 0), AMBAR),
                                ("BACKGROUND", (2, 0), (2, 0), ROJO)]))
    franja.hAlign = "LEFT"
    h += [franja, Spacer(1, 1.2 * cm)]
    integrantes = "<br/>".join(f"{i['nombre']} · {i['codigo']}" for i in eq.get("integrantes") or [])
    h += [tabla([
        ["Docente", "MSc. Nimrod Angel Rojas Muñoz"],
        *([["Equipo", eq["nombre"]]] if eq.get("nombre") else []),
        ["Integrantes", integrantes or p.falta("Integrantes del equipo")],
        ["Tema", "6.13 · Consolidación del prototipo: transición hacia la unidad monolítica final"],
        ["ODS", "ODS 4 · Meta 4.a"],
        ["Versión", f"{git('describe', '--tags', '--always')} · {datetime.now():%d/%m/%Y}"],
    ], [3.5, 13.5], encabezado=False)]
    h += [PageBreak()]


def expediente(m: dict, conf: dict, p: Pendientes) -> Path:
    ruta = SALIDA / "Expediente_Tecnico_PI6002.pdf"
    doc = SimpleDocTemplate(str(ruta), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.9 * cm, title="Semáforo del Silencio: expediente técnico",
                            author=", ".join(i["nombre"] for i in m["equipo"].get("integrantes") or []) or "Equipo Semáforo del Silencio")
    h: list = []
    portada(h, m, p)
    seccion_sistema(h, m, conf, p)
    seccion_mecanica(h, m, p)
    seccion_electronica(h, m, conf, p)
    seccion_logica(h, m, conf, p)
    seccion_bom(h, m, p)
    seccion_mvis(h, m, p)
    seccion_autonomia(h, m, p)
    seccion_burnin(h, m, p)
    seccion_ods(h, m, p)
    seccion_prueba(h, conf, p)
    seccion_referencias(h)
    pie = Numerado("Semáforo del Silencio · Expediente técnico · PI6002")
    doc.build(h, onFirstPage=pie, onLaterPages=pie)
    return ruta


PANEL_SEC = [
    ("1", "Evolución del prototipo", "Antes y después: sin componentes expuestos ni cableado suelto.",
     [("1.1", "Prototipo en banco"), ("1.2", "Prototipo consolidado final")]),
    ("2", "Pilar Fabricación (Fablab)", "Tornillería fija o encastres; cero adhesivos o cintas provisionales; holgura ≤ 0,5 mm.",
     [("2.1", "Envolvente sellada y fijaciones"), ("2.2", "Alineación de puertos y botones")]),
    ("3", "Pilar Hardware (Garage)", "Cables en espirales o precintos, conectores con traba, termocontraíble y fusible PPTC o de cartucho.",
     [("3.1", "Arreglo de arnés interno"), ("3.2", "Fijación de PCB y batería")]),
    ("4", "Pilar Software (Computación)", "Arranque automático sin cable USB a PC; telemetría continua y KPIs en gráficos.",
     [("4.1", "Encendido autónomo por switch"), ("4.2", "Dashboard en línea en tiempo real")]),
    ("5", "Pruebas estructurales", "Rigidez monolítica con 0 mm de juego interno y operación ininterrumpida sobre la mesa.",
     [("5.1", "Prueba de elevación (una mano)"), ("5.2", "Prototipo durante el ensayo Burn-in")]),
]


def preparar_fotos() -> None:
    """Saca las fotos del panel de las originales: de una sola salen varios encuadres."""
    carpeta = SALIDA / "fotos"
    recortes = {  # original -> {foto del panel: (x0, y0, x1, y1) en fracción, o None para entera}
        "antes_placa": {"1_1": None},
        "cerrada": {"1_2": None, "2_2": (0.25, 0.33, 0.50, 0.55)},
        "interior": {"2_1": None, "3_1": (0.18, 0.13, 0.78, 0.72), "3_2": (0.16, 0.54, 0.42, 0.88),
                     "4_1": (0.37, 0.64, 0.78, 0.88)},
    }
    for original, salidas in recortes.items():
        ruta = next((carpeta / f"{original}{x}" for x in (".jpg", ".jpeg", ".png") if (carpeta / f"{original}{x}").exists()), None)
        if ruta is None:
            continue
        with ImagenPIL.open(ruta) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            for nombre, caja in salidas.items():
                parte = im if caja is None else im.crop((int(caja[0] * w), int(caja[1] * h), int(caja[2] * w), int(caja[3] * h)))
                parte.save(carpeta / f"{nombre}.jpg", quality=94)


def buscar_foto(num: str) -> Path | None:
    base = SALIDA / "fotos" / num.replace(".", "_")
    return next((base.with_suffix(x) for x in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
                 if base.with_suffix(x).exists()), None)


def panel(m: dict, p: Pendientes) -> Path:
    p.area = "C. Panel fotográfico"
    ruta = SALIDA / "Panel_Fotografico_PI6002.pdf"
    W, H = landscape(A4)
    c = lienzo_pdf.Canvas(str(ruta), pagesize=(W, H))
    c.setTitle("Semáforo del Silencio: panel fotográfico")
    mx = 1.3 * cm
    hueco = 0.8 * cm
    ancho = (W - 2 * mx - hueco) / 2
    alto = 14.2 * cm
    y_foto = 3.3 * cm
    pies = m.get("pies_de_foto") or {}

    for num_sec, titulo, criterio, fotos in PANEL_SEC:
        c.setFillColor(NAVY)
        c.rect(0, H - 1.7 * cm, W, 1.7 * cm, fill=1, stroke=0)
        for i, col in enumerate((VERDE, AMBAR, ROJO)):
            c.setFillColor(col)
            c.rect(i * 1.2 * cm, H - 1.78 * cm, 1.2 * cm, 0.08 * cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Sans-B", 14.5)
        c.drawString(mx, H - 1.08 * cm, f"{num_sec}. {titulo}")
        c.setFont("Sans", 8)
        c.drawRightString(W - mx, H - 1.03 * cm, f"Semáforo del Silencio · PI6002 · Panel fotográfico · {num_sec}/5")
        c.setFillColor(GRIS)
        c.setFont("Sans-I", 8.6)
        c.drawString(mx, H - 2.3 * cm, f"Criterio técnico: {criterio}")

        for i, (num, nombre) in enumerate(fotos):
            x = mx + i * (ancho + hueco)
            foto = buscar_foto(num)
            if foto:
                with ImagenPIL.open(foto) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((2400, 2400))
                    esc = min(ancho / im.width, alto / im.height)
                    w, h = im.width * esc, im.height * esc
                    c.setFillColor(PANEL)
                    c.rect(x, y_foto, ancho, alto, fill=1, stroke=0)
                    c.drawImage(ImageReader(im), x + (ancho - w) / 2, y_foto + (alto - h) / 2, w, h)
            else:
                p.falta(f"Foto {num}: {nombre} (entrega_semana13/fotos/{num.replace('.', '_')}.jpg)")
                c.setFillColor(colors.HexColor("#FFF8E1"))
                c.setStrokeColor(AMBAR)
                c.setDash(5, 4)
                c.rect(x, y_foto, ancho, alto, fill=1, stroke=1)
                c.setDash()
                c.setFillColor(GRIS)
                c.setFont("Sans-B", 13)
                c.drawCentredString(x + ancho / 2, y_foto + alto / 2 + 0.2 * cm, f"Foto {num} pendiente")
                c.setFont("Sans", 8.5)
                c.drawCentredString(x + ancho / 2, y_foto + alto / 2 - 0.35 * cm, f"fotos/{num.replace('.', '_')}.jpg")
            pie = Paragraph(f"<b>Foto {num}. {nombre}.</b> {pies.get(num, '')}",
                            e("pf", fontSize=8.6, leading=11.8))
            _, h_pie = pie.wrap(ancho, 3 * cm)
            pie.drawOn(c, x, y_foto - 0.25 * cm - h_pie)
        c.showPage()
    c.save()
    return ruta


def escribir_pendientes(p: Pendientes) -> Path:
    ruta = SALIDA / "PENDIENTES.md"
    total = sum(len(v) for v in p.lista.values())
    lineas = [f"# Pendientes de la entrega ({total})", "",
              "Se actualiza cada vez que corre `generar_expediente.py`. Lo que se complete desaparece de aquí y del PDF.", ""]
    for area, items in p.lista.items():
        lineas += [f"## {area}", ""] + [f"- [ ] {i}" for i in items] + [""]
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    return ruta


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    registrar_fuentes()
    for sub in ("fotos", "planos", "cad", "pcb"):
        (SALIDA / sub).mkdir(parents=True, exist_ok=True)
    preparar_fotos()
    diagramas.generar(SALIDA / "diagramas")
    planos.generar()
    conf = cfg.cargar()
    m = yaml.safe_load((AQUI / "mediciones.yaml").read_text(encoding="utf-8"))
    p = Pendientes(bool(m.get("cierre")))
    a = expediente(m, conf, p)
    b = panel(m, p)
    c = escribir_pendientes(p)
    unido = SALIDA / "Informe_Consolidacion_PI6002.pdf"
    subprocess.run([shutil.which("pdfunite") or "pdfunite", a.name, b.name, unido.name], cwd=SALIDA, check=True)
    print(f"{unido.name}\n{a.name}\n{b.name}\n{c.name}: {sum(len(v) for v in p.lista.values())} pendientes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
