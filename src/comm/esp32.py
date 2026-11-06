"""Elige por dónde hablar con el ESP32: cable USB o Wi-Fi.

Lo decide esp32.transporte en config.yaml ("usb" o "wifi"). Los dos clientes
tienen la misma interfaz (enviar, girar, estadisticas, cerrar), así que el
resto del programa no sabe cuál le tocó.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402


def crear_cliente(configuracion: dict | None = None):
    conf = configuracion or cfg.cargar()
    transporte = str(conf["esp32"].get("transporte", "usb")).lower()
    if transporte in ("usb", "serie", "serial", "cable"):
        from src.comm.esp32_serial import ClienteESP32Serie

        return ClienteESP32Serie(conf)
    if transporte in ("wifi", "http", "red"):
        from src.comm.esp32_client import ClienteESP32

        return ClienteESP32(conf)
    raise ValueError(f"esp32.transporte desconocido: {transporte!r} (usa 'usb' o 'wifi')")
