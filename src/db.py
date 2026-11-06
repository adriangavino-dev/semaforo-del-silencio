"""Base de datos SQLite del proyecto.

Un solo archivo (data/semaforo.db) guarda todo lo que produce el sistema: las
sesiones con su traza segundo a segundo y sus eventos, las lecturas que envía el
ESP32 y las órdenes que se le mandan al servo. SQLite viene con Python, no hay
que instalar ni levantar nada, y el archivo se puede abrir con DB Browser for
SQLite para enseñarlo.

Los JSON de reports/ se siguen escribiendo: son lo que lee el tablero y lo que
se puede mandar por correo. La base de datos es el registro persistente y
consultable.

Ejecutable de forma independiente:
    python -m src.db                # resumen de lo guardado
    python -m src.db --sql "SELECT * FROM esp32_lecturas ORDER BY id DESC LIMIT 5"
"""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

ESQUEMA = """
CREATE TABLE IF NOT EXISTS sesiones (
    id            TEXT PRIMARY KEY,
    salon         TEXT NOT NULL,
    inicio        TEXT NOT NULL,
    fin           TEXT,
    duracion_s    REAL,
    estado_final  TEXT,
    fuente        TEXT,
    en_curso      INTEGER NOT NULL DEFAULT 0,
    resumen_json  TEXT
);
CREATE TABLE IF NOT EXISTS lecturas (
    sesion_id     TEXT NOT NULL REFERENCES sesiones(id),
    timestamp     TEXT NOT NULL,
    segundo       REAL NOT NULL,
    estado        TEXT,
    clase_audio   TEXT,
    nivel_db      REAL,
    n_personas    INTEGER,
    n_de_pie      INTEGER,
    posicion      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_lecturas_sesion ON lecturas(sesion_id, segundo);
CREATE TABLE IF NOT EXISTS eventos (
    sesion_id     TEXT NOT NULL REFERENCES sesiones(id),
    timestamp     TEXT NOT NULL,
    segundo       REAL NOT NULL,
    tipo          TEXT NOT NULL,
    zona          TEXT,
    posicion      INTEGER NOT NULL DEFAULT 0,
    detalle_json  TEXT
);
CREATE TABLE IF NOT EXISTS cambios_estado (
    sesion_id     TEXT NOT NULL REFERENCES sesiones(id),
    timestamp     TEXT NOT NULL,
    de            TEXT NOT NULL,
    a             TEXT NOT NULL,
    motivo        TEXT,
    clase_audio   TEXT,
    nivel_db      REAL,
    evento        TEXT
);
CREATE INDEX IF NOT EXISTS ix_cambios_sesion ON cambios_estado(sesion_id, timestamp);
CREATE TABLE IF NOT EXISTS esp32_lecturas (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    recibido       TEXT NOT NULL,
    nivel_db       REAL,
    estado_led     TEXT,
    posicion_servo INTEGER,
    uptime_s       INTEGER,
    raw_json       TEXT
);
CREATE INDEX IF NOT EXISTS ix_esp32_recibido ON esp32_lecturas(recibido);
CREATE TABLE IF NOT EXISTS servo_ordenes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    posicion      INTEGER NOT NULL,
    ok            INTEGER NOT NULL,
    error         TEXT
);
"""

_lock = threading.Lock()


def ruta() -> Path:
    return cfg.ruta_de("base_datos")


def conectar() -> sqlite3.Connection:
    """Abre la base creando el esquema si hace falta. Cada llamada, su conexión."""
    destino = ruta()
    destino.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(destino), timeout=5.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(ESQUEMA)
    return con


def _ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------- sesiones
def guardar_sesion(resumen: dict, filas: list[dict], eventos: list[dict], en_curso: bool = False) -> None:
    """Escribe (o reemplaza) una sesión completa con su traza y sus eventos."""
    with _lock, conectar() as con:
        con.execute(
            "INSERT OR REPLACE INTO sesiones (id, salon, inicio, fin, duracion_s, estado_final, fuente, en_curso, resumen_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                resumen["sesion_id"], resumen["salon"], resumen["inicio"], resumen.get("fin"),
                resumen.get("duracion_s"), resumen.get("estado_final"), resumen.get("fuente"),
                int(en_curso), json.dumps(resumen, ensure_ascii=False),
            ),
        )
        con.execute("DELETE FROM lecturas WHERE sesion_id = ?", (resumen["sesion_id"],))
        con.execute("DELETE FROM eventos WHERE sesion_id = ?", (resumen["sesion_id"],))
        con.execute("DELETE FROM cambios_estado WHERE sesion_id = ?", (resumen["sesion_id"],))
        con.executemany(
            "INSERT INTO cambios_estado (sesion_id, timestamp, de, a, motivo, clase_audio, nivel_db, evento)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    resumen["sesion_id"], c["timestamp"], c["de"], c["a"], c.get("motivo"),
                    c.get("clase_audio"), c.get("nivel_db"), c.get("evento"),
                )
                for c in resumen.get("cambios_estado", [])
            ],
        )
        con.executemany(
            "INSERT INTO lecturas (sesion_id, timestamp, segundo, estado, clase_audio, nivel_db, n_personas, n_de_pie, posicion)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    resumen["sesion_id"], f["timestamp"], f["segundo"], f["estado"],
                    f["clase_audio"] or None, f["nivel_db"], f["n_personas"],
                    None if f["n_de_pie"] == "" else f["n_de_pie"], f.get("posicion", 0),
                )
                for f in filas
            ],
        )
        con.executemany(
            "INSERT INTO eventos (sesion_id, timestamp, segundo, tipo, zona, posicion, detalle_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    resumen["sesion_id"], e["timestamp"], e["segundo"], e["tipo"], e.get("zona") or None,
                    e.get("posicion", 0),
                    json.dumps({k: v for k, v in e.items() if k not in ("timestamp", "segundo", "tipo", "zona", "posicion")},
                               ensure_ascii=False),
                )
                for e in eventos
            ],
        )


def listar_sesiones(limite: int = 100) -> list[dict]:
    with conectar() as con:
        filas = con.execute(
            "SELECT id, salon, inicio, fin, duracion_s, estado_final, fuente, en_curso FROM sesiones"
            " ORDER BY inicio DESC LIMIT ?", (limite,)
        ).fetchall()
    return [dict(f) for f in filas]


# ------------------------------------------------------------- ESP32
def guardar_lectura_esp32(datos: dict) -> int:
    """Guarda una lectura enviada por el ESP32. Devuelve su id."""
    with _lock, conectar() as con:
        cur = con.execute(
            "INSERT INTO esp32_lecturas (recibido, nivel_db, estado_led, posicion_servo, uptime_s, raw_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                _ahora(),
                _num(datos.get("nivel_db")), datos.get("estado_led"),
                _int(datos.get("posicion_servo")), _int(datos.get("uptime_s")),
                json.dumps(datos, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def ultima_lectura_esp32() -> dict | None:
    with conectar() as con:
        f = con.execute("SELECT * FROM esp32_lecturas ORDER BY id DESC LIMIT 1").fetchone()
    return dict(f) if f else None


def lecturas_esp32(limite: int = 300) -> list[dict]:
    with conectar() as con:
        filas = con.execute(
            "SELECT recibido, nivel_db, estado_led, posicion_servo FROM esp32_lecturas ORDER BY id DESC LIMIT ?",
            (limite,),
        ).fetchall()
    return [dict(f) for f in reversed(filas)]


def registrar_orden_servo(posicion: int, ok: bool, error: str | None = None) -> None:
    with _lock, conectar() as con:
        con.execute(
            "INSERT INTO servo_ordenes (timestamp, posicion, ok, error) VALUES (?, ?, ?, ?)",
            (_ahora(), int(posicion), int(ok), error),
        )


# ------------------------------------------------------------- utilidades
def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Consulta la base de datos del proyecto")
    ap.add_argument("--sql", help="ejecutar una consulta SELECT y mostrarla")
    args = ap.parse_args()

    if args.sql:
        with conectar() as con:
            for fila in con.execute(args.sql).fetchall():
                print(dict(fila))
        raise SystemExit(0)

    print(f"Base de datos: {ruta()}")
    with conectar() as con:
        for tabla in ("sesiones", "lecturas", "eventos", "cambios_estado", "esp32_lecturas", "servo_ordenes"):
            n = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
            print(f"  {tabla:<16} {n:>7} filas")
    ultima = ultima_lectura_esp32()
    if ultima:
        print(f"\nÚltima lectura del ESP32: {ultima['recibido']}  {ultima['nivel_db']} dB  "
              f"LED {ultima['estado_led']}  servo {ultima['posicion_servo']}")
