"""Reporte de sesión — sección 7.

Acumula lo que va pasando durante una sesión y lo escribe en su propia carpeta,
sesiones/<fecha>_sesion_<n>/ (por ejemplo sesiones/2026-11-02_sesion_01/):

    reporte.json              resumen; es el que lee el tablero web
    lecturas_por_segundo.csv  la traza segundo a segundo
    eventos.csv               movimientos anómalos, dormidos, ruidos bruscos
    cambios_estado.csv        cada cambio de color, con su causa
    fotos/, video.mp4, audio.wav   (run_session.py, ver src/fusion/grabacion.py)

Contenido mínimo exigido: duración de la sesión, porcentaje de tiempo en cada
estado, distribución de clases de audio, número de eventos por tipo y zona, y
promedio de personas de pie por intervalo de cinco minutos.

La grabación de audio y video quedó autorizada (los padres están informados),
lo que levanta la restricción 2 original; se apaga en la sección grabacion.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.fusion.state import ESTADOS  # noqa: E402

def carpeta_sesiones() -> Path:
    return cfg.ruta_de("sesiones")


def nuevo_id_sesion(inicio: datetime) -> str:
    """<fecha>_sesion_<n>: n cuenta las sesiones de ese día, empezando en 01."""
    fecha = inicio.strftime("%Y-%m-%d")
    usados = []
    for c in carpeta_sesiones().glob(f"{fecha}_sesion_*"):
        sufijo = c.name.rsplit("_", 1)[-1]
        if sufijo.isdigit():
            usados.append(int(sufijo))
    return f"{fecha}_sesion_{max(usados, default=0) + 1:02d}"


COLUMNAS_CSV = [
    "timestamp", "segundo", "estado", "clase_audio", "confianza_audio",
    "nivel_db", "n_personas", "n_de_pie", "n_dormidos", "evento", "zona_evento", "posicion",
]


class Reporte:
    """Acumulador de una sesión y escritor de los dos archivos de salida."""

    def __init__(self, configuracion: dict | None = None, salon: str | None = None) -> None:
        self.conf = configuracion or cfg.cargar()
        self.salon = salon or self.conf["proyecto"]["salon"]
        self.intervalo_min = int(self.conf["fusion"]["intervalo_resumen_min"])

        self.inicio = datetime.now()
        # La carpeta se crea ya, para reservar el número del día y para que la
        # grabación de video y las fotos tengan dónde ir desde el primer segundo.
        self.sesion_id = nuevo_id_sesion(self.inicio)
        self.carpeta = carpeta_sesiones() / self.sesion_id
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.filas: list[dict] = []
        self.eventos: list[dict] = []
        self.cambios_estado: list[dict] = []
        # Al procesar un video grabado, el tiempo de la sesión es el del video y
        # no el del reloj: el archivo se analiza más rápido que en tiempo real.
        self.duracion_forzada: float | None = None

    def _marca(self, segundo: float | None) -> tuple[str, float]:
        """Devuelve (timestamp, segundo) según el reloj o el tiempo del video."""
        if segundo is None:
            ahora = datetime.now()
            return ahora.isoformat(timespec="seconds"), round((ahora - self.inicio).total_seconds(), 1)
        instante = self.inicio + timedelta(seconds=segundo)
        return instante.isoformat(timespec="seconds"), round(segundo, 1)

    # ------------------------------------------------------------------
    def registrar(
        self,
        estado: str,
        clase_audio: str | None,
        confianza_audio: float,
        nivel_db: float,
        n_personas: int,
        n_de_pie: int | None,
        evento: str | None = None,
        zona_evento: str | None = None,
        segundo: float | None = None,
        posicion: int = 0,
        n_dormidos: int = 0,
    ) -> None:
        timestamp, seg = self._marca(segundo)
        self.filas.append({
            "timestamp": timestamp,
            "segundo": seg,
            "estado": estado,
            "clase_audio": clase_audio or "",
            "confianza_audio": round(float(confianza_audio), 3),
            "nivel_db": round(float(nivel_db), 1),
            "n_personas": int(n_personas),
            "n_de_pie": "" if n_de_pie is None else int(n_de_pie),
            "n_dormidos": int(n_dormidos),
            "evento": evento or "",
            "zona_evento": zona_evento or "",
            "posicion": int(posicion),
        })

    def registrar_evento(
        self, tipo: str, zona: str | None = None, detalle: dict | None = None,
        segundo: float | None = None, posicion: int = 0,
    ) -> None:
        """Un evento puntual: ruido brusco o movimiento anómalo."""
        timestamp, seg = self._marca(segundo)
        self.eventos.append({
            "timestamp": timestamp,
            "segundo": seg,
            "tipo": tipo,
            "zona": zona or "",
            "posicion": int(posicion),
            **(detalle or {}),
        })

    # ------------------------------------------------------------------
    @property
    def duracion_s(self) -> float:
        if self.duracion_forzada is not None:
            return self.duracion_forzada
        return (datetime.now() - self.inicio).total_seconds()

    def _porcentaje_estados(self) -> dict[str, float]:
        if not self.filas:
            return {e: 0.0 for e in ESTADOS}
        cuenta = Counter(f["estado"] for f in self.filas)
        total = len(self.filas)
        return {e: round(100.0 * cuenta.get(e, 0) / total, 1) for e in ESTADOS}

    def _distribucion_audio(self) -> dict[str, float]:
        clases = [f["clase_audio"] for f in self.filas if f["clase_audio"]]
        if not clases:
            return {}
        cuenta = Counter(clases)
        return {c: round(100.0 * n / len(clases), 1) for c, n in cuenta.most_common()}

    def _eventos_por_tipo_y_zona(self) -> dict[str, dict[str, int]]:
        salida: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in self.eventos:
            salida[ev["tipo"]][ev["zona"] or "sin_zona"] += 1
        return {k: dict(v) for k, v in salida.items()}

    def _de_pie_por_intervalo(self) -> list[dict]:
        """Promedio de personas de pie por bloques de N minutos."""
        bloques: dict[int, list[int]] = defaultdict(list)
        for f in self.filas:
            if f["n_de_pie"] == "":
                continue
            bloque = int(f["segundo"] // (self.intervalo_min * 60))
            bloques[bloque].append(int(f["n_de_pie"]))
        return [
            {
                "intervalo": f"{b * self.intervalo_min}-{(b + 1) * self.intervalo_min} min",
                "promedio_de_pie": round(sum(v) / len(v), 2),
                "maximo_de_pie": max(v),
            }
            for b, v in sorted(bloques.items())
        ]

    def _serie_temporal(self, maximo: int = 240) -> list[dict]:
        """Traza reducida para graficar en el tablero, sin mandar miles de puntos."""
        if not self.filas:
            return []
        paso = max(1, len(self.filas) // maximo)
        filas = self.filas[::paso]
        if filas[-1] is not self.filas[-1]:
            filas.append(self.filas[-1])   # el último punto siempre es el más reciente
        return [
            {
                "segundo": f["segundo"],
                "estado": f["estado"],
                "nivel_db": f["nivel_db"],
                "clase_audio": f["clase_audio"],
                "n_personas": f["n_personas"],
                "n_de_pie": f["n_de_pie"] if f["n_de_pie"] != "" else None,
            }
            for f in filas
        ]

    # ------------------------------------------------------------------
    def resumen(self, estado_final: str, extra: dict | None = None) -> dict:
        personas = [f["n_personas"] for f in self.filas]
        de_pie = [int(f["n_de_pie"]) for f in self.filas if f["n_de_pie"] != ""]
        db = [f["nivel_db"] for f in self.filas]

        return {
            "salon": self.salon,
            "sesion_id": self.sesion_id,
            "carpeta": f"sesiones/{self.sesion_id}",
            "inicio": self.inicio.isoformat(timespec="seconds"),
            "fin": (self.inicio + timedelta(seconds=self.duracion_s)).isoformat(timespec="seconds"),
            "duracion_s": round(self.duracion_s, 1),
            "estado_final": estado_final,
            "porcentaje_estados": self._porcentaje_estados(),
            "audio": {
                "distribucion_clases": self._distribucion_audio(),
                "nivel_db_promedio": round(sum(db) / len(db), 1) if db else 0.0,
                "nivel_db_maximo": round(max(db), 1) if db else 0.0,
                "n_lecturas": len(self.filas),
            },
            "camara": {
                "personas_promedio": round(sum(personas) / len(personas), 2) if personas else 0.0,
                "personas_maximo": max(personas) if personas else 0,
                "de_pie_promedio": round(sum(de_pie) / len(de_pie), 2) if de_pie else None,
                "de_pie_maximo": max(de_pie) if de_pie else None,
                "de_pie_por_intervalo": self._de_pie_por_intervalo(),
                "dormidos_maximo": max((int(f.get("n_dormidos", 0)) for f in self.filas), default=0),
            },
            "eventos": {
                "total": len(self.eventos),
                "por_tipo_y_zona": self._eventos_por_tipo_y_zona(),
                "detalle": self.eventos[-50:],
            },
            "cambios_estado": self.cambios_estado,
            "serie": self._serie_temporal(),
            **(extra or {}),
        }

    def guardar_parcial(self, estado_actual: str, extra: dict | None = None) -> Path:
        """Escribe el JSON mientras la sesión sigue corriendo.

        Permite que el tablero muestre la sesión en curso en lugar de esperar a
        que termine. Se escribe primero en un archivo temporal y luego se cambia
        el nombre, para que el tablero nunca lea un JSON a medio escribir.
        """
        datos = self.resumen(estado_actual, extra)
        datos["en_curso"] = True

        destino = self.carpeta / "reporte.json"
        temporal = self.carpeta / "reporte.json.tmp"
        temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporal, destino)
        self._a_base_de_datos(datos, en_curso=True)
        return destino

    def _a_base_de_datos(self, datos: dict, en_curso: bool) -> None:
        """Copia la sesión a SQLite. Un fallo aquí no debe tumbar la sesión."""
        try:
            from src import db

            db.guardar_sesion(datos, self.filas, self.eventos, en_curso=en_curso)
        except Exception as exc:  # noqa: BLE001
            print(f"[reporte] no se pudo escribir en la base de datos: {exc}", file=sys.stderr)

    def guardar(self, estado_final: str, extra: dict | None = None) -> tuple[Path, Path]:
        """Escribe los CSV y el JSON en la carpeta de la sesión. Devuelve (CSV de lecturas, JSON)."""
        ruta_csv = self.carpeta / "lecturas_por_segundo.csv"
        _escribir_csv(ruta_csv, self.filas, COLUMNAS_CSV)
        _escribir_csv(self.carpeta / "eventos.csv", self.eventos,
                      ["timestamp", "segundo", "tipo", "zona", "posicion", "foto", "puntaje", "id", "nivel_db"])
        _escribir_csv(self.carpeta / "cambios_estado.csv", self.cambios_estado,
                      ["timestamp", "de", "a", "motivo", "clase_audio", "nivel_db", "evento", "foto"])

        ruta_json = self.carpeta / "reporte.json"
        datos = self.resumen(estado_final, extra)
        ruta_json.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        self._a_base_de_datos(datos, en_curso=False)

        return ruta_csv, ruta_json


def _escribir_csv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    with ruta.open("w", encoding="utf-8-sig", newline="") as fh:   # -sig: Excel lee bien los acentos
        escritor = csv.DictWriter(fh, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)


def listar_reportes(carpeta: Path | None = None) -> list[dict]:
    """Todos los resúmenes de sesión guardados, del más reciente al más antiguo.

    Lee sesiones/*/reporte.json y también los de antes del cambio a carpetas
    (reports/sesion_*.json), para no perder el historial.
    """
    rutas = list((carpeta or carpeta_sesiones()).glob("*/reporte.json"))
    if carpeta is None:
        rutas += list(cfg.ruta_de("reports").glob("sesion_*.json"))
    salida = []
    for ruta in rutas:
        try:
            salida.append(json.loads(ruta.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[reporte] no se pudo leer {ruta}: {exc}", file=sys.stderr)
    salida.sort(key=lambda r: str(r.get("inicio", "")), reverse=True)
    return salida


if __name__ == "__main__":
    reportes = listar_reportes()
    if not reportes:
        print("Todavía no hay reportes de sesión en sesiones/.")
        raise SystemExit(0)
    print(f"{len(reportes)} sesiones guardadas\n")
    for r in reportes:
        print(
            f"  {r['sesion_id']}  {r['salon']:<12} {r['duracion_s'] / 60:5.1f} min  "
            f"final={r['estado_final']:<9} eventos={r['eventos']['total']}"
        )
