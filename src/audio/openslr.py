"""Voz en español peruano para el clasificador de audio (OpenSLR 73).

Dataset: "Crowdsourced high-quality Peruvian Spanish speech data set"
(Guevara-Rukoz et al., LREC 2020; openslr.org/73; licencia CC BY-SA 4.0).
Personas leyendo frases en voz alta, una por grabación.

Por qué hace falta: con solo FSD50K, `voz_docente` era la clase más débil
(~70 % de acierto) y casi toda la voz era en inglés. Aquí se añade:

    voz_docente   bloques de 1 s de una sola persona hablando en español
    murmullo      mezclas de 3 a 8 hablantes peruanos distintos a la vez
                  ("babble" en español, como el de un aula)

A cada bloque se le aplica un nivel al azar y un poco de ruido de sala, para
que no se parezca a una grabación de estudio sino a un micrófono en un aula.

La "sesión" de cada bloque es un grupo (0-9) calculado a partir del hablante:
el mismo hablante nunca está en entrenamiento y en prueba a la vez. Los grupos
comparten nombre con los de FSD50K (fsd_g0 ... fsd_g9) para que cada grupo
tenga las cuatro clases.

Uso:
    python -m src.audio.openslr --importar
    python -m src.audio.train --recalcular
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

RAIZ_SLR = Path(r"C:\Users\Adrian\datasets\openslr73")
PREFIJO = "slr"
N_GRUPOS = 10


def grupo_de(hablante: str) -> str:
    return f"fsd_g{int(hashlib.md5(hablante.encode('utf-8')).hexdigest(), 16) % N_GRUPOS}"


def wavs_por_hablante() -> dict[str, list]:
    """{hablante: [(zip, nombre_wav), ...]} sin descomprimir los zip en disco."""
    salida: dict[str, list] = defaultdict(list)
    for z in sorted(RAIZ_SLR.glob("es_pe_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                m = re.search(r"([a-z]{3}_\d{5})_\d+\.wav$", n)
                if m:
                    salida[m.group(1)].append((z, n))
    return salida


def _leer(zf_cache: dict, z: Path, nombre: str, sr: int) -> np.ndarray:
    import io

    import librosa
    import soundfile as sf

    if z not in zf_cache:
        zf_cache[z] = zipfile.ZipFile(z)
    datos, sr0 = sf.read(io.BytesIO(zf_cache[z].read(nombre)), dtype="float32", always_2d=False)
    if datos.ndim > 1:
        datos = datos.mean(axis=1)
    return librosa.resample(datos, orig_sr=sr0, target_sr=sr) if sr0 != sr else datos


def _como_aula(bloque: np.ndarray, rng) -> np.ndarray:
    """Nivel al azar (-38 a -18 dBFS) + ruido de sala suave."""
    rms = np.sqrt(np.mean(bloque ** 2)) + 1e-9
    b = bloque / rms * 10 ** (rng.uniform(-38, -18) / 20)
    ruido = rng.standard_normal(len(b)).astype(np.float32) * 10 ** (rng.uniform(-62, -50) / 20)
    return np.clip(b + ruido, -1, 1).astype(np.float32)


def importar(n_voz: int, n_murmullo: int, por_hablante: int = 45, semilla: int = 7) -> None:
    import soundfile as sf

    from src.audio.dataset import escribir_etiquetas, leer_etiquetas
    from src.audio.fsd50k import bloques_de, dbfs

    sr = int(cfg.cargar()["audio"]["sample_rate"])
    rng = np.random.default_rng(semilla)
    hablantes = wavs_por_hablante()
    if not hablantes:
        raise SystemExit(f"No hay es_pe_*.zip en {RAIZ_SLR}")
    lista = sorted(hablantes)
    rng.shuffle(lista)
    print(f"{len(lista)} hablantes, {sum(len(v) for v in hablantes.values())} grabaciones")

    carpeta = cfg.ruta_de("data_audio_clips")
    etiquetas = {k: v for k, v in leer_etiquetas().items() if not k.startswith(PREFIJO + "_")}
    zfs: dict = {}

    # ---- voz sola: pocos bloques de muchos hablantes (variedad de voces)
    voz_por_grupo: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    n = 0
    for h in lista:
        if n >= n_voz:
            break
        archivos = hablantes[h][:]
        rng.shuffle(archivos)
        tomados = 0
        for z, nombre in archivos[:25]:
            onda = _leer(zfs, z, nombre, sr)
            for b in bloques_de(onda, sr, 2):
                if dbfs(b) < -45 or tomados >= por_hablante or n >= n_voz:
                    continue
                grupo = grupo_de(h)
                voz_por_grupo[grupo].append((h, b))
                clip_id = f"{PREFIJO}_voz_{h}_{tomados}"
                destino = carpeta / grupo / f"{clip_id}.wav"
                destino.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destino), _como_aula(b, rng), sr, subtype="PCM_16")
                etiquetas[clip_id] = {"clip_id": clip_id, "clase": "voz_docente", "sesion": grupo}
                tomados += 1
                n += 1
    print(f"voz_docente: {n} bloques de 1 s")

    # ---- murmullo en español: 5-8 hablantes distintos del mismo grupo a la vez
    # 38 hablantes en 10 grupos: hay grupos con pocas voces. Se mezclan de 3 a 8,
    # las que haya en el grupo (3 personas hablando a la vez ya es murmullo).
    hab_grupo = {g: len({h for h, _ in v}) for g, v in voz_por_grupo.items()}
    grupos = sorted(g for g, n_h in hab_grupo.items() if n_h >= 3)
    m = 0
    for k in range(n_murmullo if grupos else 0):
        g = grupos[k % len(grupos)]
        candidatos = voz_por_grupo[g]
        n_voces = int(rng.integers(3, min(8, hab_grupo[g]) + 1))
        elegidos, usados = [], set()
        for i in rng.permutation(len(candidatos)):
            h, b = candidatos[i]
            if h not in usados:
                elegidos.append(b); usados.add(h)
            if len(elegidos) == n_voces:
                break
        mezcla = sum(b / (np.sqrt(np.mean(b ** 2)) + 1e-6) * rng.uniform(0.5, 1.0) for b in elegidos)
        clip_id = f"{PREFIJO}_babble_{k:05d}"
        destino = carpeta / g / f"{clip_id}.wav"
        sf.write(str(destino), _como_aula(mezcla.astype(np.float32), rng), sr, subtype="PCM_16")
        etiquetas[clip_id] = {"clip_id": clip_id, "clase": "murmullo", "sesion": g}
        m += 1
    print(f"murmullo (español): {m} bloques de 1 s")
    escribir_etiquetas(etiquetas)
    print(f"labels.csv: {len(etiquetas)} fragmentos en total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Importa voz peruana (OpenSLR 73)")
    ap.add_argument("--importar", action="store_true")
    ap.add_argument("--voz", type=int, default=1500)
    ap.add_argument("--murmullo", type=int, default=1000)
    a = ap.parse_args()
    if a.importar:
        importar(a.voz, a.murmullo)
    else:
        ap.print_help()
