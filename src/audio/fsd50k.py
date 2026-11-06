"""Importa FSD50K como conjunto de entrenamiento del clasificador de audio.

No existe un dataset público de audio de aulas etiquetado con las cuatro clases
del proyecto (los trabajos recientes, p. ej. SimClass y RealClass, lo dicen
explícitamente). FSD50K es lo más cercano: 51 197 clips de Freesound, licencia
CC-BY, etiquetados con la misma ontología de AudioSet con la que se entrenó
YAMNet. Se descarga con la librería soundata.

Correspondencia con las clases del proyecto (sección 5.1):

    voz_docente   una sola voz adulta: Male_speech / Female_speech / Speech,
                  sin multitud, conversación, música ni gritos
    murmullo      varias voces a la vez: Chatter / Crowd / Conversation
    ruido_brusco  golpes: Slam / Knock / Thump_and_thud / Drawer / Door; solo
                  los 1-2 bloques impulsivos de cada clip (factor de cresta >= 5)
    silencio      FSD50K no tiene esta clase: se toman tramos de 1 s de muy
                  bajo nivel de cualquier clip, más ruido de sala sintético

Murmullo sintético: el conjunto de evaluación de FSD50K solo trae ~150 clips de
multitud. Se completa mezclando de 5 a 8 voces de usuarios distintos, que es
como se fabrica el "babble noise" en la investigación de reconocimiento de voz.
Las voces mezcladas solo se toman de un mismo grupo, para no filtrar usuarios
entre entrenamiento y prueba.

"Sesión" (lo que separa entrenamiento y prueba): el usuario de Freesound que
subió el clip, agrupado en 10 grupos. Así la misma persona —mismo micrófono,
misma sala— nunca aparece a la vez en entrenamiento y en prueba, que es el
equivalente honesto de separar por sesión de grabación.

Limitación declarada: son grabaciones de Freesound, no de un aula. "voz_docente"
aquí es una voz que habla sola, no necesariamente un docente. Conviene sumar
grabaciones reales del aula (record_session.py) cuando las haya.

Uso:
    python -m src.audio.fsd50k --contar                      # solo cuántos clips hay por clase
    python -m src.audio.fsd50k --importar                    # escribe clips de 1 s y labels.csv
    python -m src.audio.fsd50k --importar --max-por-clase 1500
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

RAIZ_FSD = Path(r"C:\Users\Adrian\datasets\fsd50k")
N_GRUPOS = 10
PREFIJO = "fsd"

VOZ_SOLA = {"Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Speech"}
MULTITUD = {"Chatter", "Crowd", "Conversation"}
GOLPES = {"Slam", "Knock", "Thump_and_thud", "Drawer_open_or_close", "Door"}
# Si aparece cualquiera de estas, el clip es ambiguo para nuestras clases.
EXCLUIR = {
    "Music", "Musical_instrument", "Singing", "Applause", "Cheering", "Laughter",
    "Screaming", "Shout", "Yell", "Speech_synthesizer", "Child_speech_and_kid_speaking",
    "Vehicle", "Animal", "Explosion", "Gunshot_and_gunfire", "Fireworks",
}


def clase_de(etiquetas: set[str]) -> str | None:
    if etiquetas & EXCLUIR:
        return None
    if etiquetas & MULTITUD:
        return "murmullo"
    if etiquetas & VOZ_SOLA:
        return None if etiquetas & GOLPES else "voz_docente"
    if etiquetas & GOLPES:
        return "ruido_brusco"
    return None


def leer_clips(particion: str) -> list[dict]:
    """[{fname, etiquetas, clase, uploader, particion}] de dev o eval."""
    gt = RAIZ_FSD / "FSD50K.ground_truth" / f"{particion}.csv"
    info = json.loads((RAIZ_FSD / "FSD50K.metadata" / f"{particion}_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    salida = []
    with gt.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            etiquetas = set(fila["labels"].split(","))
            salida.append({
                "fname": fila["fname"],
                "etiquetas": etiquetas,
                "clase": clase_de(etiquetas),
                "uploader": info.get(fila["fname"], {}).get("uploader", "desconocido"),
                "particion": particion,
            })
    return salida


def grupo_de(uploader: str) -> str:
    """Grupo estable (0-9) por usuario de Freesound: nunca se reparte entre grupos."""
    h = int(hashlib.md5(uploader.encode("utf-8")).hexdigest(), 16)
    return f"{PREFIJO}_g{h % N_GRUPOS}"


def ruta_audio(clip: dict) -> Path:
    return RAIZ_FSD / f"FSD50K.{clip['particion']}_audio" / f"{clip['fname']}.wav"


def contar() -> None:
    for particion in ("eval", "dev"):
        clips = leer_clips(particion)
        cuenta = Counter(c["clase"] for c in clips if c["clase"])
        uploaders = defaultdict(set)
        for c in clips:
            if c["clase"]:
                uploaders[c["clase"]].add(c["uploader"])
        print(f"{particion}: {len(clips)} clips")
        for clase, n in sorted(cuenta.items()):
            print(f"  {clase:<14} {n:>5} clips de {len(uploaders[clase]):>4} usuarios distintos")


def bloques_de(onda: np.ndarray, sr: int, max_bloques: int) -> list[np.ndarray]:
    """Bloques de 1 s sin solape, del más fuerte al más débil (los fuertes son los informativos)."""
    n = len(onda) // sr
    bloques = [onda[i * sr:(i + 1) * sr] for i in range(n)]
    if not bloques and len(onda) >= sr // 2:            # clips de 0,5-1 s: se rellenan
        bloques = [np.pad(onda, (0, sr - len(onda)))]
    bloques.sort(key=lambda b: -float(np.sqrt(np.mean(b ** 2))))
    return bloques[:max_bloques]


def dbfs(bloque: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(bloque ** 2)), 1e-9)))


def es_impulsivo(bloque: np.ndarray, factor: float = 5.0) -> bool:
    """Golpe: el pico supera `factor` veces el nivel medio (factor de cresta).

    Un portazo dura menos de 1 s; los demás segundos del clip son silencio,
    ambiente o voces. Sin este filtro esos segundos quedaban etiquetados como
    golpe y el modelo confundía la voz con ruido brusco (27 % de la voz).
    """
    rms = np.sqrt(np.mean(bloque ** 2)) + 1e-9
    return float(np.abs(bloque).max() / rms) >= factor


def importar(max_por_clase: int, bloques_por_clip: int, semilla: int = 42) -> None:
    import librosa
    import soundfile as sf

    from src.audio.dataset import escribir_etiquetas, leer_etiquetas

    conf = cfg.cargar()
    sr = int(conf["audio"]["sample_rate"])
    rng = np.random.default_rng(semilla)
    carpeta_clips = cfg.ruta_de("data_audio_clips")

    clips = [c for p in ("eval", "dev") for c in leer_clips(p) if ruta_audio(c).exists()]
    if not clips:
        raise SystemExit(f"No hay audio de FSD50K en {RAIZ_FSD}. Descárgalo primero (ver docstring).")

    # Equilibrio: como mucho max_por_clase bloques por clase, repartidos entre clips.
    por_clase: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        if c["clase"]:
            por_clase[c["clase"]].append(c)
    for lista in por_clase.values():
        rng.shuffle(lista)

    etiquetas = {k: v for k, v in leer_etiquetas().items() if not k.startswith(PREFIJO + "_")}
    escritos: Counter[str] = Counter()
    silencios: list[tuple[np.ndarray, str]] = []

    for clase, lista in sorted(por_clase.items()):
        print(f"{clase}: {len(lista)} clips disponibles")
        for c in lista:
            if escritos[clase] >= max_por_clase:
                break
            onda, _ = librosa.load(str(ruta_audio(c)), sr=sr, mono=True)
            grupo = grupo_de(c["uploader"])
            # Tramos casi mudos de cualquier clip: candidatos a "silencio".
            for i in range(len(onda) // sr):
                b = onda[i * sr:(i + 1) * sr]
                if dbfs(b) < -55 and len(silencios) < max_por_clase:
                    silencios.append((b, grupo))
            # Golpes: 1-2 bloques (el resto del clip no es golpe). Voz: hasta el doble,
            # porque todo el clip es voz y es la clase con menos clips.
            maximo = {"ruido_brusco": min(2, bloques_por_clip),
                      "voz_docente": 2 * bloques_por_clip}.get(clase, bloques_por_clip)
            for k, b in enumerate(bloques_de(onda, sr, maximo)):
                if dbfs(b) < -45:          # demasiado bajo para ser la clase del clip
                    continue
                if clase == "ruido_brusco" and not es_impulsivo(b):
                    continue
                clip_id = f"{PREFIJO}_{c['particion']}_{c['fname']}_{k}"
                destino = carpeta_clips / grupo / f"{clip_id}.wav"
                destino.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destino), b, sr, subtype="PCM_16")
                etiquetas[clip_id] = {"clip_id": clip_id, "clase": clase, "sesion": grupo}
                escritos[clase] += 1
        print(f"  -> {escritos[clase]} bloques de 1 s")

    # Murmullo sintético: varias voces de distintos usuarios a la vez.
    faltan = max(0, max_por_clase - escritos["murmullo"])
    voces_por_grupo: dict[str, list[dict]] = defaultdict(list)
    for c in por_clase.get("voz_docente", []):
        voces_por_grupo[grupo_de(c["uploader"])].append(c)
    grupos = [g for g, v in voces_por_grupo.items() if len({c["uploader"] for c in v}) >= 5]
    cache_ondas: dict[str, np.ndarray] = {}

    def onda_de(c: dict) -> np.ndarray:
        if c["fname"] not in cache_ondas:
            cache_ondas[c["fname"]], _ = librosa.load(str(ruta_audio(c)), sr=sr, mono=True)
        return cache_ondas[c["fname"]]

    for k in range(faltan if grupos else 0):
        grupo = grupos[k % len(grupos)]
        candidatas = voces_por_grupo[grupo]
        n_voces = int(rng.integers(5, 9))
        elegidas, usados = [], set()
        for i in rng.permutation(len(candidatas)):
            if candidatas[i]["uploader"] not in usados:
                elegidas.append(candidatas[i]); usados.add(candidatas[i]["uploader"])
            if len(elegidas) == n_voces:
                break
        if len(elegidas) < 5:
            continue
        mezcla = np.zeros(sr, dtype=np.float32)
        for c in elegidas:
            o = onda_de(c)
            if len(o) < sr:
                o = np.pad(o, (0, sr - len(o)))
            ini = int(rng.integers(0, len(o) - sr + 1))
            trozo = o[ini:ini + sr]
            mezcla += trozo / (np.sqrt(np.mean(trozo ** 2)) + 1e-6) * rng.uniform(0.5, 1.0)
        mezcla = (mezcla / (np.abs(mezcla).max() + 1e-6) * rng.uniform(0.2, 0.6)).astype(np.float32)
        clip_id = f"{PREFIJO}_babble_{k:05d}"
        destino = carpeta_clips / grupo / f"{clip_id}.wav"
        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), mezcla, sr, subtype="PCM_16")
        etiquetas[clip_id] = {"clip_id": clip_id, "clase": "murmullo", "sesion": grupo}
        escritos["murmullo"] += 1
    print(f"murmullo: {escritos['murmullo']} bloques en total (incluye sintéticos)")

    # Silencio: tramos reales casi mudos + ruido de sala sintético (-65 a -50 dBFS).
    n_sinteticos = max(0, max_por_clase - len(silencios))
    for i in range(n_sinteticos):
        nivel = 10 ** (rng.uniform(-65, -50) / 20)
        blanco = rng.standard_normal(sr)
        rosa = np.cumsum(blanco)                             # ruido grave tipo aire acondicionado
        rosa = (rosa - np.convolve(rosa, np.ones(400) / 400, mode="same"))
        ruido = 0.5 * blanco / np.std(blanco) + 0.5 * rosa / (np.std(rosa) + 1e-9)
        silencios.append(((nivel * ruido / np.std(ruido)).astype(np.float32), f"{PREFIJO}_g{i % N_GRUPOS}"))
    for k, (b, grupo) in enumerate(silencios[:max_por_clase]):
        clip_id = f"{PREFIJO}_silencio_{k:05d}"
        destino = carpeta_clips / grupo / f"{clip_id}.wav"
        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), b, sr, subtype="PCM_16")
        etiquetas[clip_id] = {"clip_id": clip_id, "clase": "silencio", "sesion": grupo}
        escritos["silencio"] += 1
    print(f"silencio: {escritos['silencio']} bloques ({min(len(silencios), max_por_clase) - n_sinteticos} reales, "
          f"{n_sinteticos} sintéticos)")

    escribir_etiquetas(etiquetas)
    print(f"\nlabels.csv: {len(etiquetas)} fragmentos -> {dict(escritos)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Importa FSD50K para el clasificador de audio")
    ap.add_argument("--contar", action="store_true")
    ap.add_argument("--importar", action="store_true")
    ap.add_argument("--max-por-clase", type=int, default=1500)
    ap.add_argument("--bloques-por-clip", type=int, default=3,
                    help="como mucho N bloques de 1 s por clip, para no depender de pocos clips largos")
    args = ap.parse_args()
    if args.contar:
        contar()
    if args.importar:
        importar(args.max_por_clase, args.bloques_por_clip)
    if not (args.contar or args.importar):
        ap.print_help()
