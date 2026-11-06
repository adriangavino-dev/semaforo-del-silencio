# Semáforo del Silencio — subsistema de inteligencia artificial

UTEC · Proyectos Interdisciplinarios II (PI6002) · Ciclo 2026-2

Implementación del subsistema de IA que corre en la laptop: captura audio y video
del aula, clasifica el tipo de sonido, detecta movimientos anómalos y genera el
reporte de sesión. El resultado se envía por Wi-Fi al ESP32, que controla los LED.

## Estado de las etapas

| Etapa | Alcance | Estado |
|---|---|---|
| 1 | Webcam + YOLOv8-Pose en vivo, YAMNet reconociendo tipos de sonido | **Listo** |
| 2 | Grabación, corte y etiquetado del conjunto de datos | **Herramientas listas** — faltan las grabaciones |
| 3 | Entrenamiento del clasificador de audio (4 clases propias) | **Tubería lista** — falta entrenar con datos reales |
| 4 | Calibración de línea base y postura | Calibración lista; el clasificador entrenado queda pendiente |
| 5 | Eventos, fusión, reportes y envío al ESP32 | **Listo** — el clasificador de eventos tiene tubería lista; falta grabar actividad normal |

Lo único que falta para tener el sistema completo es **grabar el conjunto de
datos y entrenar**. Todo el código que lo consume ya está escrito y probado.

## Requisitos

- Windows con webcam y micrófono
- **Python 3.11** (TensorFlow no publica distribución para 3.12 en adelante)

## Instalación

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La primera ejecución descarga los pesos de YOLOv8-Pose (~7 MB) y de YAMNet
(~17 MB). Ambos quedan en caché local.

### Nota sobre PowerShell

Los comandos de este README invocan `.venv\Scripts\python.exe` directamente, así
que no hace falta activar el entorno. Eso evita dos tropiezos habituales en
Windows PowerShell:

- **`&&` no funciona.** Windows PowerShell 5.1 no admite ese encadenamiento —se
  agregó en PowerShell 7— y responde *"El token '&&' no es un separador de
  instrucciones válido"*. Hay que usar `;`:

  ```bash
  cd "C:\Users\Adrian\Desktop\Semáforo del Silencio"; .venv\Scripts\python.exe scripts\demo_live.py
  ```

  La diferencia importa: `&&` solo ejecuta lo segundo si lo primero tuvo éxito,
  mientras que `;` lo ejecuta siempre. Si el `cd` falla, el script se lanzará
  igual desde la carpeta equivocada.

- **`activate` puede estar bloqueado** por la política de ejecución de scripts.
  Invocando el ejecutable del entorno no hace falta activarlo.

---

# Flujo de trabajo completo

## 1. Demostración sin entrenar nada

```bash
.venv\Scripts\python.exe scripts\demo_live.py
```

Abre la webcam y muestra los esqueletos que detecta YOLOv8-Pose junto con las
clases de sonido que YAMNet reconoce por defecto. No usa ningún modelo propio.
Se cierra con la tecla `q`.

## 2. Grabar el conjunto de datos

Cada grabación pertenece a una **sesión** (el día o momento) y representa una
**situación** (una de las cuatro clases). Hay que grabar las cuatro clases dentro
de cada sesión, y repetir el juego completo en **al menos tres sesiones
distintas**.

```bash
python scripts\record_session.py --sesion s01 --clase silencio     --minutos 2 --sin-video
python scripts\record_session.py --sesion s01 --clase voz_docente  --minutos 2 --sin-video
python scripts\record_session.py --sesion s01 --clase murmullo     --minutos 2 --sin-video
python scripts\record_session.py --sesion s01 --clase ruido_brusco --minutos 2 --sin-video
```

Después se repite con `--sesion s02` y `--sesion s03`, en otro momento u otro
lugar. Dos razones para insistir en esto:

- Si todo se graba el mismo día en el mismo sitio, el modelo aprende a reconocer
  ese sitio en lugar del fenómeno.
- El entrenamiento reserva **sesiones enteras** para la prueba. Si una sesión
  tuviera una sola clase, el conjunto de prueba quedaría con una sola clase y la
  evaluación no mediría nada.

Qué grabar en cada clase:

| Clase | Qué hacer frente al micrófono |
|---|---|
| `silencio` | Nadie habla. Solo el ruido de fondo del ambiente. |
| `voz_docente` | **Una sola persona** hablando de corrido, con pausas naturales. |
| `murmullo` | **Varias personas** hablando a la vez, sin que ninguna domine. |
| `ruido_brusco` | Golpes en la mesa, sillas arrastradas, aplausos, gritos cortos. |

Con dos minutos por clase y tres sesiones se llega a unas 700 muestras por clase,
bastante por encima del mínimo de 150 que pide la especificación.

## 3. Cortar y revisar las etiquetas

```bash
python scripts\make_clips.py --todas
python scripts\label_tool.py --resumen
```

`make_clips.py` corta todo en fragmentos de 1 segundo con solapamiento de 0.5 s.
Como el nombre del archivo lleva la clase, los fragmentos salen **ya etiquetados**
y no hace falta etiquetar a mano.

`label_tool.py` sirve para revisar y para las sesiones mezcladas:

```bash
python scripts\label_tool.py --sesion s01 --asignar murmullo   # por lote
python scripts\label_tool.py --sesion s02                      # uno por uno
python scripts\label_tool.py --sesion s02 --sugerir            # con sugerencia de YAMNet
```

## 4. Entrenar

```bash
python -m src.audio.train
```

Calcula el vector de 1024 dimensiones de YAMNet para cada fragmento (los guarda
en caché para no repetir el cálculo), separa por sesión, entrena un RandomForest
y deja en `reports/`:

- la matriz de confusión de validación y de prueba, en PNG;
- un JSON con la exactitud balanceada, el reporte por clase y si se cumplen los
  criterios de aceptación de la sección 10.

El modelo queda en `models/audio_clf.pkl`.

### Entrenar sin grabaciones propias: FSD50K

No existe un dataset público de audio de aulas con estas cuatro clases (los
trabajos recientes que simulan aulas, SimClass y RealClass, lo dicen así). Lo
más cercano es **FSD50K** (Freesound, CC-BY, misma ontología de AudioSet que
YAMNet), que se descarga con la librería `soundata`:

```bash
.venv\Scripts\python.exe -m pip install soundata
.venv\Scripts\python.exe -m src.audio.fsd50k --contar
.venv\Scripts\python.exe -m src.audio.fsd50k --importar
.venv\Scripts\python.exe -m src.audio.train
```

`src/audio/fsd50k.py` corta bloques de 1 s y los asigna a las clases del
proyecto (una voz sola → `voz_docente`; Chatter/Crowd/Conversation y mezclas de
5-8 voces → `murmullo`; Slam/Knock/Thump/Door → `ruido_brusco`; tramos casi
mudos y ruido de sala → `silencio`). La "sesión" es el usuario de Freesound
agrupado en 10 grupos, para que el mismo micrófono nunca esté en entrenamiento
y en prueba a la vez. Son sonidos de Freesound, no de un aula: conviene sumar
grabaciones reales con `record_session.py` cuando se pueda.

Resultado (FSD50K eval, 4 899 bloques de 1 s, 10 grupos de usuarios):

| Medida | Valor | Criterio (sección 10) |
|---|---|---|
| Exactitud balanceada, grupo de prueba (`fsd_g9`) | **0,772** | ≥ 0,75 → cumple |
| Exactitud balanceada, validación cruzada (10 grupos) | **0,825** | — |
| Voz confundida con murmullo, validación cruzada | **15,5 %** | ≤ 20 % → cumple |
| Voz confundida con murmullo, grupo de prueba | 38,4 % | ≤ 20 % → no cumple |

Silencio (≈ 0,96 F1) y murmullo (≈ 0,86) salían bien; la voz sola era la
clase débil (recall ≈ 0,70), sobre todo porque casi toda era en inglés.

### Voz en español: OpenSLR 73 (Perú)

[OpenSLR 73](https://www.openslr.org/73/) (Guevara-Rukoz et al., LREC 2020,
CC BY-SA 4.0): 38 hablantes peruanos leyendo en voz alta. Se añaden 1 500
bloques de **voz_docente** (una persona) y 1 000 de **murmullo** (mezclas de 3
a 8 hablantes a la vez), con nivel al azar y ruido de sala para parecerse a un
micrófono en un aula. El hablante decide el grupo: nunca está en
entrenamiento y en prueba a la vez.

```bash
.venv\Scripts\python.exe -m src.audio.openslr --importar
.venv\Scripts\python.exe -m src.audio.train --recalcular
```

| Medida (7 399 bloques) | Solo FSD50K | + español | Criterio |
|---|---|---|---|
| Exactitud balanceada, prueba | 0,772 | **0,860** | ≥ 0,75 ✓ |
| Exactitud balanceada, validación cruzada | 0,825 | **0,868** | — |
| Voz → murmullo, prueba | 38 % ✗ | **15 %** | ≤ 20 % ✓ |
| Voz → murmullo, validación cruzada | 15,5 % | **7,2 %** | ≤ 20 % ✓ |

Dicho con honestidad, por origen: la voz y el murmullo en español se aciertan
en un 99 % y 92 %, porque son grabaciones limpias; los de Freesound siguen en
71 % y 73 %. Lo que se gana de verdad es distinguir **una voz en español de
varias voces en español a la vez**, que es lo que ocurre en el aula. Falta
probarlo con el sonido del aula real.

Para probarlo en vivo:

```bash
python -m src.audio.infer --micro
```

## 5. Calibrar la línea base (para contar personas de pie)

```bash
python -m src.vision.calibrate
```

Graba dos minutos **con todas las personas sentadas** y ajusta la altura de
hombro esperada según la posición en la imagen.

**Sin calibrar, el sistema igual cuenta personas de pie** con una regla
geométrica sobre cada persona: si se le ven las piernas, el ángulo de la rodilla
(sentado ≈ 90°, de pie ≈ 150°); si no —lo normal en un aula, donde las carpetas
las tapan: en `normal_aula_01.mp4` solo se ven en 3 de 117 personas—, la
proporción alto/ancho del recuadro, que en los videos del proyecto da
sentado ≈ 1.0 y de pie ≈ 2.0. Umbrales en `vision.postura`. El reporte indica
qué método se usó (`metodo_postura`). La calibración es más precisa cuando la
cámara está fija; la geometría funciona desde el primer minuto.

## 5b. Entrenar el clasificador de eventos de movimiento

Los eventos se deciden por **ventanas de 2 segundos**, no cuadro a cuadro
(sección 6.1). Cada ventana produce media, máximo y desviación estándar de cada
rasgo por persona, y un evento se registra solo tras tres ventanas anómalas
seguidas (~6 s sostenidos). Mientras no exista un modelo entrenado, decide una
regla provisional sobre la velocidad de las muñecas, calibrada con
`pelea.mp4` y `persona_corriendo.mp4`.

Para entrenar hacen falta videos de **las dos clases**:

```bash
python scripts\record_video.py --nombre aula_normal_01 --clase normal --segundos 120
python scripts\record_video.py --registrar pelea.mp4 --clase movimiento_anomalo
python -m src.vision.train_events
```

**Lo más importante son los videos de clase `normal`**: gente sentada,
escribiendo, levantando la mano, caminando, entrando y saliendo. Eso es lo que
el modelo tiene que aprender a NO marcar.

**Y las dos clases deben grabarse en la misma escena.** Esto es lo que más
importa y lo que más fácil se hace mal. Si el video normal es un aula lejana con
seis personas y el anómalo es una persona en primer plano, el modelo aprende a
distinguir *la escena* (cuántas personas, a qué distancia) y no *el movimiento*:
sale con 100 % de acierto y después marca como pelea a cualquiera que se acerque
a la cámara. Pasó de verdad en este proyecto con `normal_aula_01.mp4` contra
`pelea.mp4`; por eso el modelo entrenado se descartó y el entrenador ahora avisa
cuando los rasgos de escena dominan sobre los de velocidad. La receta correcta:
**misma cámara, mismo lugar, misma cantidad de gente**, grabando primero varios
minutos de actividad normal y después escenificando los movimientos bruscos.

El entrenador excluye automáticamente los videos sin personas en cuadro y evalúa
dejando un video fuera cada vez.

### Estado actual: regla provisional, medida

Mientras no exista un modelo válido, decide la regla por ventana. Se calibró
contra los tres videos disponibles y da:

| Video | Clase real | Eventos detectados |
|---|---|---|
| `normal_aula_01.mp4` (18 s, 3-6 personas) | normal | **0** |
| `pelea.mp4` (27 s) | movimiento anómalo | **2** |
| `persona_corriendo.mp4` (15 s) | movimiento anómalo | **1** |

Con `velocidad_media_por_s: 0.4` aparecía un falso positivo en el video normal;
con `0.5` desaparece sin perder los verdaderos. El margen es estrecho y el
video normal dura 18 s; el criterio de aceptación exige cero falsos positivos en
**cinco minutos**, así que hace falta más grabación normal para confirmarlo.

### Tiempo de reacción

El detector de movimiento confirma un evento con `confirmaciones_evento: 3`
ventanas de 2 s seguidas (≈ 4 s). Un evento confirmado cambia el semáforo en
ese mismo momento, sin la espera de `confirmaciones_estado` (ahora 2 s), que
solo se aplica a los cambios por sonido. Con tus videos: pelea detectada al
segundo 6, carrera al 5, y ninguna falsa alarma en el video normal. Con
`confirmaciones_evento: 2` el video normal sí daba falsa alarma (dos ventanas
seguidas de 0,79 y 0,71 al empezar la grabación), por eso se dejó en 3.

### El color sube por clase, pero solo baja por nivel

Problema medido en el aula (sesión 2026-11-02): con ruido de 80 dB, el modelo
llamaba "voz_docente" a algunos segundos sueltos, y la regla de la sección 7
("voz del docente → VERDE aunque sea alta") devolvía el semáforo a verde en
pleno ruido, unos 2 s después de ponerse rojo.

Ahora:

- Para **subir** se usa la regla de la sección 7 (murmullo + nivel).
- Para **bajar** hace falta que el **nivel** baje de verdad: desde ROJO, por
  debajo de 63 dB; desde AMARILLO, por debajo de 53 dB (umbral −
  `margen_bajada`). Lo que diga el modelo en un segundo suelto no basta.
- La voz del docente por encima de `audio.db.voz_docente_max_db` (70 dB) ya no
  excusa: se trata como ruido (un grito).

### Personas dormidas

`src/vision/sleep.py`, sin entrenamiento: usa el esqueleto de YOLOv8-Pose.

1. **Cabeza baja**: `(y_hombros - y_cabeza) / torso` ≤ `umbral_cabeza` (0,05).
   Erguido vale 0,3-0,6; con la cabeza apoyada en el pupitre, 0 o menos. Si se
   ven los hombros pero ningún punto de la cara, también cuenta.
2. **Quieto**: muñecas a menos de 0,25 alturas por segundo (escribir con la
   cabeza agachada mueve las manos; dormir no).
3. **Sostenido**: la mayoría (≥ 70 %) de los últimos ~3 s cumplen 1 y 2, y eso
   durante `segundos_minimos` (20 s).

En pantalla: `DORMIDO` sobre la persona, `zz 12s` mientras va contando, y
`dormidos N` en la línea CAMARA. Se registra el evento `persona_dormida` y el
tablero muestra "dormidos máximo". No cambia el color del semáforo.

Medido en `normal_aula_01.mp4`: los alumnos despiertos tienen la cabeza bajo el
umbral en el 3-5 % de los cuadros y nunca más de 1,2 s seguidos, así que no se
marca a nadie. Se probó con personas simuladas (dormido detectado a los ~24 s,
despierto con la cabeza baja a ratos nunca detectado). **Falta probarlo con una
grabación real de alguien dormido**: no hay dataset público de aulas con esa
clase que se pueda descargar sin cuenta.

**Comprobado con fotos reales de aulas** (SCB-Dataset, subconjunto BowTurnHead,
Hugging Face `wintonYF/SCB-Dataset`, uso académico; 2 410 fotos):

```bash
.venv\Scripts\python.exe scripts\evaluar_cabeza_scb.py
```

| Alumnos etiquetados como | Medidos | La regla dice "cabeza baja" |
|---|---|---|
| TurnHead (erguidos, cabeza girada) | 6 262 | **1,0 %** |
| BowHead (cabeza agachada: escribe, lee o duerme) | 1 753 | 15,9 % |

La regla casi nunca marca a un alumno erguido (1 %): es lo que importa para no
dar falsos "dormidos". Solo marca el 16 % de las cabezas agachadas porque exige
la cabeza **a la altura de los hombros o más abajo**; quien escribe o lee
agacha la cabeza sin bajarla tanto, y además mueve las manos, que la regla
también descarta. SCB no tiene una clase "dormido", así que no se puede medir
cuántos dormidos reales detecta. Otro dato: YOLOv8n-pose no encontró el
esqueleto de muchos alumnos de estas fotos (3 207 de 4 960 BowHead), porque
salen pequeños o tapados. Resultado en `reports/evaluacion_cabeza_scb.json`.

### Probado con un dataset público de peleas: no sirve, se queda la regla

No hay datasets públicos de peleas grabadas en aulas. Se probó el **Surveillance
Camera Fight Dataset** (Akti et al., IPTA 2019, licencia MIT): 300 clips de
cámaras de vigilancia, 150 de pelea y 150 normales, de 46 videos de origen.

```bash
.venv\Scripts\python.exe scripts\importar_videos_peleas.py
.venv\Scripts\python.exe -m src.vision.train_events
```

Con validación cruzada agrupada por video de origen (10 particiones):
**exactitud balanceada 0,53**, es decir, al nivel del azar, y 0 % en
`pelea.mp4` y `persona_corriendo.mp4`. Los clips duran ~2 s a 360 px, las
personas salen pequeñas y YOLOv8-Pose apenas las encuentra (61 de los 300 clips no
aportaron nada: 42 sin personas detectadas y 19 demasiado cortos). El modelo quedó apartado como
`models/vision_evento_clf_vigilancia_NO_USAR.pkl` y el sistema sigue con la
regla. Para entrenar un modelo útil hacen falta grabaciones del aula real con la
cámara del proyecto (`record_video.py`), normales y anómalas en el mismo sitio.

## 6. Correr una sesión y generar el reporte

```bash
python scripts\run_session.py                 # hasta pulsar q en la ventana o Ctrl+C
python scripts\run_session.py --minutos 45    # duración fija
```

La ventana muestra todo lo que explica la decisión:

| Línea | Qué muestra |
|---|---|
| Estado | color actual, motivo del último cambio y, si hay un cambio en camino, `-> AMARILLO? 2/3 s` |
| AUDIO | clase del modelo, su confianza y la probabilidad de cada clase (`sil voz mur gol`) |
| NIVEL | dB del ESP32 frente a los umbrales de amarillo y rojo |
| CAMARA | personas, de pie, FPS y posición del servo |
| MOVIM. | velocidad media de muñecas y % de cuadros rápidos de la ventana frente a sus umbrales, eventos registrados |
| ESP32 | conexión, LED encendido de verdad, bloques de audio recibidos y perdidos, tiempo |

Sobre cada persona: `#ID confianza` arriba y `de pie / sentado  v<velocidad de muñecas>` abajo.

Mira y escucha el aula, decide el color del semáforo, registra los eventos y
guarda todo en la carpeta de la sesión sin intervención manual. El
color se manda al ESP32 (LEDs) por el cable USB en cuanto cambia y se mantiene
hasta el siguiente cambio; al cerrar la sesión el semáforo vuelve a VERDE.

### Una carpeta por sesión, con grabación y fotos

La grabación quedó autorizada (los padres están informados). Cada sesión crea
su carpeta con la fecha y el número de sesión del día:

```
sesiones/
  2026-11-02_sesion_01/
    reporte.json               resumen; es lo que lee el tablero
    lecturas_por_segundo.csv   estado, clase de audio, dB, personas, de pie, dormidos
    eventos.csv                movimientos anómalos, personas dormidas, ruidos bruscos
    cambios_estado.csv         cada cambio de color, con su causa y su foto
    video.mp4                  la cámara completa, con la hora impresa
    audio.wav                  el micrófono del ESP32 (16 kHz)
    fotos/                     HHMMSS_evento_NNN.jpg y HHMMSS_cambio_NNN.jpg
  2026-11-02_sesion_02/
  ...
```

- **Fotos** (`src/vision/evidencias.py`): una al confirmarse un movimiento
  anómalo o una persona dormida, y otra en cada cambio de color. Llevan los
  recuadros de las personas y una franja con el estado, la hora y el motivo. El
  tablero las muestra en miniatura en las tablas de eventos y de cambios de
  estado; un clic la abre en grande.
- **Video y audio** (`src/fusion/grabacion.py`): solo con cámara en vivo. El
  video se graba a FPS fijos (repite cuadros si el análisis va más lento), así
  que dura lo mismo que la sesión. Tamaño aproximado: video 3–5 MB/min, audio
  1,9 MB/min; una clase de 45 min ocupa ~250–300 MB.
- Todo se apaga o ajusta en `grabacion` (`config.yaml`): `video`, `audio`,
  `fotos.al_evento`, `fotos.al_cambio_estado`, `fotos.max_por_sesion` y
  `fotos.difuminar_caras` (pixela las caras, por si algún salón no lo autoriza).
- `sesiones/` está en `.gitignore`: las grabaciones de alumnos nunca van al
  repositorio.
- Las sesiones antiguas que quedaron en `reports/sesion_*.json` el tablero las
  sigue mostrando.

### De dónde sale el sonido: `audio.fuente`

- `esp32` (**lo configurado ahora**): todo el sonido sale del INMP441 del ESP32;
  el micrófono de la laptop no se abre. Por el mismo cable USB, a 921 600
  baudios, el ESP32 manda:
  - una lectura por segundo con los **dB calibrados** (`DB_OFFSET` en
    `firmware/semaforo_esp32/config.h`; un aula tranquila marca ~35–40), y
  - con `esp32.audio_por_usb: true`, el **audio completo** a 16 kHz en bloques de
    32 ms (`@<n>:<base64>`). Con él la laptop corre YAMNet + el clasificador de
    4 clases (`src/audio/captura_esp32.py`).

  La clase de audio sale del modelo y los dB de la regla, del ESP32 (55 dB →
  AMARILLO, 65 → ROJO). Sin modelo entrenado, decide solo por nivel, igual que
  el firmware. Medido: 37 s de sesión, 1 174 bloques recibidos, 0 perdidos.
- `laptop`: el micrófono de la laptop. Con modelo entrenado, la regla completa
  de la sección 7 (voz del docente / murmullo / ruido brusco); sin modelo, lo
  que sigue.

### Con `fuente: laptop` y sin modelo de audio

Si `models/audio_clf.pkl` todavía no existe, `run_session.py` avisa y decide el
color **solo por nivel de ruido**, igual que hace el ESP32 con su micrófono: la
mediana de los últimos 60 s es el fondo del aula y el semáforo pasa a AMARILLO
al subir 8 dB sobre él y a ROJO al subir 15 dB (promedio de 3 s, con histéresis
para no parpadear; `audio.db.sin_modelo` en `config.yaml`). Los dB del micrófono
de la laptop son relativos, por eso se compara contra el fondo y no contra los
umbrales fijos de 55/65. En el tablero estos cambios aparecen con el motivo
"ruido (nivel sobre el fondo del aula)". Es un respaldo declarado: la regla de
la sección 7 con las cuatro clases entra sola en cuanto se entrena el modelo.

En todos los casos, un cambio de estado necesita `fusion.confirmaciones_estado`
segundos seguidos (no cuadros) proponiendo el mismo color.

### Analizar una grabación en vez de la cámara

Sirve para mostrar el sistema funcionando sin depender de lo que pase en vivo:

```bash
python scripts\run_session.py --video data\video\raw\persona_corriendo.mp4
```

Los videos van en `data/video/raw/`. El reporte sale igual que el de una sesión
en vivo, pero con dos diferencias:

- La duración y las marcas de tiempo son las **del video**, no las del reloj: el
  archivo se analiza más rápido que en tiempo real.
- **No hay audio.** OpenCV lee solo la imagen. Extraer la pista de sonido de un
  MP4 necesitaría ffmpeg, que no está instalado. El reporte queda marcado con
  `"audio_disponible": false` y el tablero lo indica.

Quitar `--sin-ventana` muestra la ventana con los esqueletos y el panel de
estado mientras analiza, que es lo más claro para una demostración.

## 7. Ver el tablero web

```bash
python scripts\serve_dashboard.py
```

Abre `http://localhost:8000` con los reportes por salón: el estado en grande
(VERDE, ÁMBAR o ROJO), el reporte de audio, el reporte de cámara, la línea de
tiempo de la sesión y el historial. Se actualiza solo cada diez segundos, así que
se puede dejar abierto mientras corre una sesión.

### El ESP32: por cable USB

**Servo**: 3 posiciones fijas, 30°, 90° y 150° (izquierda, centro, derecha;
`ANGULOS_SERVO` en `config.h`). Con `servo.habilitado: true` y `posiciones: 3`
en `config.yaml`, la cámara pasa `permanencia_s` en cada una y el análisis se
pausa `tiempo_giro_s` mientras gira.

El ESP32 (LEDs, servo, micrófono INMP441) se conecta a la laptop **por el cable
USB**. No hay que configurar redes ni IPs: `run_session.py` busca el puerto del
ESP32 solo (`esp32.puerto_serie: auto`), le manda el color que decide el modelo
y las órdenes del servo, y recibe sus lecturas (dB, LED, posición) una vez por
segundo. Esas lecturas van a la base de datos y el tablero las muestra en el
panel "ESP32 del aula".

Para comprobar la conexión antes de una sesión:

```bash
.venv\Scripts\python.exe scripts\probar_esp32.py
```

Prueba, en orden, que llegan lecturas, que obedece un color (enciende el rojo)
y que obedece al servo. Solo un programa puede tener el puerto abierto: si el
monitor serie del IDE de Arduino está abierto, ciérralo antes.

También existe el modo por **Wi-Fi** (`esp32.transporte: wifi` en `config.yaml`
y `MODO_WIFI true` en el firmware): el ESP32 se anuncia por UDP, la laptop lo
descubre y `serve_dashboard.py` recibe sus lecturas por HTTP. Sirve si el ESP32
tiene que estar lejos de la laptop, pero depende de que los dos estén en la
misma red (en la universidad, el hotspot de un celular).

El contrato completo en los dos sentidos está en
[`docs/contrato_esp32.md`](docs/contrato_esp32.md). El firmware listo para grabar y
los pasos para conectarlo están en [`firmware/README.md`](firmware/README.md).

### Probar la comunicación sin el circuito

```bash
python scripts\simular_esp32.py
```

Se comporta como el ESP32 real por Wi-Fi: recibe el estado y las órdenes del
servo, decide su propio color por dB con histéresis, y manda lecturas cada
segundo. Para usarlo: `esp32.transporte: wifi`, `ip: 127.0.0.1`, `puerto: 8081`
en `config.yaml`. Con el hardware por cable, volver a `transporte: usb`.

### Base de datos

Todo queda además en `data/semaforo.db` (SQLite, sin instalar nada): sesiones
con su traza segundo a segundo, eventos, lecturas del ESP32 y órdenes al servo.

```bash
python -m src.db
python -m src.db --sql "SELECT * FROM esp32_lecturas ORDER BY id DESC LIMIT 10"
```

Para una versión en un solo archivo que se pueda mandar por correo:

```bash
python scripts\exportar_tablero.py
```

Y para probar el tablero sin grabar una sesión de verdad:

```bash
python scripts\simular_sesion.py --minutos 45
```

Los reportes simulados quedan marcados con `"simulada": true` y el tablero los
muestra con una etiqueta, para que nunca se confundan con mediciones reales.

---

## Cómo colocar la cámara

Lo que el sistema mide depende de que la cámara no se mueva y de que la escena
sea la misma al entrenar y al usar:

- **Fija.** En un trípode o apoyada, sin que nadie la toque durante la sesión.
  La calibración de línea base y el clasificador de postura asumen que cada
  posición de la imagen corresponde siempre al mismo sitio del aula; si la
  cámara se mueve, todo lo aprendido deja de valer.
- **Horizontal (apaisada), no vertical.** Las zonas de `config.yaml` (docente
  arriba, alumnos abajo) y la línea base están pensadas para un encuadre
  apaisado que abarque todo el aula. `pelea.mp4` se grabó en vertical
  (478×850): el detector funciona igual, pero una persona ocupa casi todo el
  alto del cuadro y la escena no se parece a la de despliegue. Para entrenar y
  demostrar, grabar siempre en horizontal desde el punto donde irá la cámara.
- **Si la cámara gira con el servo, gira entre posiciones fijas.** Nunca un
  barrido continuo: mientras la cámara se mueve, todo lo que ve se desplaza y
  para el sistema es igual que si todas las personas se movieran a la vez. Con
  `servo.habilitado: true`, la laptop ordena el giro (nunca a mitad de un
  evento), **pausa el análisis** `servo.tiempo_giro_s` mientras la cámara se
  asienta, y reinicia el seguimiento. Cada posición tiene su propia calibración
  (`models/baseline_calib.json` para la 0, `baseline_calib_p1.json` para la 1)
  y puede tener sus propias zonas (`servo.zonas_por_posicion`). Lo grabado para
  entrenar se graba con la cámara quieta en una posición.
- **Desde el punto de despliegue.** Los videos de entrenamiento se graban desde
  el mismo lugar y altura donde quedará la cámara en la clase real. Un video
  grabado desde otro ángulo enseña otra escena.

## Probar cada módulo por separado

Cada módulo se ejecuta de forma independiente (restricción 6):

```bash
python -m src.config                  # imprime la configuración cargada
python -m src.audio.capture --listar  # lista los micrófonos disponibles
python -m src.audio.dataset           # estado del conjunto de datos
python -m src.audio.features --wav ARCHIVO
python -m src.audio.infer --micro
python -m src.vision.pose --camara 1
python -m src.vision.features --camara 1
python -m src.fusion.state            # prueba la regla de decisión del semáforo
python -m src.fusion.reporter         # lista los reportes guardados
python -m src.comm.esp32_client --probar
```

## Configuración

Todo está en [`config/config.yaml`](config/config.yaml): umbrales de dB, clases,
zonas, IP del ESP32, puerto del tablero y nombre del salón. No hay valores fijos
en el código (restricción 5).

Para conectar el ESP32 hay que poner `esp32.habilitado: true` y su IP. Mientras
esté en `false`, el sistema funciona igual pero no intenta enviar nada.

## Estructura

```
config/config.yaml        umbrales, rutas, clases, IP del ESP32
data/                     grabaciones, fragmentos y etiquetas
models/                   pesos de YOLO, clasificador entrenado, calibración
reports/                  reportes de sesión (CSV + JSON) y matrices de confusión
web/index.html            tablero de reportes por salón
src/config.py             cargador único de configuración
src/db.py                 base de datos SQLite (sesiones, ESP32, servo)
src/audio/capture.py      captura de micrófono a 16 kHz mono
src/audio/features.py     YAMNet: clases de AudioSet y vector de 1024
src/audio/dataset.py      lectura y escritura de labels.csv
src/audio/train.py        entrenamiento del clasificador de audio
src/audio/infer.py        inferencia con ventana deslizante
src/vision/pose.py        YOLOv8-Pose con seguimiento de identidad
src/vision/features.py    características por persona (sección 6.3)
src/vision/calibrate.py   calibración de la línea base de altura
src/vision/events.py      decisión de eventos por ventana de 2 s
src/vision/train_events.py  entrenamiento del clasificador de eventos
src/fusion/state.py       regla de decisión del semáforo con histéresis
src/fusion/reporter.py    acumulación y escritura del reporte de sesión
src/comm/esp32_client.py  cliente HTTP hacia el ESP32
scripts/record_video.py   grabación y registro de videos para eventos
scripts/simular_esp32.py  ESP32 simulado para probar la comunicación
scripts/probar_esp32.py   prueba paso a paso la conexión con el ESP32 real
firmware/semaforo_esp32/  programa del ESP32 (Arduino)
docs/contrato_esp32.md    contrato laptop <-> ESP32 y esbozo del firmware
scripts/                  grabación, corte, etiquetado, sesión y tablero
```

## Desviaciones respecto de la especificación

1. **Instalación de Python 3.11 aparte.** La laptop tenía únicamente Python 3.14,
   para el cual TensorFlow no publica distribución. Se instaló Python 3.11 y el
   proyecto usa un entorno virtual propio en `.venv`. Se conserva así YAMNet y no
   fue necesario recurrir al reemplazo por MFCC que la sección 3 admite.
2. `numpy` queda fijado en 1.26.4 porque TensorFlow 2.15 no es compatible con
   numpy 2.x, y `setuptools` por debajo de 81 porque `tensorflow-hub` importa
   `pkg_resources`, que las versiones nuevas eliminaron.
3. **Se añadió `lapx`** para el seguimiento de identidad de YOLO. Sin él, el id de
   cada persona es su posición en la lista de detecciones, que cambia entre
   cuadros, y la velocidad de las muñecas sale disparada al comparar personas
   distintas. Con el seguimiento, los falsos eventos de movimiento bajaron de 44
   en 45 segundos a 1 en 90 segundos.
4. **La postura se decide por umbral sobre la línea base**, no con el
   RandomForest entrenado que pide la Etapa 4. Es una regla provisional que
   permite contar personas de pie antes de tener ese modelo, y está marcada como
   tal en el código.
5. **Un movimiento anómalo confirmado pone el semáforo en AMARILLO**
   (`fusion.evento_visual_cambia_estado: true`). La sección 7 dice que el color
   depende solo del audio y que el movimiento anómalo únicamente se registra
   como evento; pero mientras el clasificador de audio no esté entrenado el color
   no cambiaría nunca. Se puede volver a la regla original en `config.yaml`.
6. **Filtro de detecciones fantasma** (`vision.filtro_personas`): se descartan
   recuadros menores al 15 % del alto del cuadro y detecciones con confianza
   media de keypoints menor a 0.30. Sin esto la cámara contaba fotos de la pared
   como personas.
7. **El ESP32 también envía datos a la laptop.** La sección 8 solo define
   laptop → ESP32. Se añadió el sentido contrario (`POST /api/esp32/lectura`)
   para que el reporte y el tablero tengan los decibeles reales del INMP441 y
   el estado real del LED, y quede todo en la base de datos. El ESP32 sigue
   conservando su control propio si la laptop no responde.
8. **Servo entre posiciones fijas con pausa de análisis.** No está en la
   especificación. Ver "Cómo colocar la cámara".
9. **La velocidad de las muñecas se mide por segundo, no por cuadro.** La
   sección 6.3 la define como el desplazamiento entre cuadros consecutivos, pero
   así el valor depende de los FPS de la fuente: el mismo movimiento a 60 FPS da
   la mitad que a 30. Se multiplica por los FPS para que el umbral sirva igual
   con la webcam (~26 FPS) que con un video grabado (60 FPS). Además se toma la
   menor de las dos muñecas —si el detector hace saltar una sola, la otra revela
   que no hubo movimiento— y se descarta cualquier velocidad físicamente
   imposible (más de 12 alturas de recuadro por segundo: eso es un salto del
   detector, no un cuerpo). Estos dos cambios eliminaron los picos de 46
   alturas/segundo que disparaban alertas falsas.

## Verificación de cada target con datasets públicos

```bash
.venv\Scripts\python.exe scripts\verificar_targets.py todo
```

Mide cada target con un dataset público y guarda el resultado en
`reports/verificacion_targets.json`. La tabla en imagen está en
`presentacion_entrenamiento/verificacion_targets.png`.

| Target | Dataset | Qué se mide |
|---|---|---|
| Sonido (4 clases) | FSD50K + OpenSLR 73 | validación cruzada por grupo |
| Personas detectadas | [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) Part A, cámaras de aulas | % de cabezas marcadas que el modelo encuentra |
| Sentado | SCB BowTurnHead, alumnos en el pupitre | % clasificados sentados |
| De pie, correr, trotar, boxeo, caminar | [KTH Actions](https://www.csc.kth.se/cvap/actions/) | evento confirmado (lo que cambia el semáforo) |
| Pelea / normal | Surveillance Camera Fight | ventana anómala (los clips duran 2 s) |
| Ojos cerrados | [CEW](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html) | exactitud según la distancia entre ojos en la imagen |
| Dormido | SCB | erguidos que NO se marcan |

### Ojos cerrados

`src/vision/ojos.py`: con los puntos de los ojos de YOLOv8-Pose se recorta cada
ojo (lado = 0,7 × distancia entre ojos), se pasa a 24×24 y un RandomForest
(HOG + píxeles) decide abierto o cerrado. Entrenado con CEW (4 846 recortes de
2 423 caras): **92,5 %** en validación cruzada por cara.

```bash
.venv\Scripts\python.exe -m src.vision.ojos --entrenar
```

Funciona mientras la cara se vea grande: 92 % con 40 px entre los ojos, 88 % con
15 px, y solo 70 % con 10 px. Por eso solo decide si hay al menos
`vision.ojos.min_px_entre_ojos` (20 px). En vivo se calcula cada 0,5 s, se ve
como `ojos cerrados` sobre la persona y cuenta para "dormido" (ojos cerrados
**o** cabeza baja, quieto, 20 s).

### Regla de movimiento ajustada con KTH

Se guardaron una vez los puntos del cuerpo de 399 videos (KTH, cámaras de
vigilancia y los propios) y se probaron variantes de la regla sin volver a
pasar YOLO. Resultado:

- Subir `fraccion_cuadros_rapidos` de 0,05 a 0,08 reduce las falsas alarmas al
  **caminar de 38 % a 17 %** sin perder carreras (83 %). Los tres videos propios
  siguen correctos. Se pierde algo de boxeo (42 % → 25 %).
- Evaluar a cada persona por separado (en vez del promedio del grupo) detecta
  más, pero dispara con **el docente**, que camina y gesticula tanto como una
  pelea. Se descartó.
- Compensar el movimiento de la cámara no aporta con cámara fija. Se descartó.

### ¿Un modelo de pose más grande?

`scripts/comparar_modelos_pose.py`, en fotos de aulas (SCUT-HEAD) y en el video
propio:

| Modelo | Peso | Personas encontradas (SCUT-HEAD) | Velocidad en CPU |
|---|---|---|---|
| YOLOv8n-pose (el actual) | 6,8 MB | 5,8 % | 24 FPS |
| YOLOv8s-pose | 23,5 MB | 7,4 % | 13 FPS |
| YOLOv8m-pose | 53,3 MB | 9,5 % | 6 FPS |

En auditorios de 60-80 alumnos de los que solo se ve la cabeza, ningún modelo
de pose los encuentra (subir la resolución de entrada a 1280 llega a 13,5 % y
cae a 8 FPS). Contar a todos ahí requeriría un detector de cabezas entrenado
aparte. Para un aula de colegio se queda el nano: es el único que mantiene la
velocidad que necesita el detector de movimiento.

## Pendientes conocidos

- Entrenar el clasificador de audio con grabaciones reales del aula.
- Entrenar el clasificador de postura y el de eventos (Etapa 4 y parte de la 5).
- **Grabar las dos clases de evento en la misma escena y entrenar.** La regla
  provisional da 0 falsos positivos en los 18 s de aula normal disponibles; el
  criterio exige cero en cinco minutos. Hacen falta varios minutos de actividad
  normal y algunos movimientos bruscos escenificados, todo desde la misma cámara
  fija y horizontal, y luego `python -m src.vision.train_events`.
- Revisar qué está detectando la cámara como personas. En las pruebas de
  escritorio contó entre 3 y 4 personas con una sola presente, probablemente por
  fotos o afiches en la pared. Conviene verificarlo en el aula.

## Limitación conocida y declarada

El sistema distingue *una voz dominante* de *varias voces superpuestas*; no
identifica quién es el docente. Si un alumno habla solo y en voz alta, será
clasificado como `voz_docente`. Reconocer a una persona por su voz exigiría
registrar su huella vocal, lo que el proyecto decidió evitar por privacidad.
Esta limitación se documenta, no se oculta.

## No guardar el proyecto dentro de OneDrive

El entorno virtual `.venv` contiene decenas de miles de archivos pequeños.
OneDrive los sincroniza y, para ahorrar espacio, convierte los que no se usan en
marcadores de solo-nube. Si después se mueve o copia la carpeta, esos marcadores
no se materializan y las librerías llegan incompletas. El síntoma típico es:

```
ImportError: cannot import name 'overrides' from 'numpy.core' (unknown location)
```

Por eso el proyecto vive en `C:\Users\Adrian\Desktop\`, fuera de OneDrive.

El `.venv` no se copia ni se mueve entre computadoras: cada integrante lo crea en
la suya con los dos comandos de la sección de instalación. Si aparece un error de
ese tipo, la solución es borrar `.venv` y volver a crearlo.
