// Semáforo del Silencio — firmware del ESP32.
//
// Qué hace:
//   1. Mide el nivel sonoro con el INMP441 por I2S y lo convierte a dB.
//   2. Enciende el LED verde, amarillo o rojo según umbrales con histéresis.
//   3. Habla con la laptop por el cable USB (puerto serie, 115200 baudios):
//        ESP32 -> laptop: cada ENVIAR_CADA_MS una línea ">{json}" con la lectura.
//        laptop -> ESP32: una línea JSON por orden, y el ESP32 contesta ">{json}":
//            {"estado":"ROJO"}   color del semáforo (se mantiene hasta la siguiente orden)
//            {"servo":1}         a qué posición girar la cámara
//            {"prueba_led":1}    enciende los tres LED por turno
//            {"lectura":1}       pide la lectura actual
//            {"audio":1} / {"audio":0}   empieza / deja de mandar el audio
//        Con el audio activado, el ESP32 manda además una línea por bloque de
//        32 ms: "@<n>:<base64>" con 512 muestras PCM de 16 bits a 16 kHz (el
//        micrófono del aula para YAMNet y el clasificador de la laptop).
//        Las líneas que no empiezan por ">" son mensajes para leer en el monitor.
//   4. Con MODO_WIFI true (config.h, lo configurado) hace lo mismo por la red:
//      servidor HTTP en el puerto 80 (POST /estado, POST /servo, POST /audio,
//      GET /lectura, GET /prueba_led), envío de lecturas a la laptop y
//      descubrimiento de su IP por UDP. El audio viaja por UDP: cada paquete es
//      [número de bloque, 4 bytes little-endian][512 muestras PCM de 16 bits].
//
// Unidad autónoma: basta el switch. Sin laptop y sin Wi-Fi el semáforo arranca
// igual y decide solo por dB. Si la laptop deja de hablar, su orden caduca a los
// ORDEN_LAPTOP_MS y el ESP32 retoma el control propio.
//
// Bibliotecas (Gestor de bibliotecas del IDE de Arduino):
//   - ArduinoJson (Benoit Blanchon)
//   - ESP32Servo (Kevin Harrington)
// Placa: "ESP32 Dev Module" (núcleo esp32 de Espressif, versión 3.x: usa ESP_I2S.h).
//
// Todo lo configurable está en config.h.

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <ESP_I2S.h>      // driver I2S del núcleo esp32 3.x (el clásico driver/i2s.h entregaba ceros)
#include <driver/gpio.h>
#include <mbedtls/base64.h>
#include <math.h>
#include "config.h"

// ------------------------------------------------------------------ I2S
#define I2S_MUESTRAS_HZ 16000
#define I2S_BLOQUE      512        // muestras por canal y lectura (32 ms a 16 kHz)

I2SClass i2s;
static int32_t bufferI2S[I2S_BLOQUE * 2];   // estéreo intercalado: L, R, L, R...
static bool i2sListo = false;

// Diagnóstico del micrófono, va en cada lectura como "mic":
//   "sin_driver"   -> no se pudo iniciar el I2S
//   "sin_muestras" -> el driver no entrega nada
//   "ceros"        -> llegan muestras pero todas valen 0 (SD sin señal)
//   "izq" / "der"  -> canal donde hay señal (L/R a GND o a 3V3)
static const char* micEstado = "sin_driver";
static float micRmsIzq = 0, micRmsDer = 0;
static int ultimasMuestras = 0;   // muestras (L+R) del último bloque leído

void iniciarI2S() {
  i2s.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1 /*sin salida*/, PIN_I2S_SD);
  // Estéreo: se leen los dos canales y se usa el que traiga señal, así da
  // igual si L/R del INMP441 está a GND o a 3V3. 32 bits por muestra: el
  // INMP441 entrega 24 bits útiles alineados arriba.
  i2sListo = i2s.begin(I2S_MODE_STD, I2S_MUESTRAS_HZ, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
  i2s.setTimeout(80);   // ms máximos esperando un bloque; no bloquear el bucle
  micEstado = i2sListo ? "sin_muestras" : "sin_driver";
  if (!i2sListo) Serial.println("[mic] no se pudo iniciar el I2S");
}

// Nivel RMS del último bloque, en dB relativos + DB_OFFSET.
// Devuelve -1 si no llegaron muestras (micrófono desconectado).
float medirDb() {
  ultimasMuestras = 0;
  if (!i2sListo) { micEstado = "sin_driver"; return -1.0f; }
  size_t leidos = i2s.readBytes((char*)bufferI2S, sizeof(bufferI2S));
  int n = leidos / sizeof(int32_t);
  if (n < 2) { micEstado = "sin_muestras"; return -1.0f; }

  double sumaA = 0.0, sumaB = 0.0;
  for (int i = 0; i + 1 < n; i += 2) {
    // Los 24 bits útiles vienen alineados arriba: se desplazan 8 para quedar en escala.
    double a = (double)(bufferI2S[i] >> 8);
    double b = (double)(bufferI2S[i + 1] >> 8);
    sumaA += a * a;
    sumaB += b * b;
  }
  micRmsIzq = sqrt(sumaA / (n / 2));
  micRmsDer = sqrt(sumaB / (n / 2));
  ultimasMuestras = n;
  double rms = micRmsIzq > micRmsDer ? micRmsIzq : micRmsDer;
  if (rms < 1.0) { micEstado = "ceros"; rms = 1.0; }
  else micEstado = micRmsIzq >= micRmsDer ? "izq" : "der";
  // 20*log10 sobre la escala de 24 bits: 0 dB = 1 LSB, ~138 dB = fondo de escala.
  return 20.0f * log10f((float)rms) + DB_OFFSET;
}

// Audio a la laptop por el cable. Se manda el canal que trae señal (el mismo que
// usa medirDb). El número de bloque permite a la laptop detectar pérdidas.
static bool enviarAudio = false;
static uint32_t bloqueAudio = 0;
static int16_t pcmAudio[I2S_BLOQUE];
static WiFiUDP udpAudio;
static IPAddress audioDestino;
static uint16_t audioPuerto = 0;          // 0 = por el cable; si no, UDP a la laptop
static unsigned long audioHasta = 0;      // si la laptop calla, se deja de transmitir
static uint8_t paqueteAudio[4 + I2S_BLOQUE * sizeof(int16_t)];
static unsigned char base64Audio[((I2S_BLOQUE * 2 + 2) / 3) * 4 + 4];

void enviarBloqueAudio(int nMuestrasEstereo) {
  if (!enviarAudio || nMuestrasEstereo < 2) return;
  int canal = (micRmsDer > micRmsIzq) ? 1 : 0;
  int n = nMuestrasEstereo / 2;
  if (n > I2S_BLOQUE) n = I2S_BLOQUE;
  for (int i = 0; i < n; i++) {
    int32_t v = bufferI2S[2 * i + canal] >> AUDIO_DESPLAZAMIENTO;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    pcmAudio[i] = (int16_t)v;
  }
  if (audioPuerto) {
    if (WiFi.status() != WL_CONNECTED) return;
    memcpy(paqueteAudio, &bloqueAudio, 4);
    memcpy(paqueteAudio + 4, pcmAudio, n * sizeof(int16_t));
    bloqueAudio++;
    udpAudio.beginPacket(audioDestino, audioPuerto);
    udpAudio.write(paqueteAudio, 4 + n * sizeof(int16_t));
    udpAudio.endPacket();
    return;
  }
  size_t largo = 0;
  if (mbedtls_base64_encode(base64Audio, sizeof(base64Audio), &largo,
                            (const unsigned char*)pcmAudio, n * sizeof(int16_t)) != 0) return;
  Serial.print('@');
  Serial.print(bloqueAudio++);
  Serial.print(':');
  Serial.write(base64Audio, largo);
  Serial.print('\n');
}

// ------------------------------------------------------------------ Estado
WebServer servidor(80);
Servo servo;
WiFiMulti wifiMulti;
WiFiUDP udp;
Preferences memoria;

String laptopUrl = LAPTOP_URL_INICIAL;   // se actualiza por descubrimiento y se guarda en memoria
unsigned long ultimoAnuncio = 0;
bool laptopDescubierta = false;

String  led = "VERDE";
String  ordenLaptop = "";
unsigned long ordenHasta = 0;
int     posicion = 0;
unsigned long girandoHasta = 0;
// Giro suave: ángulo en el que está el servo y ángulo al que va.
int anguloActual = 0;
int anguloObjetivo = 0;
unsigned long ultimoPasoServo = 0;
float   nivelDb = 0.0f;
float   nivelSuave = 0.0f;           // media móvil para que el LED no parpadee
unsigned long ultimoEnvio = 0;
unsigned long enviosOk = 0, enviosFallidos = 0;
uint32_t arranques = 0;              // veces que ha arrancado: delata reinicios en el ensayo burn-in

void encender(const String& estado) {
  led = estado;
  digitalWrite(PIN_LED_VERDE,    estado == "VERDE");
  digitalWrite(PIN_LED_AMARILLO, estado == "AMARILLO");
  digitalWrite(PIN_LED_ROJO,     estado == "ROJO");
}

// Auto-prueba: enciende cada LED por turno. Sirve para comprobar el cableado
// sin depender de la laptop ni del micrófono: si un color no se enciende, el
// problema está entre ese pin y ese LED.
void autoPruebaLeds(int vueltas) {
  const char* orden[] = {"VERDE", "AMARILLO", "ROJO"};
  for (int v = 0; v < vueltas; v++) {
    for (int i = 0; i < 3; i++) {
      encender(orden[i]);
      Serial.printf("[prueba] LED %s (pin %d)\n", orden[i],
                    i == 0 ? PIN_LED_VERDE : i == 1 ? PIN_LED_AMARILLO : PIN_LED_ROJO);
      delay(400);
    }
  }
  encender("VERDE");
}

// Control propio por dB con histéresis: sube al superar umbral + H, baja al
// caer por debajo de umbral - H. Entre medio, se mantiene.
void decidirPropio(float db) {
  if (led == "ROJO") {
    if (db < UMBRAL_ROJO - HISTERESIS_DB) encender(db > UMBRAL_AMARILLO ? "AMARILLO" : "VERDE");
  } else if (led == "AMARILLO") {
    if (db > UMBRAL_ROJO + HISTERESIS_DB)          encender("ROJO");
    else if (db < UMBRAL_AMARILLO - HISTERESIS_DB) encender("VERDE");
  } else {
    if (db > UMBRAL_ROJO + HISTERESIS_DB)          encender("ROJO");
    else if (db > UMBRAL_AMARILLO + HISTERESIS_DB) encender("AMARILLO");
  }
}

String lecturaJson() {
  StaticJsonDocument<640> doc;   // trae red, laptop_url y contadores
  doc["nivel_db"] = round(nivelSuave * 10) / 10.0;
  doc["estado_led"] = led;
  doc["posicion_servo"] = posicion;
  doc["girando"] = millis() < girandoHasta;
  doc["orden_laptop"] = ordenLaptop != "" && (ORDEN_LAPTOP_MS == 0 || millis() < ordenHasta);
  doc["uptime_s"] = millis() / 1000;
  doc["envios_ok"] = enviosOk;
  doc["envios_fallidos"] = enviosFallidos;
  doc["arranques"] = arranques;
  doc["modo"] = MODO_WIFI ? "wifi" : "usb";
  doc["mic"] = micEstado;
  doc["mic_rms_izq"] = round(micRmsIzq);
  doc["mic_rms_der"] = round(micRmsDer);
  if (MODO_WIFI) {
    doc["ip"] = WiFi.localIP().toString();
    doc["red"] = WiFi.SSID();
    doc["rssi"] = WiFi.RSSI();
    doc["laptop_url"] = laptopUrl;
    doc["laptop_descubierta"] = laptopDescubierta;
  }
  String salida;
  serializeJson(doc, salida);
  return salida;
}

// ------------------------------------------------------------------ Órdenes de la laptop
// Las mismas dos órdenes llegan por USB o por HTTP; aquí está la lógica común.
bool aplicarEstado(const String& estado) {
  if (estado != "VERDE" && estado != "AMARILLO" && estado != "ROJO") return false;
  ordenLaptop = estado;
  ordenHasta = millis() + ORDEN_LAPTOP_MS;
  audioHasta = ordenHasta;
  encender(estado);
  Serial.printf("[laptop] estado -> %s\n", estado.c_str());
  return true;
}

// Cuánto tarda el giro pendiente: grados que faltan x MS_POR_GRADO + asentamiento.
unsigned long tiempoGiroMs() {
  return (unsigned long)abs(anguloObjetivo - anguloActual) * MS_POR_GRADO + ASENTAMIENTO_MS;
}

// Avanza el servo hacia el objetivo sin bloquear el bucle: cuántos grados le
// tocan según el tiempo transcurrido (el bucle tarda ~32 ms por el micrófono).
void moverServoSuave() {
  if (anguloActual == anguloObjetivo) return;
  unsigned long ahora = millis();
  int pasos = (int)((ahora - ultimoPasoServo) / MS_POR_GRADO);
  if (pasos <= 0) return;
  ultimoPasoServo += (unsigned long)pasos * MS_POR_GRADO;
  int falta = anguloObjetivo - anguloActual;
  if (abs(falta) <= pasos) anguloActual = anguloObjetivo;
  else anguloActual += (falta > 0 ? pasos : -pasos);
  servo.write(anguloActual);
}

bool aplicarServo(int p) {
  if (p < 0 || p >= (int)N_POSICIONES) return false;
  posicion = p;
  anguloObjetivo = ANGULOS_SERVO[p];
  ultimoPasoServo = millis();
  girandoHasta = millis() + tiempoGiroMs();
  Serial.printf("[laptop] servo -> posicion %d (%d grados, giro suave de %d ms)\n",
                p, ANGULOS_SERVO[p], (int)tiempoGiroMs());
  return true;
}

// Autoprueba del micrófono: mira si el pin SD se mueve (un INMP441 alimentado
// y con reloj lo conmuta; en alta impedancia se lee 0 fijo) y enseña las
// primeras muestras crudas. No se tocan los pines de reloj: cambiarles el modo
// para leerlos desconecta la señal I2S del pin.
String pruebaI2S() {
  int altosSd = 0;
  const int N = 4000;
  for (int i = 0; i < N; i++) altosSd += gpio_get_level((gpio_num_t)PIN_I2S_SD);
  for (int k = 0; k < 3; k++) medirDb();
  float db = medirDb();
  String crudas = "";
  for (int i = 0; i < 6; i++) crudas += String(bufferI2S[i]) + (i < 5 ? "," : "");
  return "{\"ok\":true,\"orden\":\"prueba_i2s\",\"muestras\":" + String(N) +
         ",\"sd_altos\":" + String(altosSd) + ",\"estado\":\"" + micEstado +
         "\",\"rms_izq\":" + String(micRmsIzq, 0) + ",\"rms_der\":" + String(micRmsDer, 0) +
         ",\"db\":" + String(db, 1) + ",\"crudas\":[" + crudas + "]}";
}

// ------------------------------------------------------------------ Puerto serie (USB)
// La laptop manda una línea JSON por orden; se contesta con una línea ">{...}"
// para que el lector de la laptop la distinga de los mensajes normales.
static String lineaSerie;

void responderSerie(const String& json) {
  Serial.print('>');
  Serial.println(json);
}

void procesarOrdenSerie(const String& linea) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, linea)) {
    responderSerie("{\"ok\":false,\"error\":\"json\"}");
    return;
  }
  if (doc.containsKey("estado")) {
    bool ok = aplicarEstado(String((const char*)(doc["estado"] | "")));
    responderSerie(ok ? "{\"ok\":true,\"orden\":\"estado\"}"
                      : "{\"ok\":false,\"orden\":\"estado\",\"error\":\"estado\"}");
  } else if (doc.containsKey("servo")) {
    int p = doc["servo"];
    bool ok = aplicarServo(p);
    responderSerie(ok ? "{\"ok\":true,\"orden\":\"servo\",\"posicion\":" + String(p) +
                        ",\"tiempo_giro_ms\":" + String(girandoHasta - millis()) + "}"
                      : "{\"ok\":false,\"orden\":\"servo\",\"error\":\"fuera de rango\"}");
  } else if (doc.containsKey("prueba_led")) {
    responderSerie("{\"ok\":true,\"orden\":\"prueba_led\"}");
    autoPruebaLeds(2);
  } else if (doc.containsKey("audio")) {
    enviarAudio = (int)(doc["audio"] | 0) != 0;
    audioPuerto = 0;
    bloqueAudio = 0;
    responderSerie(String("{\"ok\":true,\"orden\":\"audio\",\"activo\":") +
                   (enviarAudio ? "true" : "false") + ",\"muestras_hz\":" + String(I2S_MUESTRAS_HZ) +
                   ",\"muestras_por_bloque\":" + String(I2S_BLOQUE) + "}");
  } else if (doc.containsKey("prueba_i2s")) {
    responderSerie(pruebaI2S());
  } else if (doc.containsKey("lectura")) {
    responderSerie(lecturaJson());
  } else {
    responderSerie("{\"ok\":false,\"error\":\"orden desconocida\"}");
  }
}

void atenderSerie() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      lineaSerie.trim();
      if (lineaSerie.length()) procesarOrdenSerie(lineaSerie);
      lineaSerie = "";
    } else if (lineaSerie.length() < 300) {
      lineaSerie += c;
    }
  }
}

// ------------------------------------------------------------------ Rutas HTTP
void rutaEstado() {
  StaticJsonDocument<384> doc;
  if (deserializeJson(doc, servidor.arg("plain"))) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"json\"}");
    return;
  }
  if (!aplicarEstado(String((const char*)(doc["estado"] | "")))) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"estado\"}");
    return;
  }
  servidor.send(200, "application/json", "{\"ok\":true}");
}

void rutaServo() {
  StaticJsonDocument<64> doc;
  if (deserializeJson(doc, servidor.arg("plain")) || !doc.containsKey("posicion")) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"posicion\"}");
    return;
  }
  int p = doc["posicion"];
  if (!aplicarServo(p)) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"fuera de rango\"}");
    return;
  }
  servidor.send(200, "application/json",
    "{\"ok\":true,\"posicion\":" + String(p) + ",\"tiempo_giro_ms\":" + String(girandoHasta - millis()) + "}");
}

// La laptop pide el audio e indica su puerto UDP; el destino es quien lo pidió.
void rutaAudio() {
  StaticJsonDocument<96> doc;
  if (deserializeJson(doc, servidor.arg("plain"))) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"json\"}");
    return;
  }
  bool activo = (int)(doc["activo"] | 0) != 0;
  uint16_t puerto = doc["puerto"] | 0;
  if (activo && !puerto) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"puerto\"}");
    return;
  }
  audioDestino = servidor.client().remoteIP();
  audioPuerto = activo ? puerto : 0;
  audioHasta = millis() + ORDEN_LAPTOP_MS;
  bloqueAudio = 0;
  enviarAudio = activo;
  servidor.send(200, "application/json",
    String("{\"ok\":true,\"activo\":") + (activo ? "true" : "false") +
    ",\"muestras_hz\":" + String(I2S_MUESTRAS_HZ) + ",\"muestras_por_bloque\":" + String(I2S_BLOQUE) + "}");
}

void rutaLectura() {
  servidor.send(200, "application/json", lecturaJson());
}

void rutaPruebaLed() {
  servidor.send(200, "application/json", "{\"ok\":true,\"prueba\":\"verde, amarillo, rojo x2\"}");
  autoPruebaLeds(2);
}

void rutaRaiz() {
  servidor.send(200, "application/json",
    "{\"ok\":true,\"dispositivo\":\"semaforo-esp32\",\"rutas\":[\"POST /estado\",\"POST /servo\",\"POST /audio\",\"GET /lectura\",\"GET /prueba_led\"]}");
}

// ------------------------------------------------------------------ Envío a la laptop
// Corre en su propia tarea (núcleo 0): un POST lento o sin destino no frena el
// bucle que lee el micrófono, y reconectar el Wi-Fi tampoco. El bucle deja la
// lectura lista en jsonPorEnviar y avisa.
static SemaphoreHandle_t mutexEnvio = NULL;
static TaskHandle_t tareaEnvio = NULL;
static String jsonPorEnviar, urlPorEnviar;

void encolarLectura() {
  if (tareaEnvio == NULL) return;
  String json = lecturaJson();
  xSemaphoreTake(mutexEnvio, portMAX_DELAY);
  jsonPorEnviar = json;
  urlPorEnviar = laptopUrl;
  xSemaphoreGive(mutexEnvio);
  xTaskNotifyGive(tareaEnvio);
}

void enviarLectura(const String& json, const String& url) {
  HTTPClient http;
  http.setConnectTimeout(800);      // si la laptop no está, no bloquear el bucle
  http.setTimeout(800);             // (el bucle también atiende peticiones entrantes)
  http.setReuse(false);
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Connection", "close");
  int codigo = http.POST(json);
  http.end();
  if (codigo == 200) enviosOk++; else enviosFallidos++;
  if (enviosFallidos > 0 && enviosFallidos % 30 == 1 && codigo != 200) {
    Serial.printf("[laptop] no responde (%d). Sigo con control propio.\n", codigo);
  }
}

void bucleRed(void*) {
  unsigned long ultimoIntento = 0;
  for (;;) {
    bool hayLectura = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000)) > 0;
    if (WiFi.status() != WL_CONNECTED) {
      if (hayLectura) enviosFallidos++;
      if (millis() - ultimoIntento > 15000) {
        ultimoIntento = millis();
        wifiMulti.run();                   // vuelve a probar todas las redes conocidas
      }
      continue;
    }
    if (!hayLectura) continue;
    xSemaphoreTake(mutexEnvio, portMAX_DELAY);
    String json = jsonPorEnviar, url = urlPorEnviar;
    xSemaphoreGive(mutexEnvio);
    enviarLectura(json, url);
  }
}

// ------------------------------------------------------------------ Descubrimiento
// El ESP32 no necesita saber la IP de la laptop: la anuncia por UDP a toda la
// red y serve_dashboard.py le contesta con la dirección a la que enviar. Así el
// mismo firmware sirve en casa, en la universidad o con hotspot, sin regrabar.
void anunciarse() {
  if (WiFi.status() != WL_CONNECTED) return;
  // Difusión de la subred según su máscara: el hotspot de un iPhone es /28
  // (172.20.10.x), así que x.x.x.255 no llegaría a nadie.
  IPAddress ip = WiFi.localIP(), mascara = WiFi.subnetMask(), difusion;
  for (int i = 0; i < 4; i++) difusion[i] = ip[i] | (uint8_t)~mascara[i];
  String msg = "{\"dispositivo\":\"semaforo-esp32\",\"ip\":\"" + WiFi.localIP().toString() +
               "\",\"puerto\":80,\"led\":\"" + led + "\"}";
  udp.beginPacket(difusion, PUERTO_DESCUBRIMIENTO);
  udp.print(msg);
  udp.endPacket();
}

void escucharRespuesta() {
  int largo = udp.parsePacket();
  if (largo <= 0) return;
  char buf[256];
  int n = udp.read(buf, sizeof(buf) - 1);
  if (n <= 0) return;
  buf[n] = 0;
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, buf)) return;
  const char* url = doc["laptop_url"];
  if (!url || !strlen(url)) return;
  if (laptopUrl != url) {
    laptopUrl = url;
    memoria.putString("laptop_url", laptopUrl);
    Serial.print("[descubrimiento] laptop en "); Serial.println(laptopUrl);
  }
  laptopDescubierta = true;
}

// ------------------------------------------------------------------ Setup / loop
void setup() {
  Serial.setTxBufferSize(8192);   // el audio no debe frenar el bucle mientras sale por el cable
  Serial.begin(BAUDIOS_SERIE);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARILLO, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  autoPruebaLeds(2);          // al arrancar: verde, amarillo, rojo, dos veces

  servo.setPeriodHertz(50);
  servo.attach(PIN_SERVO, 500, 2400);
  anguloActual = anguloObjetivo = ANGULOS_SERVO[0];
  servo.write(anguloActual);

  iniciarI2S();

  memoria.begin("semaforo", false);
  laptopUrl = memoria.getString("laptop_url", LAPTOP_URL_INICIAL);
  arranques = memoria.getUInt("arranques", 0) + 1;
  memoria.putUInt("arranques", arranques);

  if (!MODO_WIFI) {
    WiFi.mode(WIFI_OFF);
    Serial.println();
    Serial.println("================================================");
    Serial.println("  Modo USB: hablo con la laptop por este cable.");
    Serial.println("  (para usar Wi-Fi: MODO_WIFI true en config.h)");
    Serial.println("================================================");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);       // sin ahorro de energía del radio: el audio no se entrecorta
  for (size_t i = 0; i < N_REDES; i++) wifiMulti.addAP(REDES[i].ssid, REDES[i].clave);
  Serial.printf("\nBuscando alguna de las %d redes conocidas", (int)N_REDES);
  unsigned long inicio = millis();
  while (wifiMulti.run() != WL_CONNECTED && millis() - inicio < ESPERA_WIFI_MS) {
    delay(300);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("================================================");
    Serial.print  ("  Red:           "); Serial.println(WiFi.SSID());
    Serial.print  ("  IP del ESP32:  "); Serial.println(WiFi.localIP());
    Serial.print  ("  Laptop:        "); Serial.println(laptopUrl);
    Serial.println("  (la laptop se descubre sola; serve_dashboard.py debe estar corriendo)");
    Serial.println("================================================");
  } else {
    Serial.println("\nSin Wi-Fi. El semaforo funciona solo por dB; reintento en segundo plano.");
  }
  udp.begin(PUERTO_DESCUBRIMIENTO);

  servidor.on("/",        HTTP_GET,  rutaRaiz);
  servidor.on("/lectura", HTTP_GET,  rutaLectura);
  servidor.on("/prueba_led", HTTP_GET, rutaPruebaLed);
  servidor.on("/estado",  HTTP_POST, rutaEstado);
  servidor.on("/servo",   HTTP_POST, rutaServo);
  servidor.on("/audio",   HTTP_POST, rutaAudio);
  servidor.begin();

  mutexEnvio = xSemaphoreCreateMutex();
  xTaskCreatePinnedToCore(bucleRed, "red", 8192, NULL, 1, &tareaEnvio, 0);
}

void loop() {
  atenderSerie();
  moverServoSuave();
  if (MODO_WIFI) servidor.handleClient();

  float db = medirDb();
  enviarBloqueAudio(ultimasMuestras);      // no hace nada si el audio está apagado o no llegó bloque
  if (db >= 0) {
    nivelDb = db;
    nivelSuave = 0.9f * nivelSuave + 0.1f * db;   // suavizado ~10 lecturas
  }

  // La orden de la laptop manda hasta que llegue otra. Solo si ORDEN_LAPTOP_MS
  // es mayor que cero caduca y el ESP32 vuelve a decidir por dB.
  bool ordenVigente = ordenLaptop != "" && (ORDEN_LAPTOP_MS == 0 || millis() < ordenHasta);
  if (!ordenVigente) {
    ordenLaptop = "";
    decidirPropio(nivelSuave);
  }
  if (enviarAudio && audioPuerto && ORDEN_LAPTOP_MS > 0 && millis() > audioHasta) {
    enviarAudio = false;                    // la laptop se fue: no gastar batería ni aire
    audioPuerto = 0;
    Serial.println("[audio] la laptop dejo de hablar; se corta el envio");
  }

  if (millis() - ultimoEnvio >= ENVIAR_CADA_MS) {
    ultimoEnvio = millis();
    responderSerie(lecturaJson());          // por el cable, siempre
    if (MODO_WIFI) encolarLectura();        // por la red, solo con Wi-Fi
  }
  if (!MODO_WIFI) return;

  // Descubrimiento: se anuncia seguido hasta que la laptop conteste, y después
  // de vez en cuando por si la laptop cambió de red o de IP.
  unsigned long cadaAnuncio = laptopDescubierta ? ANUNCIAR_CADA_MS * 10 : ANUNCIAR_CADA_MS;
  if (millis() - ultimoAnuncio >= cadaAnuncio) {
    ultimoAnuncio = millis();
    anunciarse();
  }
  escucharRespuesta();
}
