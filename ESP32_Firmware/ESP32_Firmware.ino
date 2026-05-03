/*
 * IOT-RP: Final Optimized ESP32 Firmware
 * Component: IoT Node (Health Monitoring)
 * Sensors: MAX30105 (HR/SpO2), MAX30205 (Skin Temp)
 * Networking: Fernando Fallback + MQTT Streaming
 */

#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <WiFiManager.h>
#include <ArduinoJson.h>

// --- Configuration Storage ---
char mqtt_server[40] = "broker.hivemq.com";
char secondary_mqtt_server[40] = "broker.hivemq.com"; 
char patient_id[20] = "P001";
char secondary_ssid[32] = "Backup_SSID";
char secondary_pass[32] = "12345678";
bool shouldSaveConfig = false;

// --- Hardware ---
MAX30105 sensor;
const int32_t BUFFER_LENGTH = 100;
uint32_t irBuffer[BUFFER_LENGTH]; 
uint32_t redBuffer[BUFFER_LENGTH];
int32_t heartRate, spo2;
int8_t hrValid, spo2Valid;
const int MAX30205_ADDR = 0x48;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

void saveConfigCallback() {
    Serial.println("[WM] Config save triggered");
    shouldSaveConfig = true;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    Serial.print("[MQTT] Message arrived on topic: ");
    Serial.println(topic);

    DynamicJsonDocument doc(256);
    deserializeJson(doc, payload, length);

    if (doc.containsKey("type") && doc["type"].as<String>() == "config_update") {
        Serial.println("[CONFIG] Remote update received. Restarting...");
        delay(1000);
        ESP.restart();
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n\n--- IOT-RP NODE SMART START ---");
    Wire.begin(21, 22);
    
    Serial.println("\n--- I2C BUS SCAN ---");
    byte error, address;
    int nDevices = 0;
    for(address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();
        if (error == 0) {
            Serial.print("Device found at 0x");
            if (address<16) Serial.print("0");
            Serial.println(address, HEX);
            nDevices++;
        }
    }
    if (nDevices == 0) Serial.println("No I2C devices found!\n");
    else Serial.println("Scan complete.\n");

    if (sensor.begin(Wire)) {
        sensor.setup(0x1F, 4, 2, 100, 411, 4096);
        Serial.println("[HW] MAX30105 Vitals Sensor: ONLINE");
    } else {
        Serial.println("[ERROR] MAX30105 NOT FOUND! Check SDA(21)/SCL(22) wiring.");
    }

    // Quick connectivity pulse and initialization for temp sensor
    Wire.beginTransmission(MAX30205_ADDR);
    Wire.write(0x01); // Configuration Register
    Wire.write(0x00); // Normal operation (wake up)
    if (Wire.endTransmission() == 0) {
        Serial.println("[HW] MAX30205 Temp Sensor: ONLINE & INITIALIZED");
    } else {
        Serial.println("[ERROR] MAX30205 NOT FOUND! Check I2C address 0x48.");
    }

    WiFiManager wm;
    wm.setSaveConfigCallback(saveConfigCallback);

    WiFiManagerParameter custom_mqtt_server("server", "MQTT Broker IP", mqtt_server, 40);
    WiFiManagerParameter custom_patient_id("pid", "Patient ID", patient_id, 20);
    WiFiManagerParameter custom_sec_ssid("sec_ssid", "Backup SSID", secondary_ssid, 32);
    WiFiManagerParameter custom_sec_pass("sec_pass", "Backup Password", secondary_pass, 32);
    
    wm.addParameter(&custom_mqtt_server);
    wm.addParameter(&custom_patient_id);
    wm.addParameter(&custom_sec_ssid);
    wm.addParameter(&custom_sec_pass);

    if (!wm.autoConnect("IOT-RP-SETUP", "qwertyuio")) {
        Serial.println("[NET] WiFi Connection Timeout. Restarting...");
        delay(3000);
        ESP.restart();
    }

    strcpy(mqtt_server, custom_mqtt_server.getValue());
    strcpy(patient_id, custom_patient_id.getValue());
    strcpy(secondary_ssid, custom_sec_ssid.getValue());
    strcpy(secondary_pass, custom_sec_pass.getValue());

    Serial.println("[NET] Connected to WiFi!");
    mqttClient.setServer(mqtt_server, 1883);
    mqttClient.setCallback(mqttCallback);
}

void reconnect() {
    static unsigned long lastReconnectAttempt = 0;
    static int fallbackState = 0; // 0: Primary, 1: Secondary

    if (millis() - lastReconnectAttempt > 5000) {
        lastReconnectAttempt = millis();
        
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[NET] WiFi lost. Attempting failover...");
            if (fallbackState == 0 && strlen(secondary_ssid) > 0) {
                Serial.print("[NET] Trying Backup SSID: "); Serial.println(secondary_ssid);
                WiFi.persistent(false); // Prevents overwriting primary network on successful connection
                WiFi.begin(secondary_ssid, secondary_pass);
                fallbackState = 1;
            } else {
                Serial.println("[NET] Restarting for fresh connect search...");
                ESP.restart();
            }
            return;
        }

        // Choose the correct MQTT IP based on which Wi-Fi we are on
        if (fallbackState == 1) {
            mqttClient.setServer(secondary_mqtt_server, 1883);
            Serial.print("[MQTT] Attempting connection to BACKUP IP: "); Serial.println(secondary_mqtt_server);
        } else {
            mqttClient.setServer(mqtt_server, 1883);
            Serial.print("[MQTT] Attempting connection to PRIMARY IP: "); Serial.println(mqtt_server);
        }

        // Connect with Last Will and Testament (LWT) for heartbeat monitoring
        if (mqttClient.connect(patient_id, "r25_014/vitals/status", 1, true, "{\"status\":\"offline\"}")) {
            Serial.println("[MQTT] Connected & Status: ONLINE");
            mqttClient.publish("r25_014/vitals/status", "{\"status\":\"online\"}", true);
            mqttClient.subscribe("r25_014/config/update");
        } else {
            Serial.print("[MQTT] Failed, rc=");
            Serial.print(mqttClient.state());
            Serial.println(" will try again in 5s");
        }
    }
}

float readTemp() {
    static float lastValidTemp = 36.5; 
    
    Wire.beginTransmission(MAX30205_ADDR);
    Wire.write(0x00);
    byte err = Wire.endTransmission(false); // Use repeated start for stability
    
    if (err == 0) {
        Wire.requestFrom(MAX30205_ADDR, 2);
        if (Wire.available() == 2) {
            float newTemp = ((Wire.read() << 8) | Wire.read()) * 0.00390625;
            if (newTemp > 10.0 && newTemp < 50.0) {
                lastValidTemp = newTemp;
            }
        } else {
            Serial.println("[HW-WARN] Temp Sensor: No data available on request.");
        }
    } else {
        Serial.print("[HW-WARN] I2C Error: "); Serial.println(err);
        // If the bus is hung (error 4), try to force-restart it
        if (err == 4) {
            Wire.begin(21, 22);
            Wire.setClock(100000);
        }
    }
    
    return lastValidTemp;
}

void loop() {
    // --- COMPONENT 1: Proactive Signal Monitoring ---
    if (WiFi.status() == WL_CONNECTED) {
        int rssi = WiFi.RSSI();
        if (rssi < -85) { // Proactive fallback threshold
            Serial.print("[NET] Signal degradation detected: "); Serial.print(rssi); Serial.println(" dBm. Switching...");
            reconnect(); // Trigger fallback check
        }
    }

    if (!mqttClient.connected()) {
        reconnect();
    } else {
        mqttClient.loop();
    }

    sensor.check();
    while (sensor.available()) {
        for (int i = 1; i < BUFFER_LENGTH; i++) {
            irBuffer[i-1] = irBuffer[i];
            redBuffer[i-1] = redBuffer[i];
        }
        irBuffer[BUFFER_LENGTH-1] = sensor.getFIFOIR();
        redBuffer[BUFFER_LENGTH-1] = sensor.getFIFORed();
        sensor.nextSample();
    }

    static unsigned long lastMsg = 0;
    static int currentCrit = 0;
    
    // --- COMPONENT 5: Adaptive Bandwidth Allocation ---
    // Changed for Viva: True REAL-TIME transfer (every 0.5 seconds)
    unsigned long interval = 500; 

    if (millis() - lastMsg > interval) {
        lastMsg = millis();
        maxim_heart_rate_and_oxygen_saturation(irBuffer, BUFFER_LENGTH, redBuffer, &spo2, &spo2Valid, &heartRate, &hrValid);
        
        float t = readTemp();
        
        // --- HYBRID LATCHING LOGIC ---
        static int lastValidHR = 0;
        static int lastValidSpO2 = 0;
        int hr = 0;
        int ox = 0;
        
        // INSTANT FINGER-OFF DETECTION (Threshold: 10,000)
        // If IR is very low, the finger is off. Instantly clear memory and output 0.
        if (irBuffer[BUFFER_LENGTH-1] < 10000) {
            lastValidHR = 0;
            lastValidSpO2 = 0;
        } else {
            // FINGER ON: Update memory ONLY if the library produces a real, valid number
            if (heartRate > 30 && heartRate < 200) {
                lastValidHR = heartRate;
            }
            if (spo2 > 50 && spo2 <= 100) {
                lastValidSpO2 = spo2;
            }
            
            // Output the latched memory. This PREVENTS midway drops to 0!
            hr = lastValidHR;
            ox = lastValidSpO2;
        }
        if (hr > 220) hr = 0;
        if (ox > 100) ox = 100;
        
        bool isCrit = (ox > 0 && ox < 88) || (hr > 0 && (hr < 40 || hr > 150));
        
        static int critCounter = 0;
        if (isCrit) {
            critCounter++;
        } else {
            // Gradually reduce or reset to avoid triggering on random single spikes
            critCounter = 0; 
        }
        
        // DEBOUNCE: Only trigger a critical alert if it has been critical for 4 consecutive readings (2 seconds).
        // This prevents the "3000 alerts" bug caused by the sensor "locking on" when you first touch it.
        currentCrit = (critCounter >= 4) ? 1 : 0;

        // Format to match RPi Expectations: pid, hr, spo2, temp, crit
        DynamicJsonDocument doc(256);
        doc["pid"] = String(patient_id);
        doc["hr"] = hr;
        doc["spo2"] = ox;
        doc["temp"] = t;
        doc["crit"] = currentCrit;
        doc["rssi"] = WiFi.RSSI(); // Link Latency Tracking support
        
        char buffer[200];
        serializeJson(doc, buffer);
        
        if (mqttClient.connected()) {
            mqttClient.publish("r25_014/vitals/raw", buffer);
            if (currentCrit == 1) mqttClient.publish("r25_014/alerts/critical", buffer);
            Serial.print("[SEND-ADAPTIVE] "); Serial.println(buffer);
        } else {
            Serial.print("[LOCAL] "); Serial.println(buffer);
        }
    }
}