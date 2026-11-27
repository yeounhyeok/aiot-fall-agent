// UNO R4 WiFi: BLE IMU peripheral, USB serial fallback. No audio over BLE.
#include <ArduinoBLE.h>
#include <string.h>
BLEService imuService("7e516880-7e4e-48f1-a2ab-792dbae80300");
BLECharacteristic sample("7e516880-7e4e-48f1-a2ab-792dbae80301", BLENotify, 20, true);
#include <Wire.h>
const uint8_t ADDRESS = 0x68;  // AD0 low. Use 0x69 with AD0 high.
const unsigned long BAUD = 115200;
uint32_t seq = 0;

void fault() {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) { digitalWrite(LED_BUILTIN, HIGH); delay(250); digitalWrite(LED_BUILTIN, LOW); delay(250); }
}

void writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADDRESS); Wire.write(reg); Wire.write(value);
  if (Wire.endTransmission() != 0) fault();
}

void readRegs(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(ADDRESS); Wire.write(reg);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(ADDRESS, n) != n) fault();
  for (uint8_t i = 0; i < n; ++i) buf[i] = Wire.read();
}

void setup() {
  // Never wait for a serial terminal: power-bank boot is the normal path.
  Serial.begin(BAUD); Wire.begin(); Wire.setClock(100000);
#ifdef WIRE_HAS_TIMEOUT
  Wire.setWireTimeout(25000, true);
#endif
  delay(100);
  uint8_t id;
  readRegs(0x75, &id, 1); if (id != 0x68) fault();
  writeReg(0x6B, 0x80); delay(100);  // reset, then wake with PLL clock
  writeReg(0x6B, 0x01); delay(100);
  writeReg(0x1A, 3);     // DLPF on: 1 kHz gyro output rate before division
  writeReg(0x19, 9);     // 100 Hz
  writeReg(0x1B, 0x18);  // FS_SEL=3: +/-2000 deg/s
  writeReg(0x1C, 0x18);  // AFS_SEL=3: +/-16 g
  writeReg(0x38, 1);     // enable DATA_RDY; poll INT_STATUS, no INT wire
  if (!BLE.begin()) fault();
  BLE.setLocalName("Fall-IMU");
  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(sample);
  BLE.addService(imuService);
  BLE.setConnectionInterval(6, 12); // requested 7.5..15 ms; central decides
  BLE.advertise();
}

void put32(uint8_t *p, uint32_t x) {
  for (unsigned i=0; i<4; ++i) p[i] = (uint8_t)(x >> (8*i));
}

void loop() {
  BLE.poll();
  uint8_t ready, raw[14];
  readRegs(0x3A, &ready, 1);
  if (!(ready & 1)) return;
  uint32_t us = micros();
  readRegs(0x3B, raw, sizeof(raw));
  const uint32_t id = seq++;
  if (sample.subscribed()) {
    uint8_t p[20]; put32(p, id); put32(p+4, us);
    memcpy(p+8, raw, 6); memcpy(p+14, raw+8, 6); // skip temperature
    sample.writeValue(p, sizeof(p));
    return;
  }
  if (!Serial) return;
  const char hex[] = "0123456789ABCDEF";
  Serial.print(F("IMU1,")); Serial.print(id); Serial.print(',');
  Serial.print(us); Serial.print(',');
  for (uint8_t b : raw) { Serial.write(hex[b >> 4]); Serial.write(hex[b & 15]); }
  Serial.println();
}
