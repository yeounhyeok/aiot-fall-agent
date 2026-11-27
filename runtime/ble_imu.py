"""BLE GATT transport for the existing validated IMU1 window/CNN reader."""
import asyncio
import queue
import struct
import threading
import time

from .imu import WiredIMU, SensorError

SERVICE = '7e516880-7e4e-48f1-a2ab-792dbae80300'
SAMPLE = '7e516880-7e4e-48f1-a2ab-792dbae80301'


def to_imu1(payload):
    if len(payload) != 20:
        raise SensorError('invalid BLE IMU frame length')
    seq, us = struct.unpack_from('<II', payload)
    # Restore the unused temperature slot for the shared MPU6050 decoder.
    raw = bytes(payload[8:14]) + bytes(2) + bytes(payload[14:20])
    return f'IMU1,{seq},{us},{raw.hex()}\n'.encode('ascii')


class BLEPort:
    def __init__(self, address):
        self.address = address
        self.frames = queue.Queue(maxsize=16)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True, name='ble-notifications')
        self.thread.start()
        if not self.ready.wait(32) or self.error:
            self.close()
            raise SensorError('BLE subscription failed: ' + str(self.error or 'timeout'))

    def _worker(self):
        try:
            asyncio.run(self._receive())
        except Exception as exc:
            self.error = exc
        finally:
            self.ready.set()

    async def _receive(self):
        from bleak import BleakClient, BleakScanner
        device = await BleakScanner.find_device_by_address(self.address, timeout=10)
        if device is None:
            raise SensorError('board not found')
        disconnected = asyncio.Event()
        def receive(_, data):
            if self.error is not None or self.stop.is_set():
                return
            try:
                self.frames.put_nowait((time.monotonic(), to_imu1(data)))
            except (queue.Full, SensorError):
                self.error = SensorError('BLE backlog or malformed frame')
        async with BleakClient(device, disconnected_callback=lambda _: disconnected.set(), timeout=12) as client:
            service = client.services.get_service(SERVICE)
            char = service.get_characteristic(SAMPLE) if service else None
            if char is None or 'notify' not in char.properties:
                raise SensorError('IMU notification characteristic missing')
            await asyncio.wait_for(client.start_notify(char, receive), 5)
            self.ready.set()
            while not self.stop.is_set() and self.error is None:
                if disconnected.is_set() or not client.is_connected:
                    raise SensorError('BLE link lost')
                await asyncio.sleep(0.03)

    def reset_input_buffer(self):
        while True:
            try: self.frames.get_nowait()
            except queue.Empty: return

    @property
    def in_waiting(self):
        return 0  # bounded queue + frame age below replace serial byte backlog

    def read_until(self, delimiter, size):
        if self.error is not None:
            raise SensorError('BLE sensor unavailable') from self.error
        try:
            arrival, line = self.frames.get(timeout=0.10)
        except queue.Empty:
            return b''
        if time.monotonic() - arrival > 0.20:
            raise SensorError('stale BLE queue')
        return line

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)


class BluetoothIMU(WiredIMU):
    def _open_transport(self):
        return BLEPort(self.port)
