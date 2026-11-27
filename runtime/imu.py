"""Wired MPU6050 -> frozen CPU CNN -> Fall Agent sensor interface."""
import argparse
from collections import deque
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import struct
import threading
import time


DETECTOR = Path(__file__).resolve().parents[1] / 'detector'
SHA256 = '65e192e9b62cbd7458219029458fe7a2eb9811232def307d637bbe7402c592e4'
U32 = 0xffffffff
FRAME = re.compile(rb'IMU1,(\d{1,10}),(\d{1,10}),([0-9A-Fa-f]{28})\r?\n')
WINDOW, STRIDE, RECENT = 300, 25, 50
STALE, STARTUP = 0.75, 6.0


class SensorError(RuntimeError):
    """Acquisition is unavailable; not evidence of inactivity."""


def decode_burst(raw):
    """ACCEL_XOUT_H..GYRO_ZOUT_L, big-endian int16; temperature excluded."""
    if len(raw) != 14:
        raise ValueError('expected a 14-byte MPU6050 burst')
    ax, ay, az, _, gx, gy, gz = struct.unpack('>7h', raw)
    return (ax / 2048, ay / 2048, az / 2048, gx / 16.4, gy / 16.4, gz / 16.4)


def parse_frame(line):
    match = FRAME.fullmatch(line)
    if not match:
        raise ValueError('invalid IMU1 serial frame')
    seq, us = map(int, match.groups()[:2])
    if max(seq, us) > U32:
        raise ValueError('counter overflow')
    return seq, us, decode_burst(bytes.fromhex(match[3].decode('ascii')))


class Checkpoint:
    def __init__(self, detector_dir):
        import numpy as np
        import torch
        root = Path(detector_dir)
        weights = root / 'weights/net_synth_v1.pt'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != SHA256:
            raise ValueError('unexpected synth_v1 checkpoint SHA-256')
        # Use the released feature function and Net rather than maintaining a copy.
        spec = importlib.util.spec_from_file_location('_fall_detector', root / 'scripts/train_eval.py')
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        if (model.FS, model.W, model.STRIDE) != (100, WINDOW, STRIDE):
            raise ValueError('incompatible detector cadence')
        torch.set_num_threads(1)
        self.net = model.Net().cpu().eval()
        self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.np, self.torch, self.feats = np, torch, model.feats

    def __call__(self, rows):
        x = self.np.asarray(rows, dtype=self.np.float32)
        features = self.feats(x[:, :3], x[:, 3:])
        with self.torch.inference_mode():
            p = self.net(self.torch.from_numpy(features)[None]).sigmoid().item()
        if not math.isfinite(p):
            raise SensorError('non-finite checkpoint output')
        return p


class WindowStream:
    """Single writer. Never pad, bridge a loss, or infer from host arrival rate."""
    def __init__(self, score):
        self.score = score
        self.reset()

    def reset(self):
        self.rows = deque(maxlen=WINDOW)
        self.ticks = deque(maxlen=WINDOW)
        self.prev = None
        self.count = 0

    def push(self, frame):
        seq, us, axes = frame
        if self.prev is not None:
            old_seq, old_us = self.prev
            if seq != ((old_seq + 1) & U32) or not 8000 <= ((us - old_us) & U32) <= 12000:
                self.reset()
        self.prev = seq, us
        self.ticks.append(us)
        self.rows.append(axes)
        self.count += 1
        if self.count >= WINDOW and (self.count - WINDOW) % STRIDE == 0:
            elapsed = (us - self.ticks[0]) & U32
            if abs(elapsed - (WINDOW - 1) * 10000) > (WINDOW - 1) * 300:
                self.reset()  # each interval looked plausible, but the rate drifted >3%
                return None
            return self.score(self.rows)
        return None


def _open_port(port, baudrate):
    import serial  # replay / decoder tests don't need hardware libraries
    return serial.Serial(port=port, baudrate=baudrate, timeout=0.10, exclusive=True)


class WiredIMU:
    """Single-consumer context manager. The reader keeps running during dialogue."""
    def __init__(self, port, *, baudrate=115200, detector_dir=DETECTOR):
        self.port, self.baudrate, self.detector_dir = port, baudrate, detector_dir
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.thread = None
        self.error = self.last = self.pending = None
        self.total = 0
        self.armed, self.active = True, False
        self.unlock_at = 0.0

    def _open_transport(self):
        return _open_port(self.port, self.baudrate)

    def __enter__(self):
        if self.thread is not None:
            raise RuntimeError('use a new WiredIMU context')
        self.stream = WindowStream(Checkpoint(self.detector_dir))
        self.serial = self._open_transport()
        try:
            self.serial.reset_input_buffer()
            self.started = time.monotonic()
            self.thread = threading.Thread(target=self._read, name='wired-imu', daemon=True)
            self.thread.start()
        except BaseException:
            self.serial.close()
            raise
        return self

    def __exit__(self, *_):
        with self.cv:
            self.stop.set()
            self.cv.notify_all()
        try:
            self.thread.join(timeout=2)
        finally:
            self.serial.close()
        if self.thread.is_alive():
            raise SensorError('sampler did not stop')

    def _check(self):
        if self.error is not None:
            raise SensorError('IMU acquisition failed: ' + str(self.error)) from self.error
        if self.thread is None or self.stop.is_set():
            raise SensorError('wired IMU is not running')
        if time.monotonic() > (self.started + STARTUP if self.last is None else self.last + STALE):
            raise SensorError('IMU stream is stale')

    def _read(self):
        try:
            last_score = self.started
            while not self.stop.is_set():
                if self.serial.in_waiting > 1024:
                    raise SensorError('serial backlog; samples are not live')
                line = self.serial.read_until(b'\n', 80)
                with self.cv:
                    if self.stop.is_set():
                        return
                    self._check()
                    if time.monotonic() - last_score > STARTUP + WINDOW / 100:
                        raise SensorError('no contiguous 100 Hz inference window')
                    if not line:
                        continue
                    if line.startswith(b'ERR,'):
                        raise SensorError('firmware reports an I2C fault')
                    try:
                        frame = parse_frame(line)
                    except ValueError:
                        self.stream.reset()
                        self.pending = None
                        continue
                    now = time.monotonic()
                    p = self.stream.push(frame)
                    if self.stream.count <= 1:
                        self.pending = None
                    self.last = now
                    self.total += 1
                    if p is not None:
                        last_score = now
                        self._consider(p, now)
                    self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                self.error = exc
                self.cv.notify_all()

    def _consider(self, score, now):
        if not self.active and now >= self.unlock_at:
            if score <= 0.35:
                self.armed = True
            elif self.armed and score > 0.5:
                self.pending = now, score  # one-slot mailbox, never a trace queue
                self.armed = False

    def wait_candidate(self):
        """Block until a fresh rising candidate. Return its score, not a diagnosis."""
        with self.cv:
            if self.active:
                raise RuntimeError('rearm before waiting for another candidate')
            while True:
                self._check()
                if self.pending is not None:
                    when, p = self.pending
                    self.pending = None
                    if time.monotonic() - when <= STALE:
                        self.active = True
                        return p
                self.cv.wait(0.1)

    def snapshot(self):
        """Observe half a second after the call, independent of the triggering score."""
        with self.cv:
            start = self.total
            deadline = time.monotonic() + (STARTUP if self.last is None else 0) + 2
            while True:
                self._check()
                if self.total - start >= RECENT and self.stream.count >= RECENT:
                    rows = list(self.stream.rows)[-RECENT:]
                    break
                if time.monotonic() >= deadline:
                    raise SensorError('no contiguous follow-up observation')
                self.cv.wait(0.1)
        peak = max(math.hypot(*r[:3]) for r in rows)
        gyro = sum(math.hypot(*r[3:]) for r in rows) / RECENT
        return f'최근 0.5초 측정: 최대 가속도 크기 {peak:.2f} g, 평균 각속도 크기 {gyro:.1f} °/s.'

    def rearm(self):
        with self.cv:
            self._check()
            self.pending = None
            self.active = self.armed = False
            # A full window must age out, then a low score must precede a new high.
            self.unlock_at = time.monotonic() + 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', required=True)
    ap.add_argument('--baudrate', type=int, default=115200)
    ap.add_argument('--detector-dir', type=Path, default=DETECTOR)
    ap.add_argument('--observe', action='store_true', help='read one live observation without waiting for a candidate')
    args = ap.parse_args()
    with WiredIMU(args.port, baudrate=args.baudrate, detector_dir=args.detector_dir) as sensor:
        if not args.observe:
            print(f'fall_candidate p={sensor.wait_candidate():.6f}', flush=True)
        print(sensor.snapshot())


if __name__ == '__main__':
    main()
