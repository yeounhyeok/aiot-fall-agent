"""Hardware-free wire, streaming, controller-interface and frozen checkpoint tests."""
import os
from pathlib import Path
import struct
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from runtime import imu

FIXTURES = Path(__file__).resolve().parents[1] / 'tests/fixtures/imu'
DETECTOR = Path(os.environ.get('FALL_DETECTOR_DIR', imu.DETECTOR))
REST = (0, 0, 1, 0, 0, 0)


def archive(kind):
    # Stored team-generated trajectories, not generated signals or training windows.
    raw = np.load(FIXTURES / (kind + '.npy'), allow_pickle=False)
    assert raw.shape == (2100, 6) and raw.dtype == np.float32
    return raw.reshape(-1, 5, 6).mean(axis=1)


def wire(seq, axes):
    scale = [2048, 2048, 2048, 16.4, 16.4, 16.4]
    counts = np.rint(np.asarray(axes) * scale).clip(-32768, 32767).astype(int)
    raw = struct.pack('>7h', *counts[:3], -1111, *counts[3:])
    return f'IMU1,{seq},{seq * 10000 & imu.U32},{raw.hex()}\r\n'.encode()


class WireTests(unittest.TestCase):
    def test_signed_registers_units_and_temperature_skip(self):
        line = b'IMU1,42,1234,f800000008007fff00a4ff5c0000\r\n'
        seq, us, axes = imu.parse_frame(line)
        self.assertEqual((seq, us), (42, 1234))
        np.testing.assert_allclose(axes, (-1, 0, 1, 10, -10, 0))
        self.assertEqual(imu.decode_burst(bytes.fromhex('8000' * 7))[0], -16)

    def test_invalid_frames(self):
        good = wire(1, REST)
        for bad in [b'', good[:-2], good.replace(b'IMU1', b'IMU2'), good + b'\n',
                    good.replace(b',1,', b',4294967296,'), good.replace(b',1,', b',-1,'),
                    good[:-3] + b'Z\n', good.replace(b'10000,', b'10000,00')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                imu.parse_frame(bad)
        with self.assertRaises(ValueError):
            imu.decode_burst(bytes(13))

    def test_gaps_duplicates_jitter_and_wrap(self):
        for seq_delta, us_delta in [(2, 10000), (0, 10000), (1, 50000), (1, 1000)]:
            s = imu.WindowStream(lambda _: 1.0)
            for i in range(299):
                self.assertIsNone(s.push((i, i * 10000, REST)))
            self.assertIsNone(s.push((298 + seq_delta, 2980000 + us_delta, REST)))
            self.assertEqual(s.count, 1)
        s = imu.WindowStream(lambda _: 1.0)
        for i in range(326):
            p = s.push(((imu.U32-100+i) & imu.U32, (imu.U32-1000000+i*10000) & imu.U32, REST))
            self.assertEqual(p is not None, i in (299, 324))
        self.assertEqual(len(s.rows), 300)

    def test_consistent_but_wrong_device_rate(self):
        s = imu.WindowStream(lambda _: self.fail('wrong rate reached the model'))
        for i in range(600):
            self.assertIsNone(s.push((i, i*11000, REST)))

    def test_threshold_rearm_and_dialogue_suppression(self):
        s = imu.WiredIMU('unused')
        s.thread = object()  # no worker; drive score transitions and clock explicitly
        with patch.object(imu.time, 'monotonic', return_value=10):
            s.last = 10
            s._consider(0.5, 10)
            self.assertIsNone(s.pending)
            s._consider(0.51, 10)
            self.assertEqual(s.wait_candidate(), 0.51)
            s._consider(0.1, 10)
            s._consider(1, 10)
            self.assertIsNone(s.pending)
            s.rearm()
        for p, t in [(0, 12), (1, 13), (0.36, 14), (1, 15)]:
            s._consider(p, t)
            self.assertIsNone(s.pending)
        s._consider(0.35, 16)
        s._consider(0.51, 17)
        self.assertEqual(s.pending, (17, 0.51))


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.score = imu.Checkpoint(DETECTOR)

    def test_archived_stream_scores_and_first_candidate(self):
        for kind, expected, is_fall in [('stand', 0.0000430603, False), ('topple', 0.999999523, True)]:
            for over_wire in (False, True):
                with self.subTest(kind=kind, wire=over_wire):
                    stream = imu.WindowStream(self.score)
                    scores = []
                    for i, row in enumerate(archive(kind)):
                        frame = imu.parse_frame(wire(i, row)) if over_wire else (i, i*10000, row)
                        p = stream.push(frame)
                        if p is not None:
                            scores.append((i, p))
                    self.assertEqual([i for i, _ in scores], [299, 324, 349, 374, 399])
                    self.assertEqual([p > 0.5 for _, p in scores], [is_fall] * 5)
                    if not over_wire:
                        self.assertAlmostEqual(max(p for _, p in scores), expected, delta=1e-6)

    @unittest.skipUnless(os.environ.get('SISFALL_ROOT'), 'optional: provide local SisFall archive')
    def test_measured_record_replay(self):
        # No public records are distributed. The original model misses this fall:
        # this guards score parity, not an assumption that every fall is detected.
        for name, expected in [('F06_SA16_R01', .380829), ('D01_SA16_R01', .000136)]:
            src = Path(os.environ['SISFALL_ROOT']) / 'SA16' / (name + '.txt')
            with src.open() as f:
                raw = np.array([[float(v) for v in line.strip().rstrip(';').split(',')[:6]]
                                for line in f if line.strip()])
            raw *= [32/2**13]*3 + [4000/2**16]*3
            rows = raw[:len(raw)//2*2].reshape(-1, 2, 6).mean(axis=1).astype(np.float32)
            stream = imu.WindowStream(self.score)
            scores = [p for i, row in enumerate(rows)
                      if (p := stream.push((i, i*10000, row))) is not None]
            self.assertAlmostEqual(max(scores), expected, delta=2e-5)

    def test_real_serial_reader_candidate_followup_and_stall(self):
        # A POSIX pseudo-terminal exercises pySerial framing, not just a fake readline.
        master, slave = os.openpty()
        self.addCleanup(os.close, slave)
        self.addCleanup(os.close, master)
        follow = threading.Event()
        stopped = threading.Event()
        def device():
            i = 0
            for row in archive('topple'):
                if stopped.is_set() or follow.is_set():
                    break
                msg = wire(i, row)
                os.write(master, msg[:8]); os.write(master, msg[8:])
                i += 1
                time.sleep(0.01)
            while not stopped.is_set() and i < 800:
                os.write(master, wire(i, REST))
                i += 1
                time.sleep(0.01)
        with imu.WiredIMU(os.ttyname(slave), detector_dir=DETECTOR) as sensor:
            writer = threading.Thread(target=device, daemon=True)
            writer.start()
            try:
                self.assertGreater(sensor.wait_candidate(), 0.5)
                follow.set()
                sensor.snapshot()  # allow the last in-flight packet to drain
                self.assertEqual(sensor.snapshot(), '최근 0.5초 측정: 최대 가속도 크기 1.00 g, 평균 각속도 크기 0.0 °/s.')
                sensor.rearm()
            finally:
                stopped.set()
                writer.join(1)
            with self.assertRaisesRegex(imu.SensorError, 'stale'):
                sensor.snapshot()
        self.assertFalse(sensor.thread.is_alive())
        self.assertFalse(sensor.serial.is_open)

    def test_serial_disconnect(self):
        master, slave = os.openpty()
        self.addCleanup(os.close, slave)
        with imu.WiredIMU(os.ttyname(slave), detector_dir=DETECTOR) as sensor:
            os.close(master)  # real PTY hangup, not silence
            with self.assertRaises(imu.SensorError):
                sensor.wait_candidate()

    def test_i2c_fault_propagates_to_waiter(self):
        master, slave = os.openpty()
        self.addCleanup(os.close, slave)
        self.addCleanup(os.close, master)
        with imu.WiredIMU(os.ttyname(slave), detector_dir=DETECTOR) as sensor:
            os.write(master, b'ERR,I2C\r\n')
            with self.assertRaisesRegex(imu.SensorError, 'I2C fault'):
                sensor.wait_candidate()
            with self.assertRaisesRegex(imu.SensorError, 'I2C fault'):
                sensor.snapshot()


if __name__ == '__main__':
    unittest.main()
