"""Use `python -m runtime live --help` from the repo root."""
import argparse
from pathlib import Path
from .controller import monitor


def main():
    p = argparse.ArgumentParser(description='Wired Fall Agent')
    sub = p.add_subparsers(dest='mode',required=True)
    rp = sub.add_parser('replay',help='offline scripted control-flow demonstration; sends nothing')
    rp.add_argument('scenario', choices=['help','clear','silence'])
    live = sub.add_parser('live')
    live.add_argument('--port', required=True, help='wired IMU serial port')
    live.add_argument('--asr-dir', required=True, type=Path, help='local Whisper-base model directory')
    live.add_argument('--pack', type=Path, default=Path(__file__).resolve().parents[1]/'device_audio'/'wav')
    live.add_argument('--endpoint', default='http://127.0.0.1:8080')
    live.add_argument('--mic'); live.add_argument('--speaker')
    live.add_argument('--max-turns',type=int,choices=range(1,11),default=4)
    live.add_argument('--once',action='store_true'); live.add_argument('--discord',action='store_true',help='explicitly allow a real alert')
    args = p.parse_args()
    if args.mode == 'replay':
        from .replay import Replay
        sim = Replay(args.scenario)
        return monitor(sim,sim,one=True,alerts=True)
    from .imu import WiredIMU
    from .dialogue import LocalDialogue
    io = LocalDialogue(pack=args.pack, asr_dir=args.asr_dir, endpoint=args.endpoint, mic=args.mic, speaker=args.speaker)
    io.check()
    with WiredIMU(port=args.port) as sensor:
        return monitor(sensor,io,one=args.once,max_turns=args.max_turns,alerts=args.discord)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
