"""Offline device prompt playback. No microphone, TTS engine or alert dispatch."""
import argparse
from pathlib import Path
import subprocess
import wave

QUESTIONS = ('initial_check', 'clarify_status', 'offer_help')
CLIPS = QUESTIONS + ('raise_alert', 'close_event')
PACK = Path(__file__).resolve().parent / 'wav'


def resolve(route, previous=None):
    if route == 'repeat_last':
        if previous not in QUESTIONS:
            raise ValueError('repeat_last requires the previous question ID')
        return previous
    if route not in CLIPS:
        raise ValueError('unknown route')
    return route


def play(route, previous=None, pack=PACK, device=None, dry_run=False):
    path = Path(pack) / (resolve(route, previous) + '.wav')
    with wave.open(str(path), 'rb') as stream:
        if not (stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getnframes() > 0):
            raise ValueError('expected nonempty mono 16-bit PCM WAV')
    if not dry_run:
        args = ['aplay', '-q'] + (['-D', device] if device else []) + [str(path)]
        subprocess.run(args, check=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', choices=CLIPS + ('repeat_last',))
    parser.add_argument('--previous', choices=QUESTIONS)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--device', help='ALSA PCM device name')
    parser.add_argument('--dry-run', action='store_true', help='validate the asset without using a speaker')
    args = parser.parse_args()
    try:
        print(play(args.route, args.previous, args.pack, args.device, args.dry_run))
    except (ValueError, OSError, wave.Error, EOFError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'playback: {error}\n')
