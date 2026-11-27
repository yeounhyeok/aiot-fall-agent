"""Local Korean speech and step-592 dialogue adapters. No model downloads or event loop."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from urllib.parse import urlsplit


ASR_REVISION = 'e37978b90ca9030d5170a5c07aadb050351a65bb'
# SHA-256 of the weight blob and Git blob IDs of the processor/config assets
# in openai/whisper-base at ASR_REVISION (also works with copied snapshots).
WEIGHTS_SHA256 = '07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b'
ASSETS = {
    'added_tokens.json': 'e3d256c988462aa153dcabe2aa38b8e9b436c06f',
    'config.json': '708cca09e7e8a9364e444efe8b4f9c2691fc5f51',
    'generation_config.json': '38eda87a0b22fdb85af5dfa14cf065147337137a',
    'merges.txt': '6038932a2a1f09a66991b1c2adae0d14066fa29e',
    'normalizer.json': 'dd6ae819ad738ac1a546e9f9282ef325c33b9ea0',
    'preprocessor_config.json': 'c2048dfa9fd94a052e62e908d2c4dfb18534b4d2',
    'special_tokens_map.json': 'bf69932dca4b3719b59fdd8f6cc1978109509f6c',
    'tokenizer.json': '1e95340ff836fad1b5932e800fb7b8c5e6d78a74',
    'tokenizer_config.json': 'd13b786c04765fb1a06492b53587752cd67665ea',
    'vocab.json': 'd7016e21da8776c8a9d577d0f559600f09a240eb',
}
GGUF = 'Fall-Agent-Qwen3.5-2B-step592-Q4_K_M.gguf'
SYSTEM = '낙상 확인 장치의 관측 대화(history)를 읽고 마지막 발화에 근거해 JSON 객체 하나만 출력하세요. evidence는 관측 발화의 근거, call은 name과 arguments.reason, next_prompt는 후속 질문 경로입니다. 도움 요청은 raise_alert/help_requested, 오경보 확인은 close_event/false_alarm, 불명확하면 ask_checkin/unclear_reply를 사용하세요. ask_checkin의 경로는 clarify_status, repeat_last, offer_help 중 하나이고 그 밖에는 null입니다. 관측되지 않은 말을 추측하지 마세요.'
RATE = 16000
LISTEN_SECONDS = 5
ROUTES = ('clarify_status', 'repeat_last', 'offer_help')


class DialogueError(RuntimeError):
    """Unresolved adapter/model failure; never an actionable event."""


def verify_asr(directory):
    directory = Path(directory)
    for name, expected in ASSETS.items():
        data = (directory / name).read_bytes()
        if hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() != expected:
            raise DialogueError(f'wrong Whisper asset: {name}')
    digest = hashlib.sha256()
    with (directory / 'model.safetensors').open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise DialogueError('wrong Whisper weights')


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def make_prompt(history):
    if not isinstance(history, (list, tuple)) or not history:
        raise DialogueError('history must contain an observed turn')
    observed = []
    keys = ('sensor', 'device', 'transcript')
    for row in history[-2:]:
        if not isinstance(row, dict) or set(row) != set(keys):
            raise DialogueError('invalid observed turn fields')
        for key in keys:
            value = row[key]
            if (not isinstance(value, str) or len(value) > 1024 or '<|' in value or
                    (key != 'transcript' and not value.strip())):
                raise DialogueError('invalid observation')
        observed.append({key: row[key] for key in keys})
    user = compact({'history': observed})
    return (f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def parse_object(raw):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise DialogueError('duplicate JSON key')
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, RecursionError, TypeError):
        raise DialogueError('invalid JSON') from None
    if not isinstance(value, dict):
        raise DialogueError('expected JSON object')
    return value


def validate(decision):
    if (not isinstance(decision, dict) or set(decision) != {'evidence', 'call', 'next_prompt'} or
            not isinstance(decision['evidence'], str) or not decision['evidence'].strip()):
        raise DialogueError('invalid decision fields')
    call, route = decision['call'], decision['next_prompt']
    for name, reason in (('ask_checkin', 'unclear_reply'), ('close_event', 'false_alarm'),
                         ('raise_alert', 'help_requested')):
        if call == {'name': name, 'arguments': {'reason': reason}}:
            if (route in ROUTES if name == 'ask_checkin' else route is None):
                return decision
    raise DialogueError('invalid call, reason or follow-up route')


def alsa(tool, device, *args, data=None, timeout=10):
    cmd = [tool, '-q', *(['-D', device] if device else []), *args]
    try:
        return subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              check=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        raise DialogueError(f'{tool} failed or timed out') from None


class LocalDialogue:
    def __init__(self, *, pack, asr_dir, endpoint, mic=None, speaker=None):
        u = urlsplit(endpoint)
        if (u.scheme != 'http' or u.hostname not in ('localhost', '127.0.0.1', '::1') or
                u.username is not None or u.password is not None or u.query or u.fragment or
                u.path not in ('', '/')):
            raise ValueError('expected a loopback HTTP origin, without credentials or path')
        self.host = '::1' if u.hostname == '::1' else '127.0.0.1'
        self.port = 80 if u.port is None else u.port
        self.pack, self.asr_dir = Path(pack), Path(asr_dir).resolve()
        self.mic, self.speaker = mic, speaker
        self._asr = None
        self._last_question = None

    def _request(self, path, payload=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=90 if payload else 4)
        try:
            body = None if payload is None else compact(payload).encode()
            conn.request('GET' if body is None else 'POST', path, body,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise DialogueError(f'llama-server {path}: HTTP {response.status}')
            raw = response.read(131073)
            if len(raw) > 131072:
                raise DialogueError('oversized llama-server response')
            return parse_object(raw)
        except (OSError, http.client.HTTPException):
            raise DialogueError(f'llama-server {path} unavailable') from None
        finally:
            conn.close()

    def _check_server(self):
        if self._request('/health').get('status') != 'ok':
            raise DialogueError('llama-server not ready')
        path = self._request('/props').get('model_path')
        if not isinstance(path, str) or Path(path).name != GGUF:
            raise DialogueError('unexpected llama-server model')

    def _load_asr(self):
        if self._asr is None:
            verify_asr(self.asr_dir)
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(self.asr_dir, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.asr_dir, use_safetensors=True, local_files_only=True).float().eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._asr = (processor, model.to(device), device)
        return self._asr

    def check(self):
        """Validate assets and open actual configured PCMs (brief silence; no dispatch)."""
        from device_audio import play
        for clip in play.CLIPS:
            play.play(clip, pack=self.pack, dry_run=True)
        self._check_server()
        self._load_asr()
        self._capture(1600)
        alsa('aplay', self.speaker, '-t', 'raw', '-f', 'S16_LE', '-r', str(RATE), '-c', '1',
             data=bytes(3200), timeout=4)

    def say(self, clip_id):
        from device_audio import play
        clip = play.resolve(clip_id, self._last_question)
        # Run the repository player synchronously with a deadline even for a stuck PCM.
        cmd = [sys.executable, play.__file__, clip, '--pack', str(self.pack)]
        if self.speaker:
            cmd += ['--device', self.speaker]
        try:
            with subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) as player:
                try:
                    status = player.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(player.pid, signal.SIGKILL)  # include the player's aplay child
                    raise
                if status:
                    raise DialogueError('prompt playback failed')
        except (OSError, subprocess.SubprocessError):
            raise DialogueError('prompt playback failed or timed out') from None
        self._last_question = clip if clip in play.QUESTIONS else None

    def _capture(self, frames):
        data = alsa('arecord', self.mic, '-t', 'raw', '-f', 'S16_LE', '-r', str(RATE),
                    '-c', '1', '-s', str(frames), '-', timeout=frames / RATE + 4)
        if len(data) != frames * 2:
            raise DialogueError('incomplete microphone capture')
        return data

    def listen(self):
        return self.transcribe_pcm(self._capture(RATE * LISTEN_SECONDS))

    def transcribe_pcm(self, data):
        """Decode mono 16 kHz signed little-endian PCM in RAM, without writing audio."""
        import numpy as np
        if len(data) % 2 or len(data) > 2 * RATE * 30:
            raise DialogueError('invalid PCM length')
        samples = np.frombuffer(data, dtype='<i2').astype('float32') / 32768.0
        if not samples.size or np.sqrt(np.mean(samples ** 2)) <= 1e-4:
            return ''
        processor, model, device = self._load_asr()
        import torch
        with torch.inference_mode():
            features = processor(samples, sampling_rate=RATE, return_tensors='pt',
                                 return_attention_mask=True).to(device)
            ids = model.generate(**features, language='ko', task='transcribe', do_sample=False,
                                 num_beams=1, max_new_tokens=128, return_timestamps=False)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def decide(self, history):
        prompt = make_prompt(history)
        self._check_server()  # check before disclosing observations, even without check()
        reply = self._request('/completion', {'prompt': prompt, 'temperature': 0, 'seed': 42,
                                              'n_predict': 160, 'repeat_penalty': 1.0,
                                              'stream': False, 'cache_prompt': False})
        if (reply.get('stopped_limit') or reply.get('truncated') or
                reply.get('stop_type') == 'limit' or not isinstance(reply.get('content'), str)):
            raise DialogueError('incomplete llama-server completion')
        decision = validate(parse_object(reply['content']))
        if not history[-1]['transcript'].strip() and decision['call']['name'] != 'ask_checkin':
            raise DialogueError('terminal decision on silence')
        return decision

    def notify(self, decision):
        from notifications.discord_alert import notify_decision
        return notify_decision(validate(decision))
