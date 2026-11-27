# Local speech and dialogue

`runtime.dialogue.LocalDialogue` is an adapter for the event loop, not a fall detector or a standalone alert system. Put `runtime/` beside the existing `device_audio/` and `notifications/` directories and run with the repository root on the Python path. No files in the private voice pack, recordings or model weights belong in Git.

## Offline provisioning

- Supply the five mono 16-bit PCM prompt WAVs (`initial_check`, `clarify_status`, `offer_help`, `raise_alert`, `close_event`) in `device_audio/wav/`. `say()` waits for the existing WAV player to finish. `repeat_last` repeats only a successfully played question.
- Supply ALSA `aplay` and `arecord` and select usable speaker and microphone PCMs (`aplay -L`, `arecord -L`). `check()` opens both: it discards a short capture and plays silence. This checks that a PCM opens, not that a person can hear the speaker or that the correct microphone is selected.
- Pre-stage a local `openai/whisper-base` snapshot at revision `e37978b90ca9030d5170a5c07aadb050351a65bb`, including real `model.safetensors` bytes, tokenizer and processor assets (see `ASSETS` in the module). Copy symlink targets as well, not just Hub snapshot links. Startup checks the digests and loads local files only. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The Python environment needs compatible NumPy, torch and Transformers; `arecord` captures five seconds of 16 kHz mono S16_LE PCM straight into memory. Digital/near silence returns an empty string; a capture error is not silence. Noise can still produce a hallucinated transcript.
- Pre-stage the separately built `Fall-Agent-Qwen3.5-2B-step592-Q4_K_M.gguf`. Verify its SHA-256 against the build's trusted manifest before starting `llama-server -m /models/Fall-Agent-Qwen3.5-2B-step592-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 2048 -np 1`. Disable server prompt/request-body logging. The adapter checks `GET /health` and the `model_path` basename in `GET /props`; a filename is **not** a digest attestation. Router-mode servers and `/v1/chat/completions` proxies are not supported.

```python
from runtime.dialogue import LocalDialogue

io = LocalDialogue(pack='device_audio/wav', asr_dir='/models/whisper-base',
                   endpoint='http://127.0.0.1:8080', mic=None, speaker=None)
io.check()  # checks local resources only; no Discord send
```

## Event-loop contract

After `say(clip_id)` succeeds, record the **actual spoken question text**, sensor observation and result of `listen()` as `{'sensor': ..., 'device': ..., 'transcript': ...}`. `decide(history)` serializes the last two observed rows (one if only one exists; never pads fictitious replies). Prompt provenance: `protocol.json` from `yeopeter1031/aiot-fall-qwen3.5-2b-lora-r1@acd4d0a750731a600386d27c3969deb789dc6f1e`; base tokenizer revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The native `/completion` request uses the non-thinking prompt, greedy decoding, seed 42, 160-token limit and no tools. Malformed JSON, duplicate keys, invalid action/reason/route combinations, silence followed by a terminal decision, truncation and unavailable resources are unresolved failures, not alerts or all-clears. Handle failures with a bounded retry/check-in or human review; never close an event merely because an adapter failed. Schema checks do not establish the truth of a well-formed decision.

`decide()` does not send notifications. Only call `notify(decision)` once per independently confirmed event. It delegates to the existing Discord notifier: this separate, optional operation requires network access and a configured webhook. Offline operation cannot deliver a Discord message. Inspect its return status; do not blindly retry an ambiguous send. Do not enable transcript, raw PCM or prompt logging in the caller.

## Verification scope

Physical microphone/speaker playback, wired sensing and live GGUF inference together on the target Jetson have not been verified. The adapter does not automatically download models.
