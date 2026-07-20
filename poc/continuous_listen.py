"""PoC: continuous-listening alternative to voicemode's open/record/close cycle.

One persistent mic stream for the whole session. Frames flow through a VAD
state machine that clips speech segments in real time; each segment is
transcribed by the local whisper server, then passed through an attention
gate before anything would reach an agent.

Capture backends:
  default  - sounddevice/PortAudio on the system default input.
  --vpio   - macOS VoiceProcessingIO (echo-cancelled): the OS subtracts its
             own playback (music, our TTS acks) from the mic signal, so
             speaker audio no longer floods the listener. Also enables the
             Control Center mic modes (e.g. Voice Isolation) while running,
             and configurable media ducking (--duck).

Attention model:
  ENGAGED  - after a directed utterance, everything for the next
             ENGAGED_WINDOW seconds is treated as directed (dialogue);
             any speech keeps the window alive.
  AMBIENT  - segments must start with the wake name (--wake, default
             "michael") within the first GATE_WORDS words; everything else
             is discarded.
  PAUSED   - "<name> pause" discards everything until "<name> resume" /
             "<name> i'm back" (or Ctrl-C).
  --no-gate disables AMBIENT entirely: every utterance is treated as
             directed. Pause/resume still requires the wake name.

Run from the fork root (deps come from the project env):
    uv run python poc/continuous_listen.py [--vpio] [--speak] [--wake NAME] [--no-gate]

--speak plays a short Kokoro ack for directed segments. Without --vpio, wear
headphones for that mode: on speakers the ack would feed back into the mic.
"""

import argparse
import collections
import io
import json
import queue
import re
import time
import urllib.request
import wave

import numpy as np
import sounddevice as sd
import webrtcvad

FRAME_MS = 30
PRE_ROLL_FRAMES = 33          # ~1s kept from before speech starts
TRIGGER_FRAMES = 5            # speech frames (of last 10) to open a segment
HANGOVER_MS = 2500            # trailing silence that closes a segment
EARLY_CHECK_MS = 800          # silence before checking for a spoken end-marker
MAX_SEGMENT_S = 60
ENGAGED_WINDOW_S = 120
GATE_WORDS = 6
CALIBRATION_FRAMES = 67       # ~2s of ambient noise sampled at startup

WHISPER_URL = "http://127.0.0.1:2022/v1/audio/transcriptions"
KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"
SPEAK_VOICE = "am_michael"
KOKORO_PCM_RATE = 24000

# Saying one of these at the end of a thought finalizes the clip immediately
# instead of waiting out the full silence hangover.
END_MARK_RE = re.compile(r"[\s,]*\b(period|send it|over)\b[\s.!?]*$", re.I)

# Saying one of these on its own drops straight from ENGAGED to AMBIENT
# instead of waiting out the engaged window.
DISENGAGE_RE = re.compile(
    r"^(that'?s all|that is all|we'?re done|that'?ll do|end session|thanks,? that'?s it)[\s.!?]*$",
    re.I,
)


def transcribe(pcm: np.ndarray, rate: int, prompt: str = "") -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    body = buf.getvalue()
    boundary = "poc-boundary"
    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="seg.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + body + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        WHISPER_URL,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = json.load(resp).get("text", "").strip()
    # whisper emits bracketed markers ("[BLANK_AUDIO]", "(upbeat music)") or
    # bare punctuation on non-speech
    if re.fullmatch(r"[\[\(].*[\]\)]", text) or not re.search(r"\w", text):
        return ""
    return text


def speak(text: str) -> None:
    """Speak via Kokoro, streaming PCM so playback starts before synthesis ends."""
    req = urllib.request.Request(
        KOKORO_URL,
        data=json.dumps(
            {"model": "tts-1", "voice": SPEAK_VOICE, "input": text,
             "response_format": "pcm", "stream": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            with sd.RawOutputStream(samplerate=KOKORO_PCM_RATE, channels=1,
                                    dtype="int16") as out:
                while True:
                    chunk = resp.read(4800)  # ~100ms of 24kHz int16
                    if not chunk:
                        break
                    if len(chunk) % 2:
                        chunk += b"\x00"
                    out.write(chunk)
        return
    except OSError as e:
        print(f"tts stream failed ({e}); falling back to wav", flush=True)
    req = urllib.request.Request(
        KOKORO_URL,
        data=json.dumps(
            {"model": "tts-1", "voice": SPEAK_VOICE, "input": text,
             "response_format": "wav"}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with wave.open(io.BytesIO(data)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sd.play(audio, w.getframerate())
        sd.wait()


def tone(freqs: tuple[float, ...]) -> None:
    """Short cue tones (non-blocking) for mode changes."""
    t = np.linspace(0, 0.12, int(16000 * 0.12), False)
    notes = [np.sin(2 * np.pi * f * t) * np.hanning(len(t)) * 0.3 for f in freqs]
    sd.play(np.concatenate(notes).astype(np.float32), 16000)


def frame_source(args) -> tuple[queue.Queue, int, object]:
    """Return (queue of 30ms int16 frames, sample rate, closer)."""
    frames: queue.Queue = queue.Queue()
    if args.vpio:
        from vpio_stream import VPIOStream
        stream = VPIOStream(duck=args.duck)
        stream.start()
        rate = stream.rate
        frame_samples = rate * FRAME_MS // 1000
        residue = np.array([], dtype=np.int16)

        def pump():
            nonlocal residue
            while True:
                f32 = stream.frames.get()
                pcm = np.clip(f32 * 32767, -32768, 32767).astype(np.int16)
                residue = np.concatenate([residue, pcm])
                while len(residue) >= frame_samples:
                    frames.put(residue[:frame_samples])
                    residue = residue[frame_samples:]

        import threading
        threading.Thread(target=pump, daemon=True).start()
        return frames, rate, stream.stop

    rate = 16000
    frame_samples = rate * FRAME_MS // 1000
    stream = sd.InputStream(
        samplerate=rate, channels=1, dtype="int16", blocksize=frame_samples,
        callback=lambda indata, *_: frames.put(indata[:, 0].copy()),
    )
    stream.start()
    return frames, rate, stream.stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpio", action="store_true",
                    help="echo-cancelled capture via macOS VoiceProcessingIO")
    ap.add_argument("--speak", action="store_true",
                    help="Kokoro ack for directed segments")
    ap.add_argument("--duck", choices=("min", "mid", "max", "default"), default="min",
                    help="how hard macOS ducks other audio during VPIO capture")
    ap.add_argument("--wake", default="michael",
                    help="wake name that opens the attention gate")
    ap.add_argument("--no-gate", action="store_true",
                    help="treat all speech as directed (no wake name needed)")
    args = ap.parse_args()

    wake = args.wake.lower().strip()
    pause_re = re.compile(rf"\b{re.escape(wake)}\b[,.!?\s]*(pause|hold on|stop listening)", re.I)
    resume_re = re.compile(rf"\b{re.escape(wake)}\b[,.!?\s]*(resume|i'?m back|unpause)", re.I)

    def is_directed(text: str) -> bool:
        words = re.sub(r"[^\w\s']", " ", text.lower()).split()
        return wake in words[:GATE_WORDS]

    stt_prompt = (
        "Claude Code voice session. Vocabulary: voicemode, whisper, Kokoro, "
        f"MCP, uv, git, repo, pull request, branch, {wake}."
    )

    frames, rate, closer = frame_source(args)
    use_webrtc = rate in (8000, 16000, 32000, 48000)
    vad = webrtcvad.Vad(2) if use_webrtc else None

    class AdaptiveEnergy:
        """Energy VAD with a startup-calibrated, slowly adapting noise floor."""

        def __init__(self):
            self.floor = None
            self.calib: list[float] = []

        def threshold(self) -> float:
            return min(max(self.floor * 3.5, 200.0), 1500.0)

        def __call__(self, frame: np.ndarray) -> bool:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if self.floor is None:
                self.calib.append(rms)
                if len(self.calib) >= CALIBRATION_FRAMES:
                    self.floor = float(np.median(self.calib))
                    print(f"calibrated: noise floor {self.floor:.0f}, "
                          f"speech threshold {self.threshold():.0f}", flush=True)
                return False
            if rms < self.threshold():
                self.floor = 0.98 * self.floor + 0.02 * rms
            return rms >= self.threshold()

    energy_vad = AdaptiveEnergy()

    def frame_is_speech(frame: np.ndarray) -> bool:
        if vad is not None:
            return vad.is_speech(frame.tobytes(), rate)
        return energy_vad(frame)

    pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)
    recent = collections.deque(maxlen=10)
    segment: list[np.ndarray] = []
    silence_ms = 0
    early_checked = False
    mode = "OPEN" if args.no_gate else "AMBIENT"
    engaged_until = 0.0
    paused = False

    def log(tag: str, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {tag:9s} {msg}", flush=True)

    def finish_segment(text: str | None = None) -> None:
        nonlocal mode, engaged_until, paused
        pcm = np.concatenate(segment)
        segment.clear()
        dur = len(pcm) / rate
        t0 = time.time()
        if text is None:
            try:
                text = transcribe(pcm, rate, stt_prompt)
            except OSError as e:
                log("STT-ERR", f"whisper unreachable ({e}); segment dropped, retrying next clip")
                return
        text = END_MARK_RE.sub("", text).strip()
        if not text:
            log("NOISE", f"{dur:.1f}s segment, no speech (stt {time.time()-t0:.1f}s)")
            return
        if paused:
            if resume_re.search(text):
                paused = False
                log("RESUME", text)
                if args.speak:
                    speak("Back and listening.")
            else:
                log("PAUSED", f"discarded: {text}")
            return
        if pause_re.search(text):
            paused = True
            log("PAUSE", text)
            if args.speak:
                speak(f"Paused. Say {wake}, I'm back, when you're ready.")
            return
        if args.no_gate:
            log("DIRECTED", f"({dur:.1f}s, stt {time.time()-t0:.1f}s) -> {text}")
            if args.speak:
                speak(f"Heard: {' '.join(text.split()[:8])}")
            return
        engaged = time.time() < engaged_until
        if engaged and DISENGAGE_RE.match(text.strip()):
            mode = "AMBIENT"
            engaged_until = 0.0
            tone((784.0, 523.0))  # falling: name needed again
            log("DISENGAGE", f"{text} — back to ambient")
            return
        if engaged or is_directed(text):
            if mode != "ENGAGED":
                tone((523.0, 784.0))  # rising: you have my attention
            mode = "ENGAGED"
            engaged_until = time.time() + ENGAGED_WINDOW_S
            log("DIRECTED", f"({dur:.1f}s, stt {time.time()-t0:.1f}s) -> {text}")
            if args.speak:
                speak(f"Heard: {' '.join(text.split()[:8])}")
        else:
            log("AMBIENT", f"discarded: {text}")

    gate_desc = "no gate (all speech directed)" if args.no_gate else f"wake name '{wake}'"
    print(f"Continuous listening PoC — {gate_desc}, "
          f"capture {'VPIO ' + str(rate) + 'Hz echo-cancelled, duck ' + args.duck if args.vpio else 'PortAudio 16kHz'}, "
          f"vad {'webrtc' if use_webrtc else 'energy'}. Ctrl-C to quit.", flush=True)
    try:
        while True:
            try:
                frame = frames.get(timeout=0.25)
            except queue.Empty:
                pass
            else:
                is_speech = frame_is_speech(frame)
                recent.append(is_speech)
                if segment:
                    segment.append(frame)
                    if is_speech:
                        silence_ms = 0
                        early_checked = False
                    else:
                        silence_ms += FRAME_MS
                    if silence_ms >= HANGOVER_MS or len(segment) * FRAME_MS / 1000 >= MAX_SEGMENT_S:
                        finish_segment()
                        silence_ms = 0
                        early_checked = False
                    elif silence_ms >= EARLY_CHECK_MS and not early_checked:
                        # spoken end-marker check: transcribe a snapshot early;
                        # if the thought ends with "period"/"send it"/"over",
                        # finalize now instead of waiting out the hangover
                        early_checked = True
                        try:
                            snapshot = transcribe(np.concatenate(segment), rate, stt_prompt)
                        except OSError:
                            snapshot = ""
                        if snapshot and END_MARK_RE.search(snapshot):
                            finish_segment(text=snapshot)
                            silence_ms = 0
                            early_checked = False
                else:
                    pre_roll.append(frame)
                    if sum(recent) >= TRIGGER_FRAMES:
                        segment.extend(pre_roll)
                        pre_roll.clear()
                        recent.clear()
                        silence_ms = 0
            if mode == "ENGAGED" and time.time() >= engaged_until:
                mode = "AMBIENT"
                tone((784.0, 523.0))  # falling: name needed again
                log("AMBIENT", "engaged window expired — say the name to re-engage")
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        closer()


if __name__ == "__main__":
    main()
