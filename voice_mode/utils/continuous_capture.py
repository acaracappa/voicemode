"""Persistent echo-cancelled microphone capture (continuous listening, v1).

Instead of opening and closing a recording stream per converse call, this
module keeps one VoiceProcessingIO stream alive for the whole MCP server
lifetime and segments speech in the background. converse's record path pops
completed segments.

What this buys over the stock open/record/close cycle:
- macOS media ducking happens once and holds steady (configurable level via
  VOICEMODE_VPIO_DUCK: min|mid|max|default, default min) instead of bouncing
  at every turn.
- System echo cancellation: the OS subtracts its own playback (music, TTS)
  from the mic signal, so speaker audio doesn't flood recordings.
- Speech spoken while the agent is busy between converse calls is buffered
  (up to BUFFER_TTL_S) and delivered on the next listen instead of lost.

Enabled via VOICEMODE_CONTINUOUS_CAPTURE=true. macOS only; converse falls
back to the standard recorder on any failure.
"""

import logging
import os
import threading
import time
from collections import deque
from ctypes import Structure, byref, c_float, c_uint8, c_uint32, c_void_p, cast, sizeof

import numpy as np

from voice_mode.utils import macos_mic as m

logger = logging.getLogger("voicemode")

_BEACON = os.path.expanduser("~/.voicemode/logs/continuous_capture.log")


def _beacon(msg: str) -> None:
    """Append a diagnostic line; server stderr is invisible under MCP."""
    try:
        with open(_BEACON, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid {os.getpid()}] {msg}\n")
    except OSError:
        pass

FRAME_MS = 30
PRE_ROLL_FRAMES = 33          # ~1s kept from before speech starts
TRIGGER_FRAMES = 5            # speech frames (of last 10) to open a segment
MAX_SEGMENT_S = 120
CALIBRATION_FRAMES = 67       # ~2s ambient noise sampled at startup
BUFFER_TTL_S = 90             # discard buffered segments older than this
TARGET_RATE = 24000           # what converse/whisper expect
GATE_WORDS = 3                # wake name must lead the segment (or a sentence) to pass

# kAUVoiceIOProperty_OtherAudioDuckingConfiguration (macOS 14+)
_kAUVoiceIOProperty_OtherAudioDuckingConfiguration = 2108
_DUCK_LEVELS = {"default": 0, "min": 10, "mid": 20, "max": 30}


def enabled() -> bool:
    return os.getenv("VOICEMODE_CONTINUOUS_CAPTURE", "").strip().lower() in (
        "1", "true", "yes", "on"
    ) and m.is_available()


def _wake_name() -> str:
    return os.getenv("VOICEMODE_WAKE_NAME", "michael").strip().lower()


def _wake_aliases() -> list:
    default = f"{_wake_name()},mike,mikel,michel"
    raw = os.getenv("VOICEMODE_WAKE_ALIASES", default)
    return [a.strip().lower() for a in raw.split(",") if a.strip()]


def _gate_transcribe(pcm: np.ndarray, rate: int) -> str:
    """Local whisper pass used only for buffer admission (never leaves the box)."""
    import io
    import json
    import re
    import urllib.request
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(pcm * 32767, -32768, 32767).astype(np.int16).tobytes())
    body = buf.getvalue()
    boundary = "gate-boundary"
    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="seg.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + body + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    # deliberately NO prompt here: whisper echoes prompt text back on noisy
    # segments, and a prompt containing the wake name would let noise pass
    # the very gate it feeds
    url = os.getenv("VOICEMODE_GATE_STT_URL",
                    "http://127.0.0.1:2022/v1/audio/transcriptions")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = json.load(resp).get("text", "").strip()
    if re.fullmatch(r"[\[\(].*[\]\)]", text) or not re.search(r"\w", text):
        return ""
    return text


def _passes_gate(text: str) -> bool:
    """Directed if an alias leads the segment, or starts a sentence within it.

    Segments can fuse ungated chatter with a summons when the speaker doesn't
    pause between them ("...hear me now. Michael, did you..."), so a
    sentence-initial alias anywhere admits the segment. Mid-sentence mentions
    ("tell michael dinner's ready") still fail.
    """
    import re
    lowered = text.lower()
    words = re.sub(r"[^\w\s']", " ", lowered).split()
    aliases = _wake_aliases()
    if any(a in words[:GATE_WORDS] for a in aliases):
        return True
    alias_re = "|".join(re.escape(a) for a in aliases)
    return re.search(rf"[.!?][\"')\]]*\s+({alias_re})\b", lowered) is not None


class _DuckingConfig(Structure):
    _fields_ = [
        ("mEnableAdvancedDucking", c_uint8),
        ("mDuckingLevel", c_uint32),
    ]


class _AdaptiveEnergyVAD:
    """Energy VAD with a startup-calibrated, slowly adapting noise floor."""

    def __init__(self):
        self.floor = None
        self._calib = []

    def _threshold(self) -> float:
        return min(max(self.floor * 3.5, 0.006), 0.045)  # float32 scale

    def __call__(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if self.floor is None:
            self._calib.append(rms)
            if len(self._calib) >= CALIBRATION_FRAMES:
                self.floor = float(np.median(self._calib))
                logger.info(
                    f"continuous capture calibrated: noise floor {self.floor:.4f}, "
                    f"threshold {self._threshold():.4f}"
                )
                _beacon(f"calibrated: floor {self.floor:.4f}, threshold {self._threshold():.4f}")
            return False
        if rms < self._threshold():
            self.floor = 0.98 * self.floor + 0.02 * rms
        return rms >= self._threshold()


class ContinuousCapture:
    """Singleton persistent VPIO stream + background speech segmenter."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "ContinuousCapture":
        with cls._lock:
            if cls._instance is None:
                inst = cls()
                inst._start()
                cls._instance = inst
            return cls._instance

    def __init__(self):
        self.rate = 48000
        self._unit = None
        self._running = False
        self._raw: deque = deque()
        self._raw_cond = threading.Condition()
        self._segments: deque = deque()          # (timestamp, np.float32 pcm)
        self._segments_cond = threading.Condition()
        self._waiters = 0                        # live listens currently blocked

    # -- VPIO plumbing -------------------------------------------------------

    def _start(self) -> None:
        desc = m._AudioComponentDescription(
            componentType=m._kAudioUnitType_Output,
            componentSubType=m._kAudioUnitSubType_VoiceProcessingIO,
            componentManufacturer=m._kAudioUnitManufacturer_Apple,
            componentFlags=0,
            componentFlagsMask=0,
        )
        component = m._at.AudioComponentFindNext(None, byref(desc))
        if not component:
            raise RuntimeError("VoiceProcessingIO component not found")
        instance = c_void_p()
        if m._at.AudioComponentInstanceNew(component, byref(instance)) != 0:
            raise RuntimeError("AudioComponentInstanceNew failed")
        self._unit = instance.value

        fmt = m._AudioStreamBasicDescription()
        size = c_uint32(sizeof(m._AudioStreamBasicDescription))
        if m._at.AudioUnitGetProperty(
            self._unit, m._kAudioUnitProperty_StreamFormat,
            m._kAudioUnitScope_Output, 1, byref(fmt), byref(size),
        ) == 0 and fmt.mSampleRate:
            self.rate = int(fmt.mSampleRate)

        duck_name = os.getenv("VOICEMODE_VPIO_DUCK", "min").strip().lower()
        duck_cfg = _DuckingConfig(
            mEnableAdvancedDucking=1,
            mDuckingLevel=_DUCK_LEVELS.get(duck_name, 10),
        )
        status = m._at.AudioUnitSetProperty(
            self._unit, _kAUVoiceIOProperty_OtherAudioDuckingConfiguration,
            m._kAudioUnitScope_Global, 0, byref(duck_cfg), sizeof(_DuckingConfig),
        )
        if status != 0:
            logger.info(f"VPIO ducking config unsupported (err {status}); using system default")

        unit = self._unit

        def cb(ref_con, action_flags, timestamp, bus_number, num_frames, io_data):
            if not self._running:
                return 0
            buf = (c_float * num_frames)()
            bl = m._AudioBufferList()
            bl.mNumberBuffers = 1
            bl.mBuffers[0].mNumberChannels = 1
            bl.mBuffers[0].mDataByteSize = num_frames * 4
            bl.mBuffers[0].mData = cast(buf, c_void_p)
            if m._at.AudioUnitRender(
                unit, action_flags, timestamp, bus_number, num_frames, byref(bl)
            ) == 0:
                with self._raw_cond:
                    self._raw.append(np.frombuffer(buf, dtype=np.float32).copy())
                    self._raw_cond.notify()
            return 0

        self._cb = m._AURenderCallback(cb)  # keep a ref or it gets collected
        cbs = m._AURenderCallbackStruct(inputProc=self._cb, inputProcRefCon=None)
        if m._at.AudioUnitSetProperty(
            self._unit, m._kAudioOutputUnitProperty_SetInputCallback,
            m._kAudioUnitScope_Global, 0, byref(cbs), sizeof(m._AURenderCallbackStruct),
        ) != 0:
            raise RuntimeError("failed to set input callback")
        if m._at.AudioUnitInitialize(self._unit) != 0:
            raise RuntimeError("AudioUnitInitialize failed")
        self._running = True
        if m._at.AudioOutputUnitStart(self._unit) != 0:
            self._running = False
            raise RuntimeError("AudioOutputUnitStart failed")
        threading.Thread(target=self._pump_runloop, daemon=True).start()
        threading.Thread(target=self._segment_loop, daemon=True).start()
        threading.Thread(target=self._orphan_watchdog, daemon=True).start()
        logger.info(
            f"continuous capture started: VPIO {self.rate}Hz echo-cancelled, "
            f"duck={duck_name}"
        )
        _beacon(f"stream started: VPIO {self.rate}Hz, duck={duck_name}")

    @staticmethod
    def _client_alive() -> bool:
        """Walk process ancestry looking for the Claude Code client.

        The uv wrapper survives client death, so a simple ppid==1 check never
        fires for the python server; walk up until we find claude/node
        (alive) or launchd (orphaned).
        """
        import subprocess
        pid = os.getppid()
        for _ in range(8):
            if pid <= 1:
                return False
            try:
                out = subprocess.run(
                    ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return True  # can't tell; err on the side of staying alive
            if not out:
                return False  # parent vanished mid-walk
            ppid_str, _, comm = out.partition(" ")
            if "claude" in comm.lower() or "node" in comm.lower():
                return True
            pid = int(ppid_str)
        return True

    def _orphan_watchdog(self) -> None:
        """Release the mic and exit if our MCP client dies.

        Disconnected voicemode servers linger indefinitely (observed
        repeatedly: every /mcp reconnect strands the old process, which keeps
        its VPIO stream — and the microphone — alive forever). Detect a dead
        client, stop the stream, and exit the process.
        """
        while self._running:
            time.sleep(5)
            if not self._client_alive():
                _beacon("client died; stopping stream and exiting orphan server")
                try:
                    self._running = False
                    if self._unit:
                        m._at.AudioOutputUnitStop(self._unit)
                        m._at.AudioComponentInstanceDispose(self._unit)
                        self._unit = None
                finally:
                    os._exit(0)

    def _pump_runloop(self) -> None:
        # CFRunLoopRunInMode returns immediately when the loop has no sources,
        # so pace the loop ourselves or this thread spins a full core
        while self._running:
            m._cf.CFRunLoopRunInMode(m._kCFRunLoopDefaultMode, 0.1, False)
            time.sleep(0.1)

    # -- segmentation --------------------------------------------------------

    def _segment_loop(self) -> None:
        silence_threshold_ms = int(os.getenv("VOICEMODE_SILENCE_THRESHOLD_MS", "2500"))
        frame_samples = self.rate * FRAME_MS // 1000
        vad = _AdaptiveEnergyVAD()
        residue = np.array([], dtype=np.float32)
        pre_roll: deque = deque(maxlen=PRE_ROLL_FRAMES)
        recent: deque = deque(maxlen=10)
        segment: list = []
        silence_ms = 0

        while self._running:
            with self._raw_cond:
                while not self._raw and self._running:
                    self._raw_cond.wait(timeout=0.5)
                chunks = list(self._raw)
                self._raw.clear()
            if not chunks:
                continue
            residue = np.concatenate([residue] + chunks)
            while len(residue) >= frame_samples:
                frame, residue = residue[:frame_samples], residue[frame_samples:]
                is_speech = vad(frame)
                recent.append(is_speech)
                if segment:
                    segment.append(frame)
                    silence_ms = 0 if is_speech else silence_ms + FRAME_MS
                    if (silence_ms >= silence_threshold_ms
                            or len(segment) * FRAME_MS / 1000 >= MAX_SEGMENT_S):
                        pcm = np.concatenate(segment)
                        segment.clear()
                        silence_ms = 0
                        with self._segments_cond:
                            live_waiter = self._waiters > 0
                        # Live listens (the user answering a question) are
                        # ungated. Buffer admission — speech captured with no
                        # listen open — requires the wake name, so household
                        # conversation never becomes agent input.
                        admit = True
                        if not live_waiter:
                            try:
                                text = _gate_transcribe(pcm, self.rate)
                            except OSError as e:
                                admit = False
                                _beacon(f"gate STT failed ({e}); segment discarded (fail-closed)")
                            else:
                                if not text:
                                    admit = False
                                    _beacon(f"gate: non-speech discarded ({len(pcm)/self.rate:.1f}s)")
                                elif not _passes_gate(text):
                                    admit = False
                                    _beacon(f"gate: ambient discarded: {text[:80]}")
                                else:
                                    _beacon(f"gate: directed admitted: {text[:80]}")
                        if admit:
                            with self._segments_cond:
                                self._segments.append((time.time(), pcm))
                                self._segments_cond.notify_all()
                            logger.info(
                                f"continuous capture: segment ready "
                                f"({len(pcm)/self.rate:.1f}s, {len(self._segments)} buffered)"
                            )
                            _beacon(f"segment ready: {len(pcm)/self.rate:.1f}s ({len(self._segments)} buffered)")
                else:
                    pre_roll.append(frame)
                    if sum(recent) >= TRIGGER_FRAMES:
                        segment.extend(pre_roll)
                        pre_roll.clear()
                        recent.clear()
                        silence_ms = 0

    # -- public API ----------------------------------------------------------

    def capture_segment(self, max_wait: float) -> tuple:
        """Pop the next speech segment, waiting up to max_wait seconds.

        Returns (int16 audio at TARGET_RATE, speech_detected).
        """
        deadline = time.time() + max_wait
        with self._segments_cond:
            self._waiters += 1
            try:
                while True:
                    while self._segments:
                        ts, pcm = self._segments.popleft()
                        if time.time() - ts <= BUFFER_TTL_S:
                            return self._to_target(pcm), True
                        logger.info("continuous capture: dropped stale buffered segment")
                        _beacon("dropped stale buffered segment")
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        # converse treats an empty buffer as a recording failure;
                        # return brief silence so it takes the no-speech path
                        return np.zeros(TARGET_RATE // 2, dtype=np.int16), False
                    self._segments_cond.wait(timeout=min(remaining, 1.0))
            finally:
                self._waiters -= 1

    def _to_target(self, pcm: np.ndarray) -> np.ndarray:
        if self.rate != TARGET_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(TARGET_RATE, self.rate)
            pcm = resample_poly(pcm, TARGET_RATE // g, self.rate // g)
        return np.clip(pcm * 32767, -32768, 32767).astype(np.int16)


def capture_segment(max_duration: float) -> tuple:
    """Module-level entry used by converse. Raises on VPIO init failure."""
    _beacon(f"capture_segment(max={max_duration:.0f}s) called")
    audio, speech = ContinuousCapture.get().capture_segment(max_duration)
    _beacon(f"capture_segment -> {len(audio)} samples, speech={speech}")
    return audio, speech
