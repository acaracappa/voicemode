"""Continuous echo-cancelled mic capture via macOS VoiceProcessingIO.

Reuses the ctypes scaffolding from voice_mode.utils.macos_mic (borrowed from
the upstream vpio-test branch), but keeps the audio unit running indefinitely
and streams frames into a queue instead of recording one utterance.

VPIO gives us what plain CoreAudio capture can't: the system subtracts its own
playback (music, our TTS acks) from the mic signal, and macOS mic modes
(Voice Isolation etc.) become selectable in Control Center while running.
"""

import queue
import threading
import time
from ctypes import Structure, byref, c_float, c_uint8, c_uint32, c_void_p, cast, sizeof

import numpy as np

from voice_mode.utils import macos_mic as m

# kAUVoiceIOProperty_OtherAudioDuckingConfiguration (macOS 14+): controls how
# hard the system ducks other audio while a voice-processing session is live.
_kAUVoiceIOProperty_OtherAudioDuckingConfiguration = 2108
DUCK_LEVELS = {"default": 0, "min": 10, "mid": 20, "max": 30}


class _DuckingConfig(Structure):
    _fields_ = [
        ("mEnableAdvancedDucking", c_uint8),
        ("mDuckingLevel", c_uint32),
    ]


class VPIOStream:
    """Persistent VoiceProcessingIO input stream. Frames arrive on .frames."""

    def __init__(self, duck: str = "min"):
        if not m.is_available():
            raise RuntimeError("VoiceProcessingIO not available on this system")
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self.rate = 48000
        self.duck = duck
        self._running = False
        self._unit = None

    def start(self) -> None:
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
                self.frames.put(np.frombuffer(buf, dtype=np.float32).copy())
            return 0

        self._cb = m._AURenderCallback(cb)  # keep a ref or it gets collected
        cbs = m._AURenderCallbackStruct(inputProc=self._cb, inputProcRefCon=None)
        if m._at.AudioUnitSetProperty(
            self._unit, m._kAudioOutputUnitProperty_SetInputCallback,
            m._kAudioUnitScope_Global, 0, byref(cbs), sizeof(m._AURenderCallbackStruct),
        ) != 0:
            raise RuntimeError("failed to set input callback")
        duck_cfg = _DuckingConfig(
            mEnableAdvancedDucking=1,
            mDuckingLevel=DUCK_LEVELS.get(self.duck, 10),
        )
        status = m._at.AudioUnitSetProperty(
            self._unit, _kAUVoiceIOProperty_OtherAudioDuckingConfiguration,
            m._kAudioUnitScope_Global, 0, byref(duck_cfg), sizeof(_DuckingConfig),
        )
        if status != 0:
            print(f"note: ducking config not supported (err {status}); "
                  "system will duck other audio at its default level", flush=True)

        if m._at.AudioUnitInitialize(self._unit) != 0:
            raise RuntimeError("AudioUnitInitialize failed")
        self._running = True
        if m._at.AudioOutputUnitStart(self._unit) != 0:
            self._running = False
            raise RuntimeError("AudioOutputUnitStart failed")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        # CFRunLoopRunInMode returns immediately when the loop has no sources,
        # so pace the loop ourselves or this thread spins a full core
        while self._running:
            m._cf.CFRunLoopRunInMode(m._kCFRunLoopDefaultMode, 0.1, False)
            time.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        if self._unit:
            m._at.AudioOutputUnitStop(self._unit)
            m._at.AudioComponentInstanceDispose(self._unit)
            self._unit = None
