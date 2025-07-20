from typing import Union
from retico_core import AbstractModule, AbstractTriggerModule, UpdateMessage, UpdateType
from retico_core.audio import AudioIU
from retico_core.text import TextIU

from collections import deque

from lingua import LanguageDetectorBuilder
from speechbrain.inference import EncoderClassifier

import numpy as np
import threading
import time
import torch
import os

# Initialize the audio language classifier
_AUDIO_LANGUAGE_CLASSIFIER = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
)

class LanguageDetectionIU(AudioIU, TextIU):
    @staticmethod
    def type():
        return "Text IU"

    def __init__(
        self,
        creator,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        payload=None,
        rate=None,
        nframes=None,
        sample_width=None,
        raw_audio=None,
        **kwargs,
    ):
        if isinstance(self, TextIU):
            super().__init__(
                creator=creator,
                iuid=iuid,
                previous_iu=previous_iu,
                grounded_in=grounded_in,
                payload=payload,
                **kwargs,
            )
        else:
            super().__init__(
                creator=creator,
                iuid=iuid,
                previous_iu=previous_iu,
                grounded_in=grounded_in,
                rate=rate,
                nframes=nframes,
                sample_width=sample_width,
                raw_audio=raw_audio,
                **kwargs,
            )
        self.predictions = None
        self.stability = None
        self.confidence = None
        self.payload = payload
        self.text = None
        self.final = False
        self.language = None


class LanguageDetectionModule(AbstractModule):
    def __init__(self):
        super().__init__()
        self._text_buffer = []
        self._audio_buffer = deque()
        self._last_classif_ts = 0.0
        self._min_call_interval = 1.0
        self._classification_lock = threading.Lock()
        self._last_languages_buffer = deque()
        self._context_buffer = deque(maxlen=10)
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )

    @staticmethod
    def name():
        return "Automatic Audio and Text Language Detection Module"

    @staticmethod
    def description():
        return "A module that detects the language of a raw audio or text."

    @staticmethod
    def input_ius():
        return [AudioIU, TextIU]

    @staticmethod
    def output_iu():
        return LanguageDetectionIU

    def process_update(self, update: UpdateMessage):
        for iu, _ in update:
            out_iu = None
            if isinstance(iu, AudioIU):
                tmp = self.process_audio_iu(iu)
                if tmp is not None:
                    out_iu = tmp
            elif isinstance(iu, TextIU):
                if iu.text.strip():
                    out_iu = self.process_text_iu(iu)
            if out_iu is None:
                continue
            um = UpdateMessage().from_iu(out_iu, UpdateType.ADD)
            self.append(um)

    def process_audio_iu(self, iu: AudioIU, purging_time: float = 4):
        arr = np.frombuffer(iu.raw_audio, dtype="<i2").astype(np.float32) / 32768
        now = time.time()
        self._audio_buffer.append((arr, now))
        cutoff = now - purging_time
        while self._audio_buffer and self._audio_buffer[0][1] < cutoff:
            self._audio_buffer.popleft()
        if len(self._last_languages_buffer) >= 100:
            self._last_languages_buffer.popleft()
        audio = np.concatenate([chunk for chunk, _ in self._audio_buffer])

        now = time.time()
        if (
            now - self._last_classif_ts >= self._min_call_interval
            and len(audio) >= iu.rate * 2.5
        ):  # Send at least 2.5 seconds of audio at once
            self._last_classif_ts = now
            threading.Thread(
                target=self.classify_audio, args=(audio.copy(), iu.rate), daemon=True
            ).start()
        out_iu = self.create_iu(iu)
        out_iu.raw_audio = iu.raw_audio

        with self._classification_lock:
            # Most common language in a sliding window of the last n predictions also spanning over silence, to account for occasional misclassifications
            langs = [d["lang"] for d in self._last_languages_buffer]
            if langs:
                out_iu.language = max(set(langs), key=langs.count)
                # Unweighted average confidence of the last n predictions for this language
                matches = [
                    d
                    for d in self._last_languages_buffer
                    if d["lang"] == out_iu.language
                ]
                out_iu.confidence = (
                    sum(d["confidence"] for d in matches) / len(matches)
                    if matches
                    else 0.0
                )
            else:
                out_iu.confidence = 0.0
        out_iu.payload = iu.payload
        if is_tail_silence_or_noise(audio, iu.rate):
            self._audio_buffer.clear()
            return
        return out_iu

    def process_text_iu(self, iu: TextIU) -> LanguageDetectionIU:
        text = iu.text.strip()
        self._context_buffer.append(text)
        context = " ".join(self._context_buffer)
        language = self._detector.detect_language_of(context)
        if language is None:
            print(f"[ERROR] {self.name()} could not detect language for text: {text}")
        confidence = self._detector.compute_language_confidence(context, language)
        out_iu = LanguageDetectionIU(self, iuid=iu.iuid, previous_iu=iu)
        out_iu.language = language.iso_code_639_1.name.lower()
        out_iu.confidence = confidence
        out_iu.payload = text
        out_iu.text = text
        return out_iu

    def classify_audio(self, audio: np.ndarray, rate: int):
        """Background audio classification."""
        with torch.no_grad():
            signal = torch.tensor(audio[None, :])
            prediction = _AUDIO_LANGUAGE_CLASSIFIER.classify_batch(signal)
        lang_code = prediction[3][0][:2]
        confidence = float(prediction[1].exp()[0])
        with self._classification_lock:
            self._last_languages_buffer.append(
                {"lang": lang_code, "confidence": confidence}
            )

def dump_buffer_to_wav(buffer: Union[np.ndarray, bytes], rate: int):
    # For debugging purposes only
    
    if isinstance(buffer, bytes):
        import io
        import wave
        buffer = np.frombuffer(buffer, dtype="<i2").astype(np.float32) / 32768
        with io.BytesIO() as bytes_buffer:
            with wave.open(bytes_buffer, "wb") as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(rate)
                wf.writeframes((buffer * 32767).astype(np.int16).tobytes())
            bytes_buffer.seek(0)
            wav = bytes_buffer.read()
        # Convert to numpy array for saving
        wav = np.frombuffer(wav, dtype="<i2").astype(np.int16)
    else:
        wav = (buffer * 32767).astype(np.int16)
        
    import soundfile as sf
    sf.write(
        file="audio/" + f"{time.time()}" + ".wav",
        data=wav,
        samplerate=rate,
        subtype="PCM_16",
    )

def is_tail_silence_or_noise(
    buffer: np.ndarray,
    rate: int,
    silent_tail_size: float = 2.5,
    silence_max_rootmeansquare_energy_threshold=0.01,
) -> bool:
    """
    Returns True if the last `tail_size` seconds of `buffer`
    are essentially silence (low RMS) or noise (high spectral flatness).

    :param buffer: the audio buffer to check
    :param rate: the sample rate of the audio buffer (in Hz)
    :param silent_tail_size: the size of the tail to check whether it's deemed silent or not (in seconds)
    :param silence_max_rootmeansquare_energy_threshold: the maximum RMS energy to consider the tail as silence (closer to 0: silence)
    """

    n_tail = int(silent_tail_size * rate)
    if buffer.size < n_tail:
        return False
    tail = buffer[-n_tail:]
    rms = np.sqrt(np.mean(tail**2))
    if rms < silence_max_rootmeansquare_energy_threshold:
        return True
    return False
