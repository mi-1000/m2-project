import json
import os
import threading
from collections import deque
from typing import Literal

from retico_core import network, AbstractModule, UpdateMessage, UpdateType
from retico_core.text import TextIU
from retico_core.audio import AudioIU, MicrophoneModule, SpeakerModule
from retico_googleasr import GoogleASRModule

from gemini import Gemini
from LanguageDetectionModule.language_detection import LanguageDetectionModule
from LanguageDetectionModule.multilingual_tts import MultilingualTTSModule

# Silence detection imports
import numpy as np
from librosa import stft, feature

SYSTEM_PROMPT = """
You are a friendly and knowledgeable language tutor. Help the user practice {language} by having a short, natural conversation about “{topic}” at level {difficulty_level}.

Guidelines:
1. Speak mostly in {language}, using simple words and grammar for level {difficulty_level}.
2. Ask short, open questions about {topic}. Encourage the user to answer in their own words.
3. If the user makes a mistake, and it seems relevant to correct that mistake at {difficulty_level} level, gently correct it:
   - Give the correct sentence.
   - Briefly explain the rule (in {explanation_language}). This should be a standalone sentence.
   - Optionally, give a quick tip or example.
   - Do not correct the user too often, only when it seems relevant to the conversation and at the appropriate level, at a reasonable frequency.
   - Adapt the complexity of your vocabulary and grammar to the user's level.
4. Each time you respond, you must end your utterance in either a period (.), question mark (?), or exclamation mark (!), and nothing else.
5. Only write in plain text, do not use any special formatting or Markdown.
6. Do not use emojis or escaped characters such as "\\n", or anything that could not be processed by a text-to-speech system.

If asked, give cultural info, vocabulary, or grammar tips. Always be encouraging and keep answers short and clear. Answers should not be more than 2-3 sentences long.

Start by greeting the user in {language} and asking a simple question about “{topic}” at level {difficulty_level}.
"""

# Load topics
with open(os.path.join(os.path.dirname(__file__), "topics.json"), encoding="utf-8") as f:
    TOPICS = json.load(f)

# Persistence
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "user_progress.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        return json.load(open(PROGRESS_FILE, encoding="utf-8"))
    return {}

def save_progress(progress):
    json.dump(progress, open(PROGRESS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

class AudioParameterGeminiModule(AbstractModule):
    @staticmethod
    def name(): return "Audio-Enabled Gemini Module"

    @staticmethod
    def description(): return "Handles text & audio, buffers mic audio until silence, sends to Gemini."

    @staticmethod
    def input_ius(): return [TextIU, AudioIU]

    @staticmethod
    def output_ius(): return [TextIU, AudioIU]

    def __init__(self, topics, **kwargs):
        super().__init__(**kwargs)
        self.topics = topics
        self.language = self.level = self.topic = self.explanation_language = None
        self.idx = None
        self._next_prompt = 'language'
        self._gemini = None
        self.progress = load_progress()
        # Audio buffer for silence detection
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.silence_threshold = 0.01
        self.noise_flatness_threshold = 0.8
        self.silent_tail_sec = 1.0

    def _ask_text(self, msg):
        um = UpdateMessage()
        um.add_iu(TextIU(text=msg), UpdateType.ADD)
        self.append(um)

    def _init_gemini(self):
        prompt = SYSTEM_PROMPT.format(
            language=self.language, topic=self.topic,
            difficulty_level=self.level, explanation_language=self.explanation_language
        )
        self._gemini = Gemini(system_instructions=prompt, model="gemini-2.5-audioflash", streaming_audio=True)

    def process_update(self, update):
        for iu, _ in update:
            if isinstance(iu, TextIU):
                for chunk in self.process_text_iu(iu.text.strip()):
                    self.append(TextIU(text=chunk))
            elif isinstance(iu, AudioIU):
                out = self.process_audio_iu(iu)
                if out is None: continue
                for chunk in out:
                    self.append(TextIU(text=chunk))
                    # print("chunk")

    def process_text_iu(self, text):
        # Parameter collection & command handling same as before
        if self._next_prompt:
            return self._handle_param_text(text)
        if text.lower() == 'next':
            return self._handle_next()
        if text.lower() == 'exit':
            return self._handle_exit()
        # send to Gemini
        return self._gemini.add_turn(text)

    def process_audio_iu(self, iu: AudioIU):
        # buffer raw audio
        with self.buffer_lock:
            arr = np.frombuffer(iu.raw_audio, dtype=np.int16).astype(np.float32)/32768
            self.audio_buffer.append(arr)
            # maintain tail length
            combined = np.concatenate(list(self.audio_buffer))
            # if not enough or not silence end, continue buffering
            if len(combined) < iu.rate * 1.0 or not self._is_silence(combined, iu.rate):
                return None
            # on silence end, clear buffer and send
            self.audio_buffer.clear()
        return self._gemini.add_audio_turn(combined.tobytes())

    def _is_silence(self, buf, rate):
        tail = buf[-int(self.silent_tail_sec*rate):]
        rms = np.sqrt(np.mean(tail**2))
        if rms < self.silence_threshold:
            return True
        S = np.abs(stft(tail, n_fft=512, hop_length=256))
        flat = np.mean(feature.spectral_flatness(S=S))
        return flat > self.noise_flatness_threshold

    # Parameter & command handlers
    def _handle_param_text(self, text):
        step = self._next_prompt
        if step == 'language':
            self._ask_text("Lang?")
            self._next_prompt='get_language'
            return []
        if step == 'get_language':
            self.language = text or 'English'
            self._ask_text(f"Lang {self.language}, level? {list(self.topics.keys())}")
            self._next_prompt='get_level'
            return []
        if step == 'get_level':
            self.level = text if text in self.topics else list(self.topics.keys())[0]
            tlist=self.topics[self.level]
            self._ask_text(f"Topics: {tlist}")
            self._next_prompt='get_topic'
            return []
        if step == 'get_topic':
            idx=int(text)-1 if text.isdigit() else 0
            self.idx=idx
            self.topic=self.topics[self.level][self.idx]
            self._ask_text(f"Topic {self.topic}, explain in? (default {self.language})")
            self._next_prompt='get_explanation'
            return []
        if step == 'get_explanation':
            self.explanation_language = text or self.language
            self._ask_text("Starting...")
            self._init_gemini()
            self._ask_text(f"Parlez '{self.topic}' en {self.language}.")
            self._next_prompt=None
            return []

    def _handle_next(self):
        self.idx+=1
        key=f"{self.language}_{self.level}"
        self.progress[key]=self.idx
        save_progress(self.progress)
        if self.idx<len(self.topics[self.level]):
            self.topic=self.topics[self.level][self.idx]
            return [f"Next: {self.topic}"]
        network.stop()
        return ["Done!"]

    def _handle_exit(self):
        key=f"{self.language}_{self.level}"
        self.progress[key]=self.idx
        save_progress(self.progress)
        
        network.stop()
        return ["Exit."]

class GeminiLLMModule(AbstractModule):
    
    @staticmethod
    def name():
        return "Gemini LLM Module"

    @staticmethod
    def description():
        return "Queries Gemini and streams TextIU responses."

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
        self,
        language: str,
        topic: str,
        difficulty_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"],
        explanation_language: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language = language
        self.topic = topic
        self.difficulty_level = difficulty_level
        self.explanation_language = explanation_language or language
        # Initialize Gemini client
        system = SYSTEM_PROMPT.format(
            language=self.language,
            topic=self.topic,
            difficulty_level=self.difficulty_level,
            explanation_language=self.explanation_language,
        )
        self.gemini = Gemini(system_instructions=system)
        self.in_buffer = []  # list of text chunks
        self.out_buffer = ""
        
        self._timeout_timer = None
        self._timeout_interval = 1.5 # seconds
        self._last_iu = None

    def process_update(self, update_message: UpdateMessage):
        for iu, ut in update_message:
            print(f"Processing IU: {iu}, UpdateType: {ut}, committed: {iu.committed if hasattr(iu, 'committed') else 'N/A'}")
            if not isinstance(iu, TextIU):
                continue
            text = iu.text
            self._last_iu = iu
            
            if ut == UpdateType.ADD:
                self.in_buffer.append(text)
                self._reset_timer() # If no IU is coming after 1.5 seconds, we process the input

            if getattr(iu, "committed", False):
                # If the incoming IU has the committed=True flag, we cancel the timer
                if self._timeout_timer is not None:
                    self._timeout_timer.cancel()
                # And immediately process the input
                self._on_timeout()
    
    def _on_timeout(self):
        user_input = ' '.join(self.in_buffer).strip()
        print("User input:", user_input)
        self.in_buffer.clear()
        if not user_input:
            return
        # Send to Gemini and stream response
        um = UpdateMessage()
        for chunk in self.gemini.add_turn(user_input):
            self.out_buffer += chunk
            if self.out_buffer.endswith((".", "!", "?")): # Only send full sentences
                out_iu = TextIU(iuid=0)
                out_iu.payload = out_iu.text = self.out_buffer
                self.out_buffer = ""
                um.add_iu(out_iu, UpdateType.ADD)
                self.append(um)
    
    def _reset_timer(self):
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._timeout_timer = threading.Timer(self._timeout_interval, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

if __name__ == "__main__":
    mic = MicrophoneModule(rate=16000)
    lgd_audio = LanguageDetectionModule()
    asr = GoogleASRModule(rate=16000)
    llm = GeminiLLMModule(
        language="Deutsch",
        topic="Cuisine",
        difficulty_level="A1",
        explanation_language="English"
    )
    lgd_text = LanguageDetectionModule()
    tts = MultilingualTTSModule()
    spk = SpeakerModule(rate=22050)
    
    mic.subscribe(lgd_audio)
    lgd_audio.subscribe(asr)
    asr.subscribe(llm)
    llm.subscribe(lgd_text)
    lgd_text.subscribe(tts)
    tts.subscribe(spk)
    
    network.run(mic)
    input("Running...\n")
    network.stop(mic)