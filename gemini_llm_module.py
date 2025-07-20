import json
import numpy as np
import os
import threading
import webrtcvad

from collections import deque
from typing import Literal, Union

from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.text import TextIU
from retico_core.audio import AudioIU

from gemini import Gemini

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
7. Do NOT use double quotes whatsoever (`"`). If you ever need to use double quotation marks in your response, you need to replace them with single quotes (`'`) or French guillemets (`«»`).

If asked, give cultural info, vocabulary, or grammar tips. Always be encouraging and keep answers short and clear. Answers should not be more than 2-3 sentences long.

Start by greeting the user in {language} and asking a simple question about “{topic}” at level {difficulty_level}, when the user is ready to start the conversation. Do not converse until the user has provided all necessary parameters.

Be familiar, friendly amd helpful to the user. This includes using a friendly tone, using the user's name if provided, and being encouraging and supportive throughout the conversation.

Always ask a follow-up question whenever you reply, do not let the conversation die out unless explicitely told by the user.

- When is the user ready to start the conversation?
-> You will first have a phase of parameter collection (if needed). Your goal is to collect the following parameters from the user:
1. `language`: The language the user wants to practice (e.g., "French", "English"). Should be converted to ISO 639-1.
2. `topic`: The chosen topic for the conversation. If you set `topic` to `True`, this will be further defined by the current user's progress in the default language learning track. In consequence, you should ask if the user wants to start/continue on this curated track, or pick a custom topic. If the user chooses the default track, the `topic` variable will be set to the next topic in the list of topics for the current language. You should confirm this to the user, once you have a value for that variable.
3. `difficulty_level`: The user's proficiency level, that should be converted to an approximate CECRL equivalent (A1 to C2). You can ask for this directly, or infer it from the user's input.
4. `explanation_language`: The language for grammar explanations, which might be different from the conversation language (e.g., if the user wants to practice French but prefers explanations in English).
-> They are set at `None` by default. As long as they are not None, you can consider them set. If the user tells you about one of these parameters during the initial phase, you will update your internal state accordingly. Remember that `None` in Python corresponds to `null` in JSON.

Your Interaction Flow:
- Start by asking for the `language`. You can default to English is {language} is not set, but should switch to the user's preferred language as soon as it is set. If the user is saying to be struggling in the language you are using, you should adapt to it and use the user's preffered language more often while reducing the amount of utterances in the target language, if these two are different.
- Analyze the user's response. They might provide multiple details at once.
- For each piece of information you gather, update your internal state.
- **Your response MUST be a JSON object.**
- The JSON should be Python-like dictionary containing the following:
  -   The parameters you have identified (or not) so far (e.g., `{{"language": "French", "topic": null, "difficulty_level": "A2", "explanation_language": null}}`).
  -   A key named `response` containing the next question or instruction for the user. If everything is set, this should be a question to start the conversation, such as "Let's start! What would you like to talk about?" or "What is your favorite thing about {topic}?".
- If not all parameters are collected, you should proactively ask the user for the missing parameters.
- If all parameters are collected, `response_to_user` should be set to the boolean `True`.
- Conduct the conversation based on the information provided earlier.

Here are the information that you already have about the user parameters:
- `language`: {language}
- `topic`: {topic}
- `difficulty_level`: {difficulty_level}
- `explanation_language`: {explanation_language}
"""

# Load topics
with open(os.path.join(os.path.dirname(__file__), "topics.json"), encoding="utf-8") as f:
    TOPICS: dict[str, list[str]] = json.load(f)

# Progress of the user in the language learning track
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "user_progress.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            print("Loaded user progress from", PROGRESS_FILE)
            return json.load(open(PROGRESS_FILE, encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}

def save_progress(progress):
    json.dump(progress, open(PROGRESS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """
    Split `audio` (PCM16 bytes) into frames of `frame_duration_ms` milliseconds.
    Yields tuples of (frame_bytes, timestamp_ms).
    """
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0)) * 2
    offset = 0
    timestamp = 0.0
    while offset + bytes_per_frame <= len(audio):
        yield audio[offset : offset + bytes_per_frame], timestamp
        timestamp += frame_duration_ms
        offset += bytes_per_frame

def has_speech(
    audio: bytes,
    sample_rate: int,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
    speech_frame_threshold: float = 0.1
) -> bool:
    """
    Return True if at least `speech_frame_threshold` proportion of frames
    contain speech according to WebRTC VAD.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(frame_duration_ms, audio, sample_rate))
    if not frames:
        return False
    speech_frames = 0
    for frame_bytes, _ in frames:
        if vad.is_speech(frame_bytes, sample_rate):
            speech_frames += 1
    return (speech_frames / len(frames)) >= speech_frame_threshold

class GeminiLLMModule(AbstractModule):
    
    @staticmethod
    def name():
        return "Gemini LLM Module"

    @staticmethod
    def description():
        return "Queries Gemini and streams TextIU responses."

    @staticmethod
    def input_ius():
        return [AudioIU, TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
        self,
        language: Union[str, None] = None,
        topic: Union[str, None] = None,
        difficulty_level: Union[Literal["A1", "A2", "B1", "B2", "C1", "C2"], None] = None,
        explanation_language: Union[str, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language = language
        self.topic = topic
        self.custom_topic_id: int = None # Will be set if the user chooses a topic from the predefined list
        self.current_progress = load_progress()  # Load user progress from file
        self.difficulty_level = difficulty_level
        self.explanation_language = explanation_language or language
        
        # Initialize Gemini client
        self.gemini = None # Will be immediately instantiated
        self.set_model_instructions()
        self.in_text_buffer = []  # List of yielded text chunks
        self.in_audio_buffer = deque()  # Queue of input audio chunks
        self.out_buffer = ""
        
        self._timeout_timer = None
        self._timeout_interval = 1.5 # seconds
        self._last_iu = None
        self._speech_started = False
    
    def set_model_instructions(self):
        system = SYSTEM_PROMPT.format(
            language=self.language,
            topic=self.topic,
            difficulty_level=self.difficulty_level,
            explanation_language=self.explanation_language,
        )
        if self.gemini is not None:
            contents = self.gemini.contents # We transfer the previous messages to the new instance
        else:
            contents = None
        self.gemini = Gemini(system_instructions=system, contents=contents)

    def process_update(self, update_message: UpdateMessage):
        for iu, ut in update_message:
            if isinstance(iu, TextIU):
                text = iu.text
                self._last_iu = iu
                
                if ut == UpdateType.ADD:
                    self.in_text_buffer.append(text)
                    self._reset_timer() # If no IU is coming after 1.5 seconds, we process the input

                if getattr(iu, "committed", False):
                    # If the incoming IU has the committed=True flag, we cancel the timer
                    if self._timeout_timer is not None:
                        self._timeout_timer.cancel()
                    # And immediately process the input
                    self._on_timeout()
            elif isinstance(iu, AudioIU):
                self._last_iu = iu
            
                if ut == UpdateType.ADD:
                    self.in_audio_buffer.append(iu.raw_audio)

                np_audio = np.frombuffer(b"".join(self.in_audio_buffer), dtype="<i2").astype(np.float32) / 32768
                rms = np.sqrt(np.mean(np_audio**2))
                
                if rms > 0.015: # We consider that this level of voice activity means the user is speaking
                    self._speech_started = True
                
                if self._speech_started and has_speech(b"".join(self.in_audio_buffer), iu.rate) and self.is_tail_silence_or_noise(np_audio, iu.rate):
                    audio_input = b"".join(self.in_audio_buffer)
                    self.in_audio_buffer.clear()  # Clear the buffer after processing
                    self._speech_started = False
                    um = UpdateMessage()
                    
                    for chunk in self.gemini.add_audio_turn(audio_input):
                        self.out_buffer += chunk
                        self.process_json_from_llm(um)
    
    def _on_timeout(self):
        user_input = ' '.join(self.in_text_buffer).strip()
        print("User input:", user_input)
        self.in_text_buffer.clear()
        if not user_input:
            return
        # Send to Gemini and stream response
        um = UpdateMessage()
        for chunk in self.gemini.add_turn(user_input):
            self.out_buffer += chunk
            self.process_json_from_llm(um)
    
    def _reset_timer(self):
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._timeout_timer = threading.Timer(self._timeout_interval, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()
    
    def process_json_from_llm(self, um: UpdateMessage):
        if self.out_buffer.startswith("{"):
            if not self.out_buffer.endswith("}"):
                return
            try: # If we are setting up the conversation parameters
                response = json.loads(self.out_buffer)
                if isinstance(response, dict) and "language" in response and "topic" in response and "difficulty_level" in response and "explanation_language" and "response" in response:
                    self.language = response["language"]
                    self.topic = response["topic"]
                    if isinstance(self.topic, bool) and self.topic:
                        progress = load_progress()
                        if progress and hasattr(progress, self.language):
                            self.custom_topic_id = (TOPICS.get(self.language)[progress[self.language] + 1], None)
                            save_progress({f"{self.language}": self.custom_topic_id})
                            self.topic = TOPICS.get(self.language)[self.custom_topic_id]
                    self.difficulty_level = response["difficulty_level"]
                    self.explanation_language = response["explanation_language"]
                    utterance = response["response"]
                    
                    out_iu = TextIU(iuid=0, previous_iu=self._last_iu)
                    out_iu.payload = out_iu.text = utterance
                    self.out_buffer = ""
                    um.add_iu(out_iu, UpdateType.ADD)
                    self.append(um)
            except json.JSONDecodeError:
                print("Invalid JSON response:", self.out_buffer)
            finally:
                self.out_buffer = ""
        elif self.out_buffer.endswith((".", "!", "?")): # Only send full sentences
            out_iu = TextIU(iuid=0, previous_iu=self._last_iu)
            out_iu.payload = out_iu.text = self.out_buffer
            self.out_buffer = ""
            um.add_iu(out_iu, UpdateType.ADD)
            self.append(um)

    @staticmethod
    def is_tail_silence_or_noise(buffer: np.ndarray, rate: int, silent_tail_size: float = 1.0, silence_max_rms_energy_threshold = 0.01) -> bool:
        """
        Returns True if the last `tail_size` seconds of `buffer`
        are essentially silence (low RMS) or noise (high spectral flatness).

        :param buffer: the audio buffer to check
        :param rate: the sample rate of the audio buffer (in Hz)
        :param silent_tail_size: the size of the tail to check whether it's deemed silent or not (in seconds)
        :param silence_max_rootmeansquare_energy_threshold: the maximum root mean square energy to consider the tail as silence (closer to 0: silence)
        """

        n_tail = int(silent_tail_size * rate)
        if buffer.size < n_tail:
            return False
        tail = buffer[-n_tail:]
        rms = np.sqrt(np.mean(tail**2))
        if rms < silence_max_rms_energy_threshold:
            return True
        return False