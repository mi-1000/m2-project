import json
import numpy as np
import os
import threading
from collections import deque
from typing import Literal, Union

from retico_core import network, AbstractModule, UpdateMessage, UpdateType
from retico_core.text import TextIU
from retico_core.audio import AudioIU, MicrophoneModule, SpeakerModule
from retico_googleasr import GoogleASRModule

from gemini import Gemini
from LanguageDetectionModule.language_detection import LanguageDetectionModule, is_tail_silence_or_noise
from LanguageDetectionModule.multilingual_tts import MultilingualTTSModule
from RobotFilterModule.filter import RobotASRFilterModule

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

Start by greeting the user in {language} and asking a simple question about “{topic}” at level {difficulty_level}, when the user is ready to start the conversation. Do not converse until the user has provided all necessary parameters.

- When is the user ready to start the conversation?
-> You will first have a phase of parameter collection (if needed). Your goal is to collect the following parameters from the user:
1. `language`: The language the user wants to practice (e.g., "French", "English"). Should be converted to ISO 639-1.
2. `topic`: The chosen topic for the conversation. If you set `topic` to `True`, this will be further defined by the current user's progress in the default language learning track. In consequence, you should ask if the user wants to start/continue on this curated track, or pick a custom topic.
3. `difficulty_level`: The user's proficiency level, that should be converted to an approximate CECRL equivalent (A1 to C2). You can ask for this directly, or infer it from the user's input.
4. `explanation_language`: The language for grammar explanations, which might be different from the conversation language (e.g., if the user wants to practice French but prefers explanations in English).
-> They are set at `None` by default. As long as they are not None, you can consider them set. If the user tells you about one of these parameters during the initial phase, you will update your internal state accordingly.

Your Interaction Flow:
- Start by asking for the `language`. You can default to English is {language} is not set, but should switch to the user's preferred language as soon as it is set.
- Analyze the user's response. They might provide multiple details at once.
- For each piece of information you gather, update your internal state.
- **Your response MUST be a JSON object.**
- The JSON should be Python-like dictionary containing the following:
  -   The parameters you have identified (or not) so far (e.g., `{{"language": "French", "topic": None, "difficulty_level": "A2", "explanation_language": None}}`).
  -   A key named `response` containing the next question or instruction for the user. If everything is set, this should be a question to start the conversation, such as "Let's start! What would you like to talk about?" or "What is your favorite thing about {topic}?". If you need to enclose something in quotes, use single quotes ONLY ('), and no double quotes whatsoever ("). If you ever need to use double quotation marks in your response, you need to escape them with a backslash (e.g., `\\"`). This is the only case you can use backslashes in your response.
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

# Persistence
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "user_progress.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            return json.load(open(PROGRESS_FILE, encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}

def save_progress(progress):
    json.dump(progress, open(PROGRESS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

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
                if rms > 0.01: # We consider that this is not silence
                    self._speech_started = True
                
                if self._speech_started and is_tail_silence_or_noise(np_audio, iu.rate, silent_tail_size=1.0):
                    um = UpdateMessage()
                    audio_input = b"".join(self.in_audio_buffer)
                    self.in_audio_buffer.clear()  # Clear the buffer after processing
                    self._speech_started = False
                    
                    for chunk in self.gemini.add_audio_turn(audio_input):
                        self.out_buffer += chunk
                        if self.out_buffer.startswith("{"):
                            if not self.out_buffer.endswith("}"):
                                continue
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
            if self.out_buffer.startswith("{"):
                if not self.out_buffer.endswith("}"):
                    continue
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
    
    def _reset_timer(self):
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._timeout_timer = threading.Timer(self._timeout_interval, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

if __name__ == "__main__":
    mic = MicrophoneModule(rate=16000)
    # lang_in = LanguageDetectionModule()
    # asr = GoogleASRModule(rate=16000)
    llm = GeminiLLMModule()
    lang_out = LanguageDetectionModule()
    tts = MultilingualTTSModule()
    fil = RobotASRFilterModule()
    spk = SpeakerModule(rate=22050)
    
    mic.subscribe(llm)
    # asr.subscribe(lang_in)
    # lang_in.subscribe(llm)
    llm.subscribe(lang_out)
    lang_out.subscribe(tts)
    tts.subscribe(spk)
    
    network.run(mic)
    input("Running...\n")
    network.stop(mic)