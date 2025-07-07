import json
import os

from librosa import ex
from retico_core import network, AbstractModule
from retico_core.text import TextIU
from retico_mistyrobot.misty_action import MistyActionModule
from retico_mistyrobot.mistyPy import Robot
from LanguageDetectionModule.language_detection import (
    SimpleTerminalInputModule,
    LanguageDetectionModule,
)
from LanguageDetectionModule.multilingual_tts import MultilingualTTSModule

from gemini import Gemini

SYSTEM_PROMPT = """
You are a friendly and knowledgeable language tutor. Help the user practice {language} by having a short, natural conversation about “{topic}” at level {difficulty_level}.

Guidelines:
1. Speak mostly in {language}, using simple words and grammar for level {difficulty_level}.
2. Ask short, open questions about {topic}. Encourage the user to answer in their own words.
3. If the user makes a mistake, gently correct them:
   - Give the correct sentence.
   - Briefly explain the rule (in {explanation_language}).
   - Optionally, give a quick tip or example.

If asked, give cultural info, vocabulary, or grammar tips. Always be encouraging and keep answers short and clear.

Start by greeting the user in {language} and asking a simple question about “{topic}” at level {difficulty_level}."""

# Load topics
with open(
    os.path.join(os.path.dirname(__file__), "topics.json"), encoding="utf-8"
) as f:
    TOPICS = json.load(f)

# Cache for user progress
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "user_progress.json")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

class GeminiLLMModule(AbstractModule):
    @staticmethod
    def name():
        return "Gemini LLM Module"

    @staticmethod
    def description():
        return "A module that streams responses from Gemini LLM given a user prompt."

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
        self,
        conversation_language,
        topic,
        difficulty_level,
        explanation_language,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conversation_language = conversation_language
        self.topic = topic
        self.difficulty_level = difficulty_level
        self.explanation_language = explanation_language
        self.gemini = Gemini(
            system_instructions=SYSTEM_PROMPT.format(
                language=conversation_language,
                topic=topic,
                difficulty_level=difficulty_level,
                explanation_language=explanation_language or conversation_language,
            )
        )
        self.buffer = ""
        self.topic_advanced = False
        self.should_exit = False

    def process_update(self, update):
        user_input = update.text.strip()
        if user_input.lower() == "next":
            self.topic_advanced = True
            print("Moving to the next topic...")
            network.stop()
            return
        if user_input.lower() == "exit":
            self.should_exit = True
            print("Exiting. Progress saved.")
            network.stop()
            return
        self.buffer = ""
        for chunk in self.gemini.add_turn(user_input):
            self.buffer += chunk
            self.append(TextIU(text=chunk))


class ConversationParameterModule(AbstractModule):
    @staticmethod
    def name():
        return "Conversation Parameter Module"

    @staticmethod
    def description():
        return "A module that collects conversation parameters from the user."

    @staticmethod
    def input_ius():
        return []

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, topics, **kwargs):
        super().__init__(**kwargs)
        self.topics = topics
        self.language = None
        self.level = None
        self.topic = None
        self.difficulty_level = None
        self.explanation_language = None
        self.idx = 0
        self._paused = False

    def run(self):
        print("Welcome to the Retico Language Practice System!")
        self.language = input("Which language do you want to practice? (e.g., French, English, Magyar): ")
        levels = list(self.topics.keys())
        print("Available levels:", ", ".join(levels).strip(", "))
        self.level = input(f"Choose your level {levels}: ")
        if self.level not in self.topics:
            print("Invalid level. Defaulting to A1.")
            self.level = "A1"
        self.difficulty_level = self.level
        self.explanation_language = input(f"In which language do you want explanations? (default: {self.language}): ") or self.language
        topics = self.topics[self.level]
        self.idx = 0
        print("Available topics:")
        for i, t in enumerate(topics):
            print(f"{i+1}. {t}")
        topic_choice = input(f"Choose a topic by number (1-{len(topics)}) or press Enter for the first: ")
        if topic_choice.isdigit() and 1 <= int(topic_choice) <= len(topics):
            self.idx = int(topic_choice) - 1
        self.topic = topics[self.idx]
        # Pass IU to next module
        self.append(TextIU(text=f"LANG:{self.language}|LEVEL:{self.level}|TOPIC:{self.topic}|EXPL:{self.explanation_language}|IDX:{self.idx}", iuid = 0))
        self._paused = True
        return self.language, self.topic, self.difficulty_level, self.explanation_language, self.idx

    def process_update(self, update):
        # No-op: this module only produces IU at start
        pass


if __name__ == "__main__":
    try:
        progress = load_progress()
        param_mod = ConversationParameterModule(TOPICS)
        # Lancer la pipeline avec param_mod comme source
        # Les modules suivants doivent être créés dynamiquement après réception des paramètres
        language, topic, level, explanation_language, idx = param_mod.run()
        user_key = f"{language}_{level}"
        topics = TOPICS[level]
        while idx < len(topics):
            topic = topics[idx]
            print(f"Your next topic is: {topic}")
            llm = GeminiLLMModule(
                conversation_language=language,
                topic=topic,
                difficulty_level=level,
                explanation_language=explanation_language
            )
            lang = LanguageDetectionModule()
            tts = MultilingualTTSModule(language=llm.conversation_language)
            # Pipeline : param_mod -> llm -> lang -> tts
            param_mod.subscribe(llm)
            llm.subscribe(lang)
            lang.subscribe(tts)
            print("Type 'next' to move to the next topic, or 'exit' to quit.")
            network.run(param_mod)
            if hasattr(llm, "topic_advanced") and llm.topic_advanced:
                idx += 1
                progress[user_key] = idx
                save_progress(progress)
                print("Moving to the next topic...")
                continue
            if hasattr(llm, "should_exit") and llm.should_exit:
                print("Exiting. Progress saved.")
                progress[user_key] = idx
                save_progress(progress)
                exit()
        print("Congratulations! You have completed all topics for this level.")
        progress[user_key] = idx
        save_progress(progress)
    except KeyboardInterrupt:
        print("\nExiting dialogue. Progress saved.")
        if user_key in progress:
            progress[user_key] = idx
        save_progress(progress)
        exit()