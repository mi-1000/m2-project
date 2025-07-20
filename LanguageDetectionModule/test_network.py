from retico_core import AbstractModule, AbstractTriggerModule, UpdateMessage, UpdateType, network
from retico_core.audio import MicrophoneModule, SpeakerModule, AudioDispatcherModule
from retico_core.debug import CallbackModule
from retico_core.text import SpeechRecognitionIU, TextIU
from retico_googleasr import GoogleASRModule
# from retico_mistyrobot import misty_action
# from retico_opendialdm.dm import OpenDialModule, DialogueActIU, DialogueDecisionIU
# from retico_speechbraintts import SpeechBrainTTSModule
from retico_multilingual_tts import MultilingualTTSModule
# from ..RobotFilterModule.misty_robot_asr_filter import RobotASRFilterModule

from retico_language_detection import LanguageDetectionIU, LanguageDetectionModule

import os
import re
import threading
import time

from dotenv import load_dotenv
load_dotenv()
os.environ['PYOD']="./pyopendial/"

opendial_variables = ['firstname', 'lastname', 'work', 'email', 'notes']
domain_dir = 'dialogue.xml'

class SimpleNLUModule(AbstractModule):
    def __init__(self):
        super().__init__()
        self.word_buffer = []
        
    @staticmethod
    def name():
        return "Simple NLU Module"

    @staticmethod
    def description():
        return "A module that performs simple NLU."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU, TextIU]

    @staticmethod
    def output_iu():
        return DialogueActIU
    
    def process_update(self, update: UpdateMessage):
        for iu, _ in update:
            word = iu.payload
            if word:
                self.word_buffer.append(word)
            if iu.final:
                utterance = ' '.join(self.word_buffer).lower()
                self.word_buffer.clear()
            else:
                utterance = ' '.join(self.word_buffer).lower()
            print(f"[DEBUG] Current utterance: {utterance}")
            payload = {}
            # First name
            fn_match = re.search(r"first name is ([a-zA-Z'-]+)", utterance)
            if fn_match:
                payload['firstname'] = fn_match.group(1)
            # Last name
            ln_match = re.search(r"last name is ([a-zA-Z'-]+)", utterance)
            if ln_match:
                payload['lastname'] = ln_match.group(1)
            # Work
            work_match = re.search(r"work(s|ed|ing)? (at|in) ([a-zA-Z0-9' -]+)", utterance)
            if work_match:
                payload['work'] = work_match.group(3)
            # Email
            utterance = utterance.lower()

            email_match = re.search(
                r"email is\s+([\w\s\.\-]+\s*(?:@|at)\s*[\w\.-]+\s*(?:\.|dot)\s*[a-zA-Z]{2,3})",
                utterance
            )
            if email_match:
                payload["email"] = email_match.group(1).strip().replace(' ', '').replace('at', '@').replace('dot', '.')
            # Note
            note_match = re.search(r"(?:well|um|so|yep|yeah|yes|add|say|tell) ([a-zA-Z0-9' -]+)", utterance)
            if note_match:
                payload['notes'] = note_match.group(1)
        iu_out = DialogueActIU(payload=payload)
        
        um = UpdateMessage()
        um.add_iu(iu_out, UpdateType.ADD)
        self.append(um)

class ResponseModule(AbstractModule):
    def __init__(self):
        super().__init__()
        
    responses = {
        'ask_about_firstname': "What is the person's first name?",
        'ask_about_lastname': "What is {firstname}'s last name?",
        'ask_about_work': "Where does {firstname} {lastname} work?",
        'ask_about_email': "What is {firstname} {lastname}'s email?",
        'ask_about_notes': "Do you have a note to add about {firstname} {lastname}?",
        'all_slots_filled': "That's all I need for this person! Let's recap: Name: {firstname} {lastname}. Work: {work}. Email: {email}. Notes: {notes}. Is that correct?",
    }
    
    @staticmethod
    def name():
        return "Response Module"

    @staticmethod
    def description():
        return "A module that generates responses based on DM decisions."

    @staticmethod
    def input_ius():
        return [DialogueDecisionIU]

    @staticmethod
    def output_iu():
        return TextIU
    
    def process_update(self, update: UpdateMessage):
        for iu, _ in update:
            decision = iu.payload.get('decision', None)
            dialogue_vars = iu.payload.get('concepts', {})
            if decision is None:
                print(f"[WARNING] No 'decision' in DM output payload: {iu.payload}")
            response = self.responses.get(decision, "I didn't understand, could you please repeat?")
            for var in dialogue_vars: # Template-based response generation
                if dialogue_vars[var] is not None:
                    response = response.format(**dialogue_vars)
            
            iu_out = TextIU(payload=response, iuid=hash(response)) # Getting errors when no iuid is provided, this is an ugly fix
            um = UpdateMessage()
            um.add_iu(iu_out, UpdateType.ADD)
            um.add_iu(iu=TextIU(payload='', iuid=hash(time.time())), update_type=UpdateType.COMMIT)
            self.append(um)

class SimpleTerminalInputModule(AbstractTriggerModule):
    @staticmethod
    def name():
        return "Terminal Input Module"

    @staticmethod
    def description():
        return "Reads text input from the terminal and outputs it as TextIU."

    @staticmethod
    def input_ius():
        return []

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                os.system("cls" if os.name == "nt" else "clear")
                line = input("Enter text: ")
            except (EOFError, KeyboardInterrupt):
                self._stop_event.set()
                break
            text = line.strip()
            if not text:
                continue
            self.trigger({"text": text}, UpdateType.COMMIT)

    def trigger(self, data={}, update_type=UpdateType.ADD):
        text = data.get("text", "")
        iu = TextIU(self, iuid=0, payload=text)
        um = UpdateMessage()
        um.add_iu(iu, update_type)
        self.append(um)

def callback(update_msg):
    for iu, ut in update_msg:
        if hasattr(iu, 'language') and iu.language:
            print(f"Detected language: '{iu.language}' with confidence {iu.confidence:.2f}")
        else:
            text = getattr(iu, 'text', iu.payload if hasattr(iu, 'payload') else None)
            print(f"{ut}: {text}")

if __name__ == "__main__":

    # ---- Text ----

    # ter = SimpleTerminalInputModule()
    # lgr = LanguageDetectionModule()
    # debug = CallbackModule(callback)
    
    # ter.subscribe(lgr)
    # lgr.subscribe(debug)
    
    # network.run(ter)
    
    # print("Running...\n")

    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     network.stop(ter)
    
    # ---- Audio ----
    
    # mic = MicrophoneModule(rate=16000, frame_length=0.2)
    # lgr_audio = LanguageDetectionModule()
    # asr = GoogleASRModule(rate=16000)
    # lgr_text = LanguageDetectionModule()
    # tts = MultilingualTTSModule()
    # spk = SpeakerModule(rate=22050)
    # debug = CallbackModule(callback)
    
    # mic.subscribe(lgr_audio)
    # lgr_audio.subscribe(asr)
    # asr.subscribe(lgr_text)
    # lgr_text.subscribe(tts)
    # tts.subscribe(spk)
    
    # asr.subscribe(debug)
    
    # network.run(mic)
    
    # input("Running...\n")
    
    # network.stop(mic)
    
    # ----- TTS ----- #

    ter = SimpleTerminalInputModule()
    lgr = LanguageDetectionModule()
    tts = MultilingualTTSModule()
    spk = SpeakerModule(rate=22050)
    debug = CallbackModule(callback)
    
    ter.subscribe(lgr)
    lgr.subscribe(tts)
    tts.subscribe(spk)
    
    lgr.subscribe(debug)
    
    network.run(ter)
    
    print("Running...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        network.stop(ter)