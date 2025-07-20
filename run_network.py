from retico_core import network
from retico_core.audio import MicrophoneModule, SpeakerModule
from retico_googleasr import GoogleASRModule
from retico_language_detection import LanguageDetectionModule
from retico_multilingual_tts import MultilingualTTSModule

from gemini_llm_module import GeminiLLMModule

if __name__ == "__main__":
    mic = MicrophoneModule(rate=16000)
    lang_in = LanguageDetectionModule()
    asr = GoogleASRModule(rate=16000)
    llm = GeminiLLMModule()
    lang_out = LanguageDetectionModule()
    tts = MultilingualTTSModule()
    spk = SpeakerModule(rate=22050)
    
    mic.subscribe(llm)
    llm.subscribe(lang_out)
    lang_out.subscribe(tts)
    tts.subscribe(spk)
    
    network.run(mic)
    input("Running...\n")
    network.stop(mic)
