import threading
import queue
import retico_core


class SimpleTerminalInputIU(retico_core.IncrementalUnit):
    """IU pour l'entrée texte du terminal."""

    @staticmethod
    def type():
        return "Simple Terminal Input IU"

    def __init__(self, text=None, **kwargs):
        super().__init__(payload=text, **kwargs)
        self.text = text

    def set_text(self, text):
        self.text = text
        self.payload = text


class SimpleTerminalInputModule(retico_core.AbstractProducingModule):
    """Module Retico conforme pour l'entrée texte utilisateur via le terminal."""

    @staticmethod
    def name():
        return "Simple Terminal Input Module"

    @staticmethod
    def description():
        return "Un module produisant des IUs à partir de l'entrée texte du terminal."

    @staticmethod
    def output_iu():
        return SimpleTerminalInputIU

    def __init__(self, prompt="Vous: ", **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.input_queue = queue.Queue()
        self._running = False
        self._thread = None

    def _input_thread(self):
        while self._running:
            try:
                user_input = input(self.prompt)
                if user_input.strip() == "":
                    continue
                self.input_queue.put(user_input)
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

    def process_update(self):
        if not self.input_queue:
            return None
        try:
            text = self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return None
        iu = self.create_iu()
        iu.set_text(text)
        self.append(retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD))

    def setup(self):
        self._running = True
        self._thread = threading.Thread(target=self._input_thread, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.input_queue = queue.Queue()
