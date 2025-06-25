import pyttsx3


class TTS:
    def __init__(self, rate=150, voice=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)  # speaking speed
        self.engine.setProperty(
            'voice', self.engine.getProperty('voices')[1].id)

    def speak(self, bus_number, location):
        """Speak the given text aloud"""
        print(f"[TTS] Speaking:")
        self.engine.say(f"Bus {bus_number} heading to {location} arriving.")
        self.engine.runAndWait()

    def speak_bus_number(self, bus_number):
        """Speak the given text aloud"""
        print(f"[TTS] Speaking:")
        self.engine.say(f"Bus {bus_number} arriving.")
        self.engine.runAndWait()

    def fail(self):
        self.engine.say("Unsure of bus number.")
        self.engine.runAndWait()

    def fail_to_read_number(self):
        self.engine.say("Unsure of bus number.")
        self.engine.runAndWait()
