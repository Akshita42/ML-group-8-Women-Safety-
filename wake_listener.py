import pvporcupine
import pyaudio
import struct
import os
from dotenv import load_dotenv
import threading

load_dotenv("twilio.env")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

def listen_for_wake_word():
    porcupine = pvporcupine.create(
        access_key=PICOVOICE_ACCESS_KEY,
        keyword_paths=[
            r"C:\Users\akshi\Downloads\Help-me_en_windows_v3_0_0 (2)\Help-me_en_windows_v3_0_0.ppn",
            r"C:\Users\akshi\Downloads\Red-Red-Red_en_windows_v3_0_0\Red-Red-Red_en_windows_v3_0_0.ppn"
        ]
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("🎤 Listening for wake word...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index == 0:
                print("🔊 Wake Word Detected: help me")
                return True
            elif keyword_index == 1:
                print("🔊 Wake Word Detected: red red red")
                return True

    except KeyboardInterrupt:
        print("🛑 Wake word listening stopped.")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()

    return False

def start_wake_word_detection(callback):
    def detect():
        if listen_for_wake_word():
            callback()
    threading.Thread(target=detect, daemon=True).start()
