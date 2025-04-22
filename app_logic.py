from wake_word.wake_listener import listen_for_wake_word
from distress.predict_distress import predict_distress
from gesture.predict_gesture import detect_shake
from utils.sos_trigger import trigger_sos
import time

def run_system():
    print("🛡️ Backend system is running...")

    while True:
        try:
            if listen_for_wake_word():
                print("🎙️ Wake word detected.")
                if predict_distress():
                    print("😨 Distress confirmed.")
                    trigger_sos("Wakeword + Distress")
                else:
                    print("🙂 No distress detected.")

            if detect_shake():
                print("📳 Shake detected.")
                trigger_sos("Shake")

            time.sleep(1)

        except KeyboardInterrupt:
            print("🛑 Manually stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
