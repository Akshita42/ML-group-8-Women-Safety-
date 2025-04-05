import pvporcupine
import pyaudio
import struct
import os
import geocoder
from dotenv import load_dotenv
from twilio.rest import Client
from flask import Flask, jsonify, request
import threading

# Load environment variables
load_dotenv("twilio.env")

# Twilio + Porcupine keys
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE")
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# Flask app
app = Flask(__name__)
latest_gps_location = None  # ⬅️ Accurate location from phone will be saved here

# Use GPS location from phone if available
def get_location():
    if latest_gps_location:
        return latest_gps_location
    location = geocoder.ip('me')
    if location.ok:
        return f"https://www.google.com/maps?q={location.latlng[0]},{location.latlng[1]}"
    return "Location not found"

# Endpoint to receive GPS location from Kivy Android app
@app.route('/save-location')
def save_location():
    global latest_gps_location
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    if lat and lng:
        latest_gps_location = f"https://www.google.com/maps?q={lat},{lng}"
        print("📍 Accurate GPS Location received:", latest_gps_location)
        return "Location saved!"
    return "Missing lat/lng", 400

# For checking current location (testing)
@app.route('/location')
def location_api():
    return jsonify({"location": get_location()})

@app.route("/")
def home():
    return "Wake word SOS system running!"

# Send SOS alert via SMS
def send_sos_alert():
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    location_link = get_location()
    client.messages.create(
        body=f"🚨 SOS Alert! I need help!\nMy Location: {location_link}",
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_CONTACT
    )
    print("✅ SOS Alert Sent!")

# Wake word detection
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
                detected_word = "help me"
                print(f"🔊 Wake Word Detected: {detected_word}")
                send_sos_alert()
            elif keyword_index == 1:
                detected_word = "Red Red Red"
                print(f"🔊 Wake Word Detected: {detected_word}")
                send_sos_alert()


    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()

# Run Flask + Wake Word in parallel
if __name__ == '__main__':
    threading.Thread(target=listen_for_wake_word, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
