import pvporcupine  # Detects wake word
import pyaudio  # Handles microphone input
import struct  # Converts audio data
from twilio.rest import Client  # Sends SMS alerts
import os # Lets us use system variables 
from dotenv import load_dotenv #Used to get secret info from .env file
from twilio.rest import Client # Needed to send SMS using Twilio

load_dotenv(r"C:\Users\akshi\OneDrive\Desktop\Women safety\twilio.env") # Load the .env file that contains Twilio credentials

# Get the Twilio details from the .env file
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE") #Twilio number that sends the message  
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT") #number that will receive the SOS alert

# It initializes Porcupine with a custom wake word model
porcupine = pvporcupine.create(
    access_key="hmq36RWoWvdsQdKbLsm6l5dyWn9bvdNj86J4vpq+cuqQG7xy45iN9Q==",  # Picovoice key
    keyword_paths=[r"C:\Users\akshi\Downloads\help-me_en_windows_v3_0_0 (1)\help-me_en_windows_v3_0_0.ppn"]  #wake word file path
)

# sets up microphone input
pa = pyaudio.PyAudio()
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print(" Listening for 'Help Me'...")




# sends an SOS message
def send_sos_alert():
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)  # It initializes Twilio client
    message = client.messages.create(
        body="SOS Alert! I need help!",
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_CONTACT
    )
    print(" SOS Alert Sent!")  # It confirms SOS was sent

# listens for wake word
while True:
    pcm = stream.read(porcupine.frame_length)  # captures audio
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)  # It processes audio

    if porcupine.process(pcm) >= 0:  # It checks if wake word is detected
        print(" Wake Word Detected! Sending SOS...")
        send_sos_alert()  
        break  

# cleans up resources
stream.close()
pa.terminate()
porcupine.delete()
