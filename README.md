# ML-group-8-Women-Safety-
Women Safety SOS Wake Word Detection

📌 About the Project
This project is an SOS Alert System that listens for a custom wake word ("Help Me") and sends an emergency SMS alert to a predefined contact using Twilio. 

It is built using:

Porcupine for wake word detection

Twilio API for sending SMS alerts

Python for implementation

✨ Features:-

Wake Word Activation – Detects the phrase "Help Me" using Porcupine.

Emergency SMS Alert – Sends an SOS message to a predefined emergency contact.

Environment Variables for Security – Stores sensitive credentials in a .env file.

Customizable – Can be extended with gesture recognition, GPS tracking, etc.

🛠 Technologies Used :- 

Python

Porcupine (Picovoice)

Twilio API

pyaudio

dotenv

🚀 How to Run

1️⃣ Install Required Libraries

bash:-
pip install pvporcupine pyaudio twilio python-dotenv

2️⃣ Set Up Twilio Credentials

Create a twilio.env file in the project directory.

Add the following credentials (replace with your actual Twilio details):

TWILIO_ACCOUNT_SID=your_twilio_sid

TWILIO_AUTH_TOKEN=your_twilio_auth_token

TWILIO_PHONE=your_twilio_phone_number

EMERGENCY_CONTACT=your_emergency_contact_number

3️⃣ Run the Script

bash:-
python Wakeword.py

How It Works:- 

The script listens for the wake word using the microphone.

If the wake word "Help Me" is detected, the script triggers an SOS alert.

An SMS alert is sent to the predefined emergency contact.

Future Improvements:-

Gesture-Based SOS – Shake detection for silent distress alerts.

GPS Location Sharing – Send location along with SOS.

Offline Support – Work without an internet connection.

