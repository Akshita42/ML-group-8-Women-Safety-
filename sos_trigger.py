# utils/sos_trigger.py

import os
from dotenv import load_dotenv
from twilio.rest import Client
import geocoder

load_dotenv("twilio.env")

# Load from env
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE")
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT")

latest_gps_location = None

def set_latest_location(lat, lng):
    global latest_gps_location
    latest_gps_location = f"https://www.google.com/maps?q={lat},{lng}"

def get_location():
    if latest_gps_location:
        return latest_gps_location
    loc = geocoder.ip('me')
    if loc.ok:
        return f"https://www.google.com/maps?q={loc.latlng[0]},{loc.latlng[1]}"
    return "Location not found"

def trigger_sos(source="Unknown"):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        location = get_location()
        message = f"🚨 SOS Triggered ({source})!\n📍 Location: {location}"
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT
        )
        print("✅ SOS Sent Successfully!")
    except Exception as e:
        print(f"❌ Failed to send SOS: {e}")
