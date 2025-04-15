
ML-group-8-Women-Safety  
Women Safety SOS Wake Word Detection  

📌 About the Project  
This project is an SOS Alert System that listens for a custom wake word ("Help Me") and sends an emergency SMS alert to a predefined contact using Twilio.  

It is built using:  

- Porcupine for wake word detection  
- Twilio API for sending SMS alerts  
- Python for implementation  
- Librosa and machine learning for distress detection  
- CNN + LSTM for shake-based SOS detection  

✨ Features:  

- Wake Word Activation – Detects the phrase "Help Me" using Porcupine.  
- Emergency SMS Alert – Sends an SOS message to a predefined emergency contact.  
- Distress Detection – Uses Librosa with a SVM model trained on the RAVDESS dataset.  
- Shake Detection – Uses a CNN + LSTM model for smarter distress detection.  
- Environment Variables for Security – Stores sensitive credentials in a `.env` file.  
- Customizable – Can be extended with GPS tracking, voice authentication, etc.  

🛠 Technologies Used:  

- Python  
- Porcupine (Picovoice)  
- Twilio API  
- Librosa  
- pyaudio  
- dotenv  
- Scikit-learn (Random Forest for distress detection)  
- TensorFlow (CNN + LSTM for shake detection)  


How It Works:  

- The script listens for the wake word using the microphone.  
- If the wake word "Help Me" is detected, the script triggers an SOS alert.  
- The distress detection model analyzes the audio using MFCC features to identify distress.  
- The shake detection model detects rapid movements for silent distress activation.  
- An SMS alert is sent to the predefined emergency contact.  

Future Improvements:  

- GPS Location Sharing – Send location along with SOS.  
- Offline Support – Work without an internet connection.  
- AI-Based Voice Analysis – More advanced distress detection models.  
