from kivy.uix.screenmanager import Screen
import re

class SignUpScreen(Screen):
    def validate_signup(self):
        name = self.ids.name_input.text.strip()
        email = self.ids.email_input.text.strip()
        phone = self.ids.phone_input.text.strip()
        password = self.ids.password_input.text.strip()
        confirm = self.ids.confirm_input.text.strip()

        if not all([name, email, phone, password, confirm]):
            self.ids.error_label.text = "Please fill all fields."
            return

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            self.ids.error_label.text = "Invalid email format."
            return

        if len(phone) != 10 or not phone.isdigit():
            self.ids.error_label.text = "Enter valid 10-digit phone."
            return

        if len(password) < 8 or not re.search(r"[0-9]", password) or not re.search(r"[!@#$%^&*]", password):
            self.ids.error_label.text = "Weak password. Must have number & symbol."
            return

        if password != confirm:
            self.ids.error_label.text = "Passwords do not match."
            return

        # Signup successful
        self.ids.error_label.text = ""
        print("✅ Signup successful.")
        self.manager.current = 'emergency'  # Go to emergency contact screen
