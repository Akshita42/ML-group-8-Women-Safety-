from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.app import App

Builder.load_file("kv/login.kv")  # or "login.kv" if not in subfolder


class LoginScreen(Screen):
    def validate_login(self):
        email = self.ids.email_input.text.strip()
        password = self.ids.password_input.text.strip()

        if not email or not password:
            self.ids.error_label.text = "Please enter both fields."
            return

        if email == "test@example.com" and password == "1234":
            self.ids.error_label.text = ""
            print("Login successful")
            App.get_running_app().root.current = 'emergency'
