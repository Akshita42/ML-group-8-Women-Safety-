from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty
from kivy.app import App
from kivy.lang import Builder

Builder.load_file("kv/mode.kv")

class ModeScreen(Screen):
    selected_mode = StringProperty("")

    def set_mode(self, mode):
        self.selected_mode = mode
        App.get_running_app().selected_mode = mode  # 👈 Save to MainApp
        print(f"Selected mode: {mode}")

    def go_to_dashboard(self):
        if self.selected_mode:
            self.manager.current = 'dashboard'
        else:
            print("Please select a mode.")
