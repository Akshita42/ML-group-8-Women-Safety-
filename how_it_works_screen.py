from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

Builder.load_file("kv/how_it_works.kv")

class HowItWorksScreen(Screen):
    def go_to_dashboard(self):
        self.manager.current = "dashboard"
