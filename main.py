from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screens.welcome_screen import WelcomeScreen
from screens.login_screen import LoginScreen
from screens.signup_screen import SignUpScreen
from screens.emergency_screen import EmergencyContactScreen
from screens.mode_screen import ModeScreen
from screens.dashboard_screen import DashboardScreen
from kivy.lang import Builder
from screens.how_it_works_screen import HowItWorksScreen




Builder.load_file("kv/signup.kv")
Builder.load_file("kv/dashboard.kv")
Builder.load_file("kv/emergency.kv")
Builder.load_file("kv/how_it_works.kv")


class MainApp(App):
    selected_mode = "auto"  # 👈 Default value

    def build(self):
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcome'))
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(SignUpScreen(name='signup'))
        sm.add_widget(EmergencyContactScreen(name='emergency'))
        sm.add_widget(HowItWorksScreen(name='how_it_works'))
        sm.add_widget(DashboardScreen(name='dashboard'))
  # 👈 Start from welcome screen

        return sm


if __name__ == '__main__':
    MainApp().run()
