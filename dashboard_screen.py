from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from wake_word.wake_listener import start_wake_word_detection
from utils.sos_trigger import trigger_sos as send_sos_alert
from datetime import datetime

class DashboardScreen(Screen):
    sos_triggered = False
    cancel_timer = None
    manual_cancel_timer = None

    def on_enter(self, *args):
        self.sos_logs = []

    def set_mode(self, mode):
        App.get_running_app().selected_mode = mode
        self.ids.sos_status_label.text = ""  # Changed from status_label to sos_status_label
        self.ids.auto_controls.opacity = 0
        self.ids.manual_controls.opacity = 0

        if mode == "auto":
            self.ids.sos_status_label.text = "🎤 Listening for wake word..."
            self.ids.auto_controls.opacity = 1
            self.ids.auto_cancel_btn.text = "Cancel SOS"
            start_wake_word_detection(lambda: Clock.schedule_once(lambda dt: self.on_wake_word_detected()))

        elif mode == "manual":
            self.ids.sos_status_label.text = "📱 Manual Mode Active"
            self.ids.manual_controls.opacity = 1

    def on_wake_word_detected(self):
        if not self.sos_triggered:
            self.sos_triggered = True
            self.ids.sos_status_label.text = "🔊 Wake word detected! Sending SOS in 5 sec..."
            self.cancel_timer = Clock.schedule_once(self.send_sos, 5)

    def cancel_sos(self):
        if self.sos_triggered:
            if self.cancel_timer:
                self.cancel_timer.cancel()
            self.ids.sos_status_label.text = ""
            self.sos_triggered = False
            self.show_popup("❌ SOS Cancelled")

    def send_sos(self, *args):
        send_sos_alert()
        self.ids.sos_status_label.text = ""
        self.sos_triggered = False
        self.update_sos_log()
        self.show_popup("✅ SOS Sent!")

    def manual_sos(self):
        if not self.sos_triggered:
            self.sos_triggered = True
            self.ids.manual_sos_btn.text = "Cancel SOS"
            self.ids.sos_status_label.text = "Sending SOS in 5 sec..."
            self.manual_cancel_timer = Clock.schedule_once(self.send_manual_sos, 5)

    def send_manual_sos(self, *args):
        send_sos_alert()
        self.ids.sos_status_label.text = ""
        self.ids.manual_sos_btn.text = "Send SOS"
        self.sos_triggered = False
        self.update_sos_log()
        self.show_popup("✅ SOS Sent!")

    def test_sos(self):
        self.show_popup("✅ Test SOS Triggered (No real alert)")

    def show_popup(self, message):
        popup = Popup(
            title='Status',
            content=Label(text=message, color=(1, 0.2, 0.6, 1)),
            size_hint=(None, None), size=(300, 150)
        )
        popup.open()

    def update_sos_log(self):
        time_now = datetime.now().strftime("%I:%M %p")
        self.sos_logs.insert(0, f"• SOS sent at {time_now}")
        self.ids.sos_log_label.text = "\n".join(self.sos_logs[:10])
