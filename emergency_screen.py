from kivy.uix.screenmanager import Screen
from kivy.properties import ListProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.lang import Builder

Builder.load_file("kv/emergency.kv")

class EmergencyContactScreen(Screen):
    contacts = ListProperty([])

    def add_contact(self):
        name = self.ids.name_input.text.strip()
        phone = self.ids.phone_input.text.strip()

        if name and phone:
            self.contacts.append((name, phone))
            self.ids.name_input.text = ""
            self.ids.phone_input.text = ""
            self.display_contact(name, phone)
        else:
            self.show_popup("Please enter both name and phone number.")

    def display_contact(self, name, phone):
        contact_box = BoxLayout(size_hint_y=None, height=30, spacing=10)
        contact_box.add_widget(Label(text=name, color=(0, 0, 0, 1)))
        contact_box.add_widget(Label(text=phone, color=(0, 0, 0, 1)))

        delete_btn = Button(text="❌", size_hint_x=None, width=30, background_color=(0.8, 0.2, 0.3, 1))
        delete_btn.bind(on_release=lambda x: self.remove_contact(contact_box, (name, phone)))
        contact_box.add_widget(delete_btn)

        self.ids.contacts_container.add_widget(contact_box)

    def remove_contact(self, widget, contact):
        if contact in self.contacts:
            self.contacts.remove(contact)
        self.ids.contacts_container.remove_widget(widget)

    def save_contacts(self):
        if not self.contacts:
            self.show_popup("Please add at least one contact before continuing.")
        else:
            print("Saved contacts:", self.contacts)
            self.manager.current = 'how_it_works'

    def show_popup(self, message):
        popup = Popup(title="Info", content=Label(text=message),
                      size_hint=(None, None), size=(300, 200))
        popup.open()
