from kivy.app import App
from kivy.uix.widget import Widget


class CameraWidget(Widget):
    pass


class AuthWidget(Widget):
    pass


class AuthApp(App):
    def build(self):
        return AuthWidget()


if __name__ == '__main__':
    AuthApp().run()
