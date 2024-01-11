from datetime import datetime
from pathlib import Path

from PIL import Image
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivymd.uix.screen import MDScreen

from components.db.client import UserService
from components.detection.retinaface import detect
from components.extraction.arcface import extract
from components.trasnformations import det2ext


class CameraWidget(Camera):

    def __init__(self, *args, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)

        self.detections = None
        self.or_img = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
        Path(f'./log/imgs/{self.run_id}/').mkdir(0o733, True, True)

        detect.init()

    def on_tex(self, camera):
        texture = camera.texture
        size = texture.size
        frame = texture.pixels

        self.or_img = Image.frombytes(mode='RGBA', size=size, data=frame).convert('RGB')

        img = self.or_img.copy()
        self.detections = detect.run(img, run_id=self.run_id, return_image=True)

        detections_texture: Texture = Texture.create(size=size)
        detections_texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        detections_texture.flip_vertical()

        self.texture = texture = detections_texture
        self.texture_size = list(texture.size)
        self.canvas.ask_update()

    def register(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        for i, detection in enumerate(self.detections):
            face_img = det2ext.align_face(self.or_img, detection[5:])
            vector = extract.run(face_img)[0]
            UserService.register_user(username="Test User", vector=vector)

    def login(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        for i, detection in enumerate(self.detections):
            face_img = det2ext.align_face(self.or_img, detection[5:])
            vector = extract.run(face_img)[0]
            has_access = UserService.has_access(username="Test User", vector=vector)
            print(f"Test User has access: {has_access}")



class AuthScreen(MDScreen):
    pass


class AuthApp(App):
    def build(self):
        return AuthScreen()


if __name__ == '__main__':
    AuthApp().run()
