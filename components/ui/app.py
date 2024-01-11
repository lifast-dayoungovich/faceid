from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget

from components.db.client import UserService
from components.detection.retinaface import detect
from components.extraction.arcface import extract
from components.trasnformations import det2ext


def _circle2xy(x_c, y_c, r):
    return (x_c - r, y_c - r), (x_c + r, y_c + r)

def draw_detections(img, dets, usernames):
    draw = ImageDraw.Draw(img)
    for username, b in zip(usernames, dets):

        text = f"{username} - {b[4]:.4f}"
        draw.rectangle(b[:4], outline='#0000ea', width=2)

        cx = b[0]
        cy = b[1] + 12

        draw.text((cx, cy), text, '#000000', ImageFont.load_default(16), align="left")

        # Landmarks
        draw.ellipse(_circle2xy(b[5], b[6], 3), '#ff0000', width=3)
        draw.ellipse(_circle2xy(b[7], b[8], 3), '#00ffff', width=3)
        draw.ellipse(_circle2xy(b[9], b[10], 3), '#ff00ff', width=3)
        draw.ellipse(_circle2xy(b[11], b[12], 3), '#00ff00', width=3)
        draw.ellipse(_circle2xy(b[13], b[14], 3), '#ffff00', width=3)

        if img:
            return img

class CameraWidget(Camera):

    def __init__(self, *args, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)

        self.users = None
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

        if not self.users:
            self.users = ['Unknown']*len(self.detections)

        if self.detections is not None and len(self.detections) > 0:
            img = draw_detections(img, self.detections, self.users)

        detections_texture: Texture = Texture.create(size=size)
        detections_texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        detections_texture.flip_vertical()

        self.texture = texture = detections_texture
        self.texture_size = list(texture.size)
        self.canvas.ask_update()

    def register(self, username='Alina'):
        self.users = []
        for i, detection in enumerate(self.detections):
            face_img = det2ext.align_face(self.or_img, detection[5:])
            vector = extract.run(face_img)[0]
            UserService.register_user(username=username, vector=vector)
            self.users.append(username)
            break

    def login(self, username='Alina'):
        self.users = []
        for i, detection in enumerate(self.detections):
            face_img = det2ext.align_face(self.or_img, detection[5:])
            vector = extract.run(face_img)[0]
            has_access = UserService.has_access(username=username, vector=vector)
            print(f"Test User has access: {has_access}")
            if has_access:
                self.users.append(username)



class AuthWidget(Widget):
    pass


class AuthApp(App):
    def build(self):
        return AuthWidget()


if __name__ == '__main__':
    AuthApp().run()
