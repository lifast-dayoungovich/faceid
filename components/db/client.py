from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class DataAccessor:

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(mode=0o744, parents=True, exist_ok=True)

        self._vectors_fname = self._data_dir / "vectors.dat"
        self._ids_fname = self._data_dir / "ids.dat"
        self._vectors = self._load_data(self._vectors_fname)
        self._ids = self._load_data(self._ids_fname)
        print(f"Current record count: {self._vectors.shape[0] if self._vectors is not None else 0}")

        self._search_model = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=2)
        self.update_search_engine()

    def _load_data(self, filename: Path) -> np.ndarray:
        if filename.is_file():
            return joblib.load(filename)

    def save_data(self):
        print(f"Saving. Current record count: {self._vectors.shape[0]}")
        joblib.dump(self._vectors, self._vectors_fname)
        joblib.dump(self._ids, self._ids_fname)

    def update_search_engine(self):
        if self._vectors is not None:
            self._search_model.fit(self._vectors)

    def add_record(self, record_id: str, vector: np.ndarray):
        if self._vectors is not None:
            self._vectors = np.concatenate([self._vectors, vector[None]], axis=0)
            self._ids = np.concatenate([self._ids, np.array([[record_id]])], axis=0)
        else:
            self._vectors = vector[None]
            self._ids = np.array([[record_id]])

    def delete_record_by_id(self, record_id: str):
        self._ids = self._ids[np.where(self._ids != record_id)]
        self._vectors = self._vectors[np.where(self._ids != record_id)]

    def delete_record_by_vector(self, record_vector: np.ndarray):
        self._ids = self._ids[np.where(self._vectors != record_vector)]
        self._vectors = self._vectors[np.where(self._vectors != record_vector)]

    def get_id_by_sim_vector(self, vector: np.ndarray, sim_th: float) -> tuple[str, float] | tuple[None, None]:
        distances, indices = self._search_model.kneighbors(vector[None], n_neighbors=1)

        filtered = distances < 1 - sim_th

        top_indices = indices[filtered]
        top_distances = distances[filtered]
        if top_indices.size > 0:
            return self._ids[top_indices[0]][0], top_distances[0]
        else:
            return None, None


class DBClient:

    def __init__(self):
        self.data_accessor = DataAccessor("./data/")

    def __enter__(self):
        return self.data_accessor

    def __exit__(self, *args, **kwargs):
        self.data_accessor.save_data()
        self.data_accessor.update_search_engine()


class UserService:
    client = DBClient()

    @classmethod
    def register_user(cls, username: str, vector: np.ndarray):
        with cls.client as data_accessor:
            data_accessor.add_record(username, vector)

    @classmethod
    def has_access(cls, username: str, vector: np.ndarray, sim_th: float = 0.6):
        with cls.client as data_accessor:
            top_username, distance = data_accessor.get_id_by_sim_vector(vector, sim_th)
            return top_username == username


def register_pipeline(image_path, name, register_one=False):
    from datetime import datetime
    from PIL import Image
    from components.detection.retinaface import detect
    from components.trasnformations import det2ext
    from components.extraction.arcface import extract

    run_id = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
    Path(f'./log/imgs/{run_id}/').mkdir(0o733, True, True)

    img = Image.open(image_path)

    detections = detect.run(img, run_id=run_id)
    for i, detection in enumerate(detections):
        face_img = det2ext.align_face(img, detection[5:])
        vector = extract.run(face_img)[0]
        UserService.register_user(username=name if register_one else f"{name}_{i}", vector=vector)
        if register_one:
            break


def auth_pipeline(image_path, name, similarity_threshold):
    from datetime import datetime
    from PIL import Image
    from components.detection.retinaface import detect
    from components.trasnformations import det2ext
    from components.extraction.arcface import extract

    run_id = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
    Path(f'./log/imgs/{run_id}/').mkdir(0o733, True, True)

    img = Image.open(image_path)

    detections = detect.run(img, run_id=run_id)
    for i, detection in enumerate(detections):
        face_img = det2ext.align_face(img, detection[5:])
        vector = extract.run(face_img)
        return UserService.has_access(username=name, vector=vector, sim_th=similarity_threshold)


def test_register_zaluzhnyi():
    registration_img_path = '../detection/retinaface/curve/zaluzhnyi_register.jpg'
    register_pipeline(registration_img_path, "Valerii Zaluzhnyi", register_one=True)


def test_register_guys():
    registration_img_path = '../detection/retinaface/curve/test.jpg'
    register_pipeline(registration_img_path, "guy", register_one=False)


def test_auth_zaluzhnyi():
    registration_img_path = '../detection/retinaface/curve/zaluzhnyi_test1.jpg'
    authenticated = auth_pipeline(registration_img_path, "Valerii Zaluzhnyi", 0.65)
    print(f"{registration_img_path} is {authenticated}")

    registration_img_path = '../detection/retinaface/curve/zaluzhnyi_test2.jpg'
    authenticated = auth_pipeline(registration_img_path, "Valerii Zaluzhnyi", 0.65)
    print(f"{registration_img_path} is {authenticated}")


def test_auth_zelenskyi():
    registration_img_path = '../detection/retinaface/curve/zelenskyi_test1.jpg'
    authenticated = auth_pipeline(registration_img_path, "Zelenskyi", 0.65)
    print(f"{registration_img_path} is {authenticated}")


def test_register_zelenskyi():
    registration_img_path = '../detection/retinaface/curve/zelenskyi_register.jpg'
    register_pipeline(registration_img_path, "Zelenskyi", register_one=True)


if __name__ == '__main__':
    # test_register_zelenskyi()
    # test_register_zaluzhnyi()
    # test_register_guys()
    test_auth_zaluzhnyi()
    test_auth_zelenskyi()
