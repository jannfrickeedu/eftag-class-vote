import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import cache
from pathlib import Path
from PIL import Image

MODEL_DIR = Path(__file__).parent / "resources" / "model"
MARGIN = 0.2
SCORE_THRESHOLD = 0.25


class Option:
    def __init__(self, name):
        self.name = name
        self.votes = 0

    def __str__(self) -> str:
        return f"{self.name}: {self.votes}"


def to_mp_image(image: Image.Image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image.convert("RGB")))


@cache
def get_detector(category_list=("person",)):
    base_options = python.BaseOptions(model_asset_path=str(MODEL_DIR / "efficientdet_lite0.tflite"))
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=SCORE_THRESHOLD,
                                           category_allowlist=list(category_list))
    return vision.ObjectDetector.create_from_options(options)


@cache
def get_gesture_recognizer():
    base_gesture_options = python.BaseOptions(model_asset_path=str(MODEL_DIR / "gesture_recognizer.task"))
    gesture_options = vision.GestureRecognizerOptions(base_options=base_gesture_options)
    return vision.GestureRecognizer.create_from_options(gesture_options)


def get_bbox_from_image(image: Image.Image, category_list=("person",)):
    detection_result = get_detector(tuple(category_list)).detect(to_mp_image(image))
    return [r.bounding_box for r in detection_result.detections]


class Vote:
    def __init__(self, image: Image.Image, option_a: Option, option_b: Option, margin=MARGIN, score_threshold=SCORE_THRESHOLD):
        self.option_a = option_a
        self.option_b = option_b
        # list of (bounding_box, gesture_name or None) per detected person
        self.details = []

        (self.option_a.votes, self.option_b.votes) = self.evaluate(image)

    @classmethod
    def from_path(cls, image_path: Path, option_a: Option, option_b: Option):
        return cls(Image.open(image_path), option_a, option_b)

    def evaluate(self, image: Image.Image):
        people = get_bbox_from_image(image, category_list=["person"])
        crops = self.get_crops(image, people)
        return self.get_votes(people, crops)

    def get_votes(self, people, crops):
        gesture_recognizer = get_gesture_recognizer()
        pos = 0
        neg = 0
        other = 0

        for person, crop in zip(people, crops):
            recognition_result = gesture_recognizer.recognize(to_mp_image(crop))

            if not recognition_result.gestures:
                self.details.append((person, None))
                continue

            top_gesture = recognition_result.gestures[0][0]
            self.details.append((person, top_gesture.category_name))

            if top_gesture.category_name == "Thumb_Up":
                pos += 1
            elif top_gesture.category_name == "Thumb_Down":
                neg += 1
            else:
                print(f"did not recognize category_name: {top_gesture.category_name}")
                other += 1


        return (pos, neg, other)

    def get_crops(self, image: Image.Image, people):
        images = []
        for person in people:

            tl_x = person.origin_x - (MARGIN*person.width)
            tl_y = person.origin_y - (MARGIN*person.height)
            br_x = person.origin_x + person.width + (MARGIN*person.width)
            br_y = person.origin_y + person.height + (MARGIN*person.height)

            cropped_image = image.crop((tl_x, tl_y, br_x, br_y))
            images.append(cropped_image)

        return images


if __name__ == "__main__":
    a = Option("positive")
    b = Option("negative")
    training_dir = Path("resources/images/class-data-set")
    for file in training_dir.iterdir():
        poll = Vote.from_path(file, a, b)
        print(poll.option_a, poll.option_b)
