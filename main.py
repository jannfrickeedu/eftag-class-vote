import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from PIL import Image
import os

class Option:
    def __init__(self, name):
        self.name = name
        self.votes = 0

    def __str__(self) -> str:
        return f"{self.name}: {self.votes}"


def get_bbox_from_image(image_path, category_list=["person"]):
    base_options = python.BaseOptions(model_asset_path='resources/model/efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.3,
                                           category_allowlist=category_list)
    detector = vision.ObjectDetector.create_from_options(options)

    image = mp.Image.create_from_file(str(image_path))

    detection_result = detector.detect(image)

    return [r.bounding_box for r in detection_result.detections]


class Vote:
    def __init__(self, image_path: Path, option_a: Option, option_b: Option):
        self.option_a = option_a
        self.option_b = option_b

        (self.option_a.votes, self.option_b.votes) = self.evaluate(image_path)

    def evaluate(self, image_path: Path):
        people = get_bbox_from_image(image_path, category_list=["person"])
        crops = self.get_crops(people)
        return self.get_votes(crops)

    def get_votes(self, crops):
        base_gesture_options = python.BaseOptions(model_asset_path='resources/model/gesture_recognizer.task')
        gesture_options = vision.GestureRecognizerOptions(base_options=base_gesture_options)
        gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
        pos = 0
        neg = 0

        for crop in crops:
            crop.convert('RGB')
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop))
            recognition_result = gesture_recognizer.recognize(mp_image)

            top_gesture = recognition_result.gestures[0][0]

            if top_gesture.category_name == "Thumb_Up":
                pos += 1
            elif top_gesture.category_name == "Thumb_Down":
                neg += 1
            else:
                print(f"did not recognize category_name: {top_gesture.category_name}")

        return (pos, neg)


    def get_crops(self, people):
        im = Image.open(r'resources/images/class-data-set/class0.jpeg')
        images = []
        for person in people:

            tl_x = person.origin_x - (0.1*person.width)
            tl_y = person.origin_y - (0.1*person.height)
            br_x = person.origin_x + person.width + (0.1*person.width)
            br_y = person.origin_y + person.height + (0.1*person.height)

            cropped_image = im.crop((tl_x, tl_y, br_x, br_y))
            images.append(cropped_image)

        return images

if __name__ == "__main__":
    a = Option("positive")
    b = Option("negative")
    training_dir = Path("resources/images/class-data-set")
    for file in training_dir.iterdir():
        poll = Vote(file, a, b)
        print(poll.option_a, poll.option_b)