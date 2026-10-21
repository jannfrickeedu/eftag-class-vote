import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from PIL import Image

class Option:
    def __init__(self, name):
        self.name = name
        self.votes = 0

    def __str__(self) -> str:
        return f"{self.name}: {self.votes}"

class Vote:
    def __init__(self, image_path: Path, option_a: Option, option_b: Option):
        self.option_a = option_a
        self.option_b = option_b
        self.people_guesses = self.get_people(image_path)
        self.crops = self.get_crops()
        self.votes = self.get_votes(self.crops)


    def get_people(self, image_path):
        base_options = python.BaseOptions(model_asset_path='resources/model/efficientdet_lite0.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                               score_threshold=0.3,
                                               category_allowlist=["person"])
        detector = vision.ObjectDetector.create_from_options(options)

        image = mp.Image.create_from_file(str(image_path))

        detection_result = detector.detect(image)

        return [r.bounding_box for r in detection_result.detections]

    def get_votes(self, crops):
        # GestureRecognizer-Objekt erstellen basieren auf heruntergeladenem Modell
        base_gesture_options = python.BaseOptions(model_asset_path='resources/model/gesture_recognizer.task')
        gesture_options = vision.GestureRecognizerOptions(base_options=base_gesture_options)
        gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

        # Liste von lokalen Bild-Dateien
        for crop in crops:
            crop.convert('RGB')
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop))
            # Bild durch Recognizer schicken
            recognition_result = gesture_recognizer.recognize(mp_image)

            # Top-Geste der Erkennung auslesen
            top_gesture = recognition_result.gestures[0][0]

            if top_gesture.category_name == "Thumb_Up":
                self.option_a.votes += 1
            elif top_gesture.category_name == "Thumb_Down":
                self.option_b.votes += 1
            else:
                print(f"did not recognize category_name: {top_gesture.category_name}")


    def get_crops(self):
        im = Image.open(r'resources/images/class.jpeg')
        images =[]
        for people in self.people_guesses:

            tl_x = people.origin_x - (0.1*people.width)
            tl_y = people.origin_y - (0.1*people.height)
            br_x = people.origin_x + people.width + (0.1*people.width)
            br_y = people.origin_y + people.height + (0.1*people.height)

            cropped_image = im.crop((tl_x, tl_y, br_x, br_y))
            images.append(cropped_image)

        return images

if __name__ == "__main__":
    a = Option("Pizza yes")
    b = Option("Pizza no")
    poll = Vote("resources/images/class.jpeg", a, b)
    print(poll.option_a, poll.option_b)