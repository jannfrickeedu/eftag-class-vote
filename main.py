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
        self.crops = self.get_crops(self.people_guesses)
        self.votes = self.get_votes(self.crops)


    def get_people(self, image_path):
        base_options = python.BaseOptions(model_asset_path='resources/model/efficientdet_lite0.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                               score_threshold=0.3,
                                               category_allowlist=["person"])
        detector = vision.ObjectDetector.create_from_options(options)

        image = mp.Image.create_from_file(str(image_path))

        detection_result = detector.detect(image)

        return [r.bbox for r in detection_result.bounding_box]

    def get_results(self):
        print(self.option_a, self.option_b)

    def get_crops(self):
        im = image.open(r'resources/images/class.jpeg')
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
    Vote("resources/class.jpeg")
    print(Vote.get_results())