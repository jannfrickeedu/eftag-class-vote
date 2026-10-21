import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

class Vote:
    def __init__(self, image_path):
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
        pass


if __name__ == "__main__":
    Vote("resources/class.jpeg")
    print(Vote.get_results())