class Vote(image):
    def __init__(self, image):
        self.people_guesses = self.get_people(image)
        self.crops = self.get_crops(self.people_guesses)
        self.votes = self.get_votes(self.crops)

    def get_results(self):
        pass


if __name__ == "__main__":
    Vote("resources/class.jpeg")
    print(Vote.get_results())