import pandas as pd
from pathlib import Path
from main import Vote, Option
from scipy.optimize import brute

names = ["pos", "neg", "other"]
correct = pd.read_csv('resources/images/class-data-set/correct.csv', names=names)
def loss_function(p):
    MARGIN, SCORE_THRESHOLD = p
    print(p)
    results = []
    training_dir = Path("resources/images/class-data-set")
    for file in training_dir.iterdir():
        if file.name == "correct.csv":
            continue
        a = Option("positive")
        b = Option("negative")
        poll = Vote.from_path(file, a, b, margin=MARGIN, score_threshold=SCORE_THRESHOLD)
        results.append({"pos": poll.option_a.votes, "neg": poll.option_b.votes, "other": poll.other_votes})

    pandas_results = pd.DataFrame(results)
    diff = correct - pandas_results
    mean_abs_dev = diff.abs().to_numpy().mean()
    return mean_abs_dev

# Initial guess for the parameters
initial_guess = [0.2, 0.25]

best = brute(loss_function, ranges=[(0.1, 0.5), (0.1, 0.5)], Ns=10, finish=None)
print("best vvalues are:")
print(best)
