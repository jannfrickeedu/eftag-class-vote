import pandas as pd
from pathlib import Path
from main import Vote, Option
from scipy.optimize import minimize

names = ["pos", "neg", "other"]
correct = pd.read_csv('resources/images/class-data-set/correct.csv', names=names)
def loss_function(p):
    MARGIN, SCORE_THRESHOLD = p
    results = []
    training_dir = Path("resources/images/class-data-set")
    for file in training_dir.iterdir():
        if file.name == "correct.csv":
            continue
        a = Option("positive")
        b = Option("negative")
        poll = Vote.from_path(file, a, b)
        results.append({"pos": poll.option_a.votes, "neg": poll.option_b.votes, "other": poll.other_votes})

    pandas_results = pd.DataFrame(results)
    diff = correct - pandas_results
    mean_abs_dev = diff.abs().to_numpy().mean()
    return mean_abs_dev

# Initial guess for the parameters
initial_guess = [0.2, 0.25]

# Minimize the loss function
result = minimize(loss_function, initial_guess, method='Nelder-Mead')

print("Optimal parameters:")
print(f"Margin: {result.x[0]}")
print(f"Score threshold: {result.x[1]}")
print(f"Optimal value: {result.fun}")