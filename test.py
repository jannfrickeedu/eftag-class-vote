import pandas as pd
from pathlib import Path
from main import Vote, Option

names = ["pos", "neg", "other"]
correct = pd.read_csv('resources/images/class-data-set/correct.csv', names=names)

training_dir = Path("resources/images/class-data-set")
results = []
for file in training_dir.iterdir():
    if file.name == "correct.csv":
        continue
    a = Option("positive")
    b = Option("negative")
    poll = Vote.from_path(file, a, b)
    results.append({"pos": poll.option_a.votes, "neg": poll.option_b.votes, "other": poll.other_votes})

pandas_results = pd.DataFrame(results)
print(pandas_results)
    

diff = correct - pandas_results
mean_abs_dev = diff.abs().to_numpy().mean()
print(mean_abs_dev)