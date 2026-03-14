from file_checks import classifiers_exist
from training import train_classifiers_csv
from evaluation import run_kfold_evaluation
from batch import batch_process_csv
import time

tic = time.perf_counter()

print("\n=== Hybrid Conformance Checker ===")

if not classifiers_exist():
    print("\nNo trained classifiers found. Training models for the first time...")
    run_kfold_evaluation(k=5)
    train_classifiers_csv()
else:
    print("\nClassifiers found. Skipping training.")

print("\nRunning batch conformance checking...")
batch_process_csv()

print("\nDone.")

toc = time.perf_counter()
print(f"Elapsed: {toc - tic:.6f} seconds")
