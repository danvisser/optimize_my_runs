import os
import json
from runna_modelling.scheduler import N_DAYS_IN_WEEK, Scheduler
from runna_modelling.workouts import Workout

DATA_DIR = "evaluation_data"

def load_workouts(file_path: str) -> list[list[Workout]]:
    with open(file_path, "r") as f:
        return [[Workout(w) for w in ws.split(",")] for ws in f.read().split("\n") if ws]

def get_results_for_scheduler(scheduler: Scheduler, evaluation_file_names) -> dict[int, float]:
    results = {}
    for file_name in evaluation_file_names:
        workouts = load_workouts(f"{DATA_DIR}/{file_name}")
        scheduled_workouts = scheduler.schedule(workouts)
        score = scheduler.score(scheduled_workouts)
        evaluation_id =  int(file_name.split("_")[1].split(".")[0])
        results[evaluation_id] = score
    return results

def load_benchmark_results() -> dict[int, float]:
    with open(f"{DATA_DIR}/benchmark_results.json", "r") as f:
        return {int(k): v for k, v in json.load(f).items()}

def main():

    # define evaluation files
    evaluation_files = [f for f in os.listdir(DATA_DIR) if f.startswith("workouts_")]

    # get current results
    scheduler = Scheduler()
    results = get_results_for_scheduler(scheduler, evaluation_files)
    
    # look at differences from benchmark
    benchmark_results = load_benchmark_results()
    changes = {}
    for plan_id, score in results.items():
        benchmark_score = benchmark_results[plan_id]
        changes[plan_id] = score - benchmark_score
    
    print("Results:")
    print("---")
    print(f"- Mean change (negative is good): {sum(changes.values()) / len(changes)}")
    print("---")
    print(f"- Number decreased (good):     {len([v for v in changes.values() if v < 0])}")
    print(f"- Number unchanged (neutral):  {len([v for v in changes.values() if v == 0])}")
    print(f"- Number increased (bad):      {len([v for v in changes.values() if v > 0])}")



if __name__ == '__main__':
    main()


