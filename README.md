# modelling-take-home-1

## Setup
With Python 3.11
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Task 1 [Primary]
Implement a scheduling algorithm to schedule runs in each week of a training plan.

### Workout Types
There are three types of workout:
1. Easy runs - these are the easiest run of the week, and are the least fatiguing. They involve running for a set distance at an easy pace.
2. Hard runs - these are harder than easy runs, and are more fatiguing. They involve running at different paces, some fast and some slow. 
3. Long runs - these are the most fatiguing. They involve running for a long distance (much longer than easy or hard runs).

There are also rests, which occupy non-run days.

These are all defined in `runna_modelling.workouts`.

### The Scheduler
The scheduler class has been created in `runna_modelling.scheduler` and there is a `TODO` where you can write your code.

The optimum schedule will minimise the maximum fatigue across the whole schedule (the `score` method is already implemented!).

Note: workouts cannot be moved between weeks!

Example usage:
```python
from runna_modelling.scheduler import Scheduler
from runna_modelling.workouts import Workout

# Create a scheduler
scheduler = Scheduler()

# Get the fatigue for a run - the fatigue of a run will start at a maximum value
# (this max depends on the run type)
# on the day of the run, and decay over the course of the next 7 days
long_run_fatigue = scheduler.get_fatigue_for_workout(Workout.LONG_RUN)
print(f"Example {long_run_fatigue=}\n")

# Note: the fatigue for a rest is always 0
rest_fatigue = scheduler.get_fatigue_for_workout(Workout.REST)
print(f"Example {rest_fatigue=}\n")

# For all the workouts in a schedule, the fatigues can be combined (i.e. summed):
workouts = [
    [Workout.LONG_RUN, Workout.HARD_RUN, Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.REST, Workout.REST],
    [Workout.LONG_RUN, Workout.HARD_RUN, Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.REST, Workout.REST],
    [Workout.LONG_RUN, Workout.HARD_RUN, Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.REST, Workout.REST],
]
all_fatigues = scheduler.get_combined_fatigues(workouts)
print("Example combined fatigues 1:")
print(all_fatigues)

# The 'score' method can be used to determine 'how good' a schedule is
# The lower the score, the better the schedule
# Note: this is the maximum fatigue on any given day
score = scheduler.score(workouts)
print(f"Example score 1: {score}\n")

# Some schedules with the same workouts in will have better scores
# For example, this is a better schedule, as the runs are spread out 
# more and the peak fatigue is now 2.128, instead of 2.586
workouts = [
    [Workout.LONG_RUN, Workout.REST, Workout.EASY_RUN, Workout.REST, Workout.HARD_RUN, Workout.REST, Workout.REST],
    [Workout.LONG_RUN, Workout.REST, Workout.EASY_RUN, Workout.REST, Workout.HARD_RUN, Workout.REST, Workout.REST],
    [Workout.LONG_RUN, Workout.REST, Workout.EASY_RUN, Workout.REST, Workout.HARD_RUN, Workout.REST, Workout.REST],
]
all_fatigues = scheduler.get_combined_fatigues(workouts)
print("Example combined fatigues 2:")
print(all_fatigues)
score = scheduler.score(workouts)
print(f"Example score 2: {score}\n")


# To schedule the workouts, the 'schedule' method can be called
# For this task, you need to add code to the schedule method
scheduled_workouts = scheduler.schedule(workouts)
print("Example scheduled workouts:")
for week_index in range(len(scheduled_workouts)):
    print(f"{week_index=}: {[workout.value for workout in scheduled_workouts[week_index]]}")
```

#### Summary
So to put simply, the task is to add code to this repo, so that that the output of `Scheduler.schedule(...)` is the schedule which has the minimum `score` associated with it!

### Tests
There are a few sample tests in `tests.unit.runna_modelling.test_scheduler`, these should still pass after you have implemented your solution.
To run them:
```bash
pytest tests
```

### Evaluation
There is an evaluation script, `evaluate.py`, which can be run to determine the performance of the scheduling algorithm. This compares the current implementation of the scheduler with the initial implementation (i.e. no scheduling logic).

I suggest you use this script to track your progress and understand how well your algorithm is performing.

To run the script:
```bash
python evaluate.py
```

## Task 2 [Extension]
Add another scheduler that calculates the fatigue in a new way:

For `0 <= n <= 6`:
- `fatigue`(`n`) = `peakFatigue` * `e`^(-`λn`)

For `n < 0`, `n > 7`:
- `fatigue`(`n`) = 0

Where `n` is the number of days since the workout.

I.e. max on the day of the workout, and decays over the course of 7 days, then 0.

There is no need to change anything other than how the fatigue is calculated - purely interested to see the approach you would take to implementation of this!

Note: we want to be respectful of your time - if you don't have enough to do this, please think about how you would do it and we can discuss in the tech interview if appropriate!

## FAQs/Tips/Guidance
- If you think appropriate, please feel free to:
    - Add any extra modules, classes or functions - anything goes!
    - Write any tests
    - Use any 3rd party packages
- We're not expecting a perfect solution! Document your thought process, add comments in the code to explain what's happening and see how you get on.
- There may be more than one optimal solution for any given input
- If there are any questions or anything is unclear, please drop a message to harry@runna.com, and I'll be happy to chat!
- It's up to you how long you spend on this - there are no min/max time constraints. However, we'd like to be respectful of your time so if there's more you'd like to add having already invested a reasonable amount of time, please feel free to write what you would do given more time.
- Good luck!