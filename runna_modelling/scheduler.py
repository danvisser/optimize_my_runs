from runna_modelling.workouts import Workout
from itertools import permutations, product

import numpy as np


N_DAYS_IN_WEEK = 7

DEFAULT_FATIGUE_MAPPING = {
    Workout.EASY_RUN: 0.8,
    Workout.HARD_RUN: 1.0,
    Workout.LONG_RUN: 1.3,
    Workout.REST: 0.0,
}


class Scheduler:

    def __init__(self, fatigue_mapping: dict[Workout, float] = DEFAULT_FATIGUE_MAPPING):
        """
        A class to schedule workouts.

        Args:
            fatigue_mapping (dict[Workout, float], optional): The fatigue mapping. 
                Maps each workout type to a peak fatigue value. 
                E.g. {
                    Workout.EASY_RUN: 0.8, 
                    Workout.HARD_RUN: 1.0, 
                    Workout.LONG_RUN: 1.3, 
                    Workout.REST: 0.0
                }.
                Defaults to DEFAULT_FATIGUE_MAPPING.
        """
        self.fatigue_mapping = fatigue_mapping
        # hash fatigue profiles for use in get_combined_fatigues
        self.fatigue_profile_mapping = {
            workout: self.get_fatigue_for_workout(workout) for workout in fatigue_mapping.keys()
            }
        # register optimizers for ensembling
        self._register_optimizers()

    def _register_optimizers(self):
        self.optimizer_registry = {
            "return_input_schedule": self.return_input_schedule,
            "brute_optimize_weeks_independently": self.brute_optimize_weeks_independently
            }

    def get_fatigue_for_workout(self, workout: Workout) -> np.ndarray:
        """
        Calculate the fatigue for a workout over the course of a week.
        
        The fatigue is highest on the day of the workout and decreases 
        linearly to 0 over the course of a week. Rounded to 3 d.p.

        The fatigue is calculated by...
            For 0 <= n <= 6:    fatigue(n) = peakFatigue * ((7 - n) / 7)
            For n < 0, n > 7:   fatigue(n) = 0

        Args:
            workout (Workout): The workout to calculate fatigue for.

        Returns:
            np.ndarray: The fatigue for the workout over the course of a week. Always 7 days long.
                The first element is the fatigue on the day of the workout, 
                the last element is the fatigue 6 days after the workout.
                E.g. [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] for a peak fatigue of 0.7.
        """
        peak_fatigue = self.fatigue_mapping[workout]
        return np.round(peak_fatigue * (N_DAYS_IN_WEEK - np.arange(N_DAYS_IN_WEEK)) / N_DAYS_IN_WEEK, 3)

    def get_combined_fatigues(self, workouts: list[list[Workout]]) -> np.ndarray:
        """
        Calculate the fatigue for a schedule of workouts.
        Calculates the contribution from each workout to the fatigue 
        on each day, and sums these contributions.

        Args:
            workouts (list[list[Workout]]): The scheduled workouts.
                Each element represents a week, and each element of that list
                is a workout for that day.

        Returns:
            np.ndarray: The fatigue for each day of the schedule.
        """
        n_weeks = len(workouts)
        all_fatigues = np.zeros(n_weeks * N_DAYS_IN_WEEK)
        for week_index, week in enumerate(workouts):
            for day_index, workout in enumerate(week):
                fatigue = self.fatigue_profile_mapping[workout]
                start_index = week_index * N_DAYS_IN_WEEK + day_index
                end_index = min(start_index + N_DAYS_IN_WEEK, all_fatigues.shape[0])
                all_fatigues[start_index: end_index] += fatigue[:end_index - start_index]
        return np.round(all_fatigues.reshape(n_weeks, N_DAYS_IN_WEEK), 3)

    def score(self, workouts: list[list[Workout]]) -> float:
        """
        Scores the quality of the schedule.
        The score is the maximum fatigue on any day.
        A lower score is better!

        Args:
            workouts (list[list[Workout]]): The scheduled workouts.
                Each element represents a week, and each element of that list 
                is a workout for that day.

        Returns:
            float: The score of the schedule.
        """
        return self.get_combined_fatigues(workouts).max()

    def schedule(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        """
        Schedule a list of workouts for each week independently to minimize weekly fatigue.

        Args:
            workouts (list[list[Workout]]): Each element is a list of un-scheduled workouts that will appear in that week.
                E.g. [
                    [Workout.LONG_RUN, Workout.HARD_RUN, Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.REST, Workout.REST],
                    [Workout.LONG_RUN, Workout.HARD_RUN, Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.REST, Workout.REST],
                ]

        Returns:
            list[list[Workout]]: Each element is a list of scheduled workouts for that week.
                E.g. [
                    [Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.HARD_RUN, Workout.REST, Workout.REST, Workout.LONG_RUN],
                    [Workout.EASY_RUN, Workout.REST, Workout.REST, Workout.HARD_RUN, Workout.REST, Workout.REST, Workout.LONG_RUN],
                ]
        """
        for week_index, week_workouts in enumerate(workouts):
            if len(week_workouts) != N_DAYS_IN_WEEK:
                raise ValueError(f"Invalid number of workouts for week {week_index=}")
            
        return self._pick_best_optimizer(workouts)
            

    def _pick_best_optimizer(self, workouts: list[list[Workout]]) -> callable:
        optimizer_results = {}

        for optimizer_name, optimizer in self.optimizer_registry.items():
            workout_scedule = optimizer(workouts)
            optimizer_results[optimizer_name] = {
                "score": self.score(workout_scedule),
                "workout_scedule": workout_scedule
            }

        best_optimizer = min(optimizer_results, key=lambda k: optimizer_results[k]["score"])
        return optimizer_results[best_optimizer]["workout_scedule"]


    def brute_optimize_weeks_independently(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        optimized_schedule = []

        for week_workouts in workouts:
            best_week_schedule = None
            best_week_score = float('inf')

            for permuted_week in permutations(week_workouts):
                current_score = self.score([list(permuted_week)])

                if current_score < best_week_score:
                    best_week_schedule = list(permuted_week)
                    best_week_score = current_score

            optimized_schedule.append(best_week_schedule)

        return optimized_schedule
    
    def return_input_schedule(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        return workouts
