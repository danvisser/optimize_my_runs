import pytest

import numpy as np

from runna_modelling.scheduler import N_DAYS_IN_WEEK, Scheduler
from runna_modelling.workouts import Workout


class TestScheduler:

    ################################
    # Test get_fatigue_for_workout #
    ################################
    def test_get_fatigue_for_workout(self) -> None:
        
        fatigue_mapping = {
            Workout.EASY_RUN: 0.7,
            Workout.HARD_RUN: 1.4,
            Workout.LONG_RUN: 7.0,
            Workout.REST: 0.0,
        }
        
        scheduler = Scheduler(fatigue_mapping=fatigue_mapping)

        easy_run_fatigue = scheduler.get_fatigue_for_workout(Workout.EASY_RUN)
        np.testing.assert_array_equal(easy_run_fatigue, np.asarray([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))

        hard_run_fatigue = scheduler.get_fatigue_for_workout(Workout.HARD_RUN)
        np.testing.assert_array_equal(hard_run_fatigue, np.asarray([1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]))

        long_run_fatigue = scheduler.get_fatigue_for_workout(Workout.LONG_RUN)
        np.testing.assert_array_equal(long_run_fatigue, np.asarray([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))

        rest_fatigue = scheduler.get_fatigue_for_workout(Workout.REST)
        np.testing.assert_array_equal(rest_fatigue, np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    ##############
    # Test score #
    ##############
    def test_score_all_rests(self) -> None:
        scheduler = Scheduler(fatigue_mapping={Workout.REST: 0.0})
        assert scheduler.score([[Workout.REST] * N_DAYS_IN_WEEK]) == 0.0
    
    def test_score_peak_long_runs(self) -> None:
        scheduler = Scheduler(fatigue_mapping={Workout.LONG_RUN: 5.0, Workout.REST: 0.0})
        assert scheduler.score([[Workout.LONG_RUN] + [Workout.REST] * (N_DAYS_IN_WEEK-1)]) == 5.0
    
    def test_score_combined_workouts(self) -> None:
        scheduler = Scheduler(fatigue_mapping={Workout.LONG_RUN: 7.0, Workout.EASY_RUN: 1.4, Workout.REST: 0.0})
        assert scheduler.score([
            [Workout.REST] * (N_DAYS_IN_WEEK-1) + [Workout.LONG_RUN], 
            [Workout.EASY_RUN] + [Workout.REST] * (N_DAYS_IN_WEEK-1), 
        ]) == 7.4

    ###########################
    # Test schedule_all_rests #
    ###########################
    @pytest.mark.parametrize(
        "workout",
        [Workout.REST, Workout.HARD_RUN, Workout.EASY_RUN, Workout.LONG_RUN],
    )
    def test_schedule_all_workouts_the_same(self, workout: Workout) -> None:
        scheduler = Scheduler()
        n_weeks = 3
        unscheduled_workouts = [[workout] * N_DAYS_IN_WEEK] * n_weeks
        scheduled_workouts = scheduler.schedule(unscheduled_workouts)
        assert scheduled_workouts == [[workout] * N_DAYS_IN_WEEK] * n_weeks
