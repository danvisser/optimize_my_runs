from runna_modelling.workouts import Workout
from itertools import permutations, product
import random

import numpy as np


N_DAYS_IN_WEEK = 7

DEFAULT_FATIGUE_MAPPING = {
    Workout.EASY_RUN: 0.8,
    Workout.HARD_RUN: 1.0,
    Workout.LONG_RUN: 1.3,
    Workout.REST: 0.0,
}


class Scheduler:
    def __init__(self, fatigue_mapping: dict[Workout, float] = DEFAULT_FATIGUE_MAPPING, random_seed: int = 101):
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
        # hash fatigue profiles to prevent recalculation in get_combined_fatigues
        self.fatigue_profile_mapping = {
            workout: self.get_fatigue_for_workout(workout)
            for workout in fatigue_mapping.keys()
        }
        # random seed for reproducibility
        self.random_seed = random_seed
        # register optimizers for ensembling
        self._register_optimizers()

    def _register_optimizers(self):
        self.optimizer_registry = {
            # "return_input_schedule": self.return_input_schedule,
            # "brute_optimize_weeks_independently": self.brute_optimize_weeks_independently,
            # "simulated_annealing": self.simulated_annealing,
            "genetic_algorithm": self.genetic_algorithm,
            # "particle_swarm_optimization": self.particle_swarm_optimization,
            # "constraint_programming": self.constraint_programming,
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
        return np.round(
            peak_fatigue
            * (N_DAYS_IN_WEEK - np.arange(N_DAYS_IN_WEEK))
            / N_DAYS_IN_WEEK,
            3,
        )

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
                all_fatigues[start_index:end_index] += fatigue[
                    : end_index - start_index
                ]
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
        # Validate input schedule
        for week_index, week_workouts in enumerate(workouts):
            if len(week_workouts) != N_DAYS_IN_WEEK:
                raise ValueError(f"Invalid number of workouts for week {week_index=}.")

        best_score = float("inf")
        best_schedule = None
        best_optimizer = None
        optimizer_results = {}

        # Run all optimizers and collect results
        for optimizer_name, optimizer in self.optimizer_registry.items():
            try:
                # Generate schedule and compute score
                workout_schedule = optimizer(workouts)
                score = self.score(workout_schedule)

                # Store results
                optimizer_results[optimizer_name] = {
                    "score": score,
                    "workout_schedule": workout_schedule,
                }

                # Track the best result
                if score < best_score:
                    best_score = score
                    best_schedule = workout_schedule
                    best_optimizer = optimizer_name

            except Exception as e:
                # Handle optimizer failures gracefully
                print(f"Optimizer {optimizer_name} failed with error: {e}")

        # Ensure at least one optimizer succeeded
        if not optimizer_results:
            raise ValueError("No valid optimizers produced a schedule.")

        # Print results
        print(" ")
        print("-" * 100)
        print("Optimizer results :")
        print({
            optimizer: optimizer_results[optimizer]["score"]
            for optimizer in optimizer_results.keys()
        })
        print(f"Best optimizer: {best_optimizer}")
        print("-" * 100)
        print(" ")

        return best_schedule


    def brute_optimize_weeks_independently(
        self, workouts: list[list[Workout]]
    ) -> list[list[Workout]]:
        optimized_schedule = []

        for week_workouts in workouts:
            best_week_schedule = None
            best_week_score = float("inf")

            for permuted_week in permutations(week_workouts):
                current_score = self.score([list(permuted_week)])

                if current_score < best_week_score:
                    best_week_schedule = list(permuted_week)
                    best_week_score = current_score

            optimized_schedule.append(best_week_schedule)

        return optimized_schedule
    
    def simulated_annealing(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        if self.random_seed is not None:
            random.seed(self.random_seed)

        def neighbor(schedule):
            week = random.randint(0, len(schedule) - 1)
            day1, day2 = random.sample(range(len(schedule[week])), 2)
            new_schedule = [list(w) for w in schedule]
            new_schedule[week][day1], new_schedule[week][day2] = (
                new_schedule[week][day2],
                new_schedule[week][day1],
            )
            return new_schedule

        current_schedule = self.return_input_schedule(workouts)
        current_score = self.score(current_schedule)
        best_schedule, best_score = current_schedule, current_score
        temperature = 1000

        while temperature > 0.01:
            candidate_schedule = neighbor(current_schedule)
            candidate_score = self.score(candidate_schedule)
            delta = candidate_score - current_score

            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_schedule, current_score = candidate_schedule, candidate_score
                if current_score < best_score:
                    best_schedule, best_score = current_schedule, current_score

            temperature *= 0.98

        return best_schedule


    def genetic_algorithm(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        if self.random_seed is not None:
            random.seed(self.random_seed)

        def crossover(parent1, parent2):
            # Combine two schedules
            return [
                parent1[i] if i % 2 == 0 else parent2[i] for i in range(len(parent1))
            ]

        def mutate(schedule):
            # Swap two workouts in a random week
            week = random.randint(0, len(schedule) - 1)
            day1, day2 = random.sample(range(len(schedule[week])), 2)
            schedule[week][day1], schedule[week][day2] = (
                schedule[week][day2],
                schedule[week][day1],
            )
            return schedule
        
        POPULATION_SIZE = 100
        GENERATIONS = 500

        # Initialize population with consistent randomness
        population = [self.return_input_schedule(workouts) for _ in range(POPULATION_SIZE)]
        for _ in range(GENERATIONS):  # Number of generations
            scores = [self.score(individual) for individual in population]
            selected = sorted(zip(scores, population), key=lambda x: x[0])[:POPULATION_SIZE // 2]
            parents = [ind[1] for ind in selected]

            population = []
            for _ in range(POPULATION_SIZE):  # Maintain population size
                parent1, parent2 = random.sample(parents, 2)
                child = mutate(crossover(parent1, parent2))
                population.append(child)

        best_schedule = min(population, key=self.score)
        return best_schedule

    def particle_swarm_optimization(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        # Seed the random number generator for reproducibility
        if hasattr(self, 'random_seed') and self.random_seed is not None:
            random.seed(self.random_seed)

        class Particle:
            def __init__(self, schedule, score):
                self.schedule = schedule  # Current schedule (list of lists)
                self.velocity = []  # A list of swaps (perturbations)
                self.best_schedule = schedule  # Best known schedule
                self.best_score = score  # Best score achieved

        def initialize_random_schedule(workouts):
            """Creates a random valid schedule."""
            random_schedule = [random.sample(week, len(week)) for week in workouts]
            return random_schedule

        def apply_velocity(schedule, velocity):
            """Applies a list of swaps (velocity) to a schedule."""
            new_schedule = [week[:] for week in schedule]  # Deep copy
            for week_idx, day1, day2 in velocity:
                new_schedule[week_idx][day1], new_schedule[week_idx][day2] = (
                    new_schedule[week_idx][day2],
                    new_schedule[week_idx][day1],
                )
            return new_schedule

        def generate_velocity(schedule, target_schedule):
            """Generates a velocity (list of swaps) to move closer to the target schedule."""
            velocity = []
            for week_idx, (current_week, target_week) in enumerate(zip(schedule, target_schedule)):
                for i in range(len(current_week)):
                    if current_week[i] != target_week[i]:
                        swap_idx = target_week.index(current_week[i])
                        velocity.append((week_idx, i, swap_idx))
            return velocity

        # Initialize particles
        particles = []
        num_particles = 10  # Number of particles
        for _ in range(num_particles):
            schedule = initialize_random_schedule(workouts)
            score = self.score(schedule)
            particles.append(Particle(schedule, score))

        # Initialize global best
        global_best_particle = min(particles, key=lambda p: p.best_score)
        global_best_schedule = global_best_particle.schedule
        global_best_score = global_best_particle.best_score

        # PSO Hyperparameters
        inertia_weight = 0.1  # Momentum component
        cognitive_weight = 5  # Self-best attraction
        social_weight = 5  # Global-best attraction
        max_iterations = 5000  # Number of iterations

        for iteration in range(max_iterations):

            for particle in particles:
                # Generate new velocity
                personal_velocity = generate_velocity(particle.schedule, particle.best_schedule)
                global_velocity = generate_velocity(particle.schedule, global_best_schedule)

                # Combine velocities probabilistically
                new_velocity = []
                if random.random() < inertia_weight:
                    new_velocity.extend(random.sample(personal_velocity, min(len(personal_velocity), 3)))
                if random.random() < cognitive_weight:
                    new_velocity.extend(random.sample(personal_velocity, min(len(personal_velocity), 3)))
                if random.random() < social_weight:
                    new_velocity.extend(random.sample(global_velocity, min(len(global_velocity), 3)))

                particle.velocity = new_velocity

                # Apply velocity to update the schedule
                new_schedule = apply_velocity(particle.schedule, particle.velocity)
                new_score = self.score(new_schedule)

                # Update personal best if this schedule is better
                if new_score < particle.best_score:
                    particle.best_schedule = new_schedule
                    particle.best_score = new_score

                # Update global best if this schedule is better
                if new_score < global_best_score:
                    global_best_schedule = new_schedule
                    global_best_score = new_score

            # Early stopping if global best score is perfect
            if global_best_score == 0:
                break

        return global_best_schedule


    def constraint_programming(self, workouts: list[list[Workout]]) -> list[list[Workout]]:
        model = cp_model.CpModel()
        schedule = {}
        for week_index, week in enumerate(workouts):
            for day_index, workout in enumerate(week):
                schedule[(week_index, day_index)] = model.NewIntVar(
                    0, len(week) - 1, f"w{week_index}_d{day_index}"
                )

        model.AddAllDifferent(schedule.values())
        fatigue = model.NewIntVar(0, 1000, "fatigue")
        fatigue_contributions = []
        for (week_index, day_index), var in schedule.items():
            fatigue_contributions.append(self.fatigue_profile_mapping[workouts[week_index][day_index]][day_index])

        model.AddMaxEquality(fatigue, fatigue_contributions)
        model.Minimize(fatigue)
        solver = cp_model.CpSolver()
        solver.Solve(model)

        optimized_schedule = []
        for week_index in range(len(workouts)):
            week_schedule = [
                workouts[week_index][solver.Value(schedule[(week_index, day_index)])]
                for day_index in range(len(workouts[week_index]))
            ]
            optimized_schedule.append(week_schedule)
        return optimized_schedule

    def return_input_schedule(
        self, workouts: list[list[Workout]]
    ) -> list[list[Workout]]:
        return workouts
    
