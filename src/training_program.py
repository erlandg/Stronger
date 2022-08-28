import numpy as np
from scipy.stats import norm
import pandas as pd
import random
from ast import literal_eval

import config
from utils import round_to_closest


LOW_REP_LIMIT = 3 # inclusive


def get_session_parameters(week, return_relative_intensity = False):
    """
    Determines how the intensity and focus of each exercise
    throughout the week should be distributed.
    """
    week = pd.DataFrame(week)
    focus = week / week.sum() # Day-wise attention to exercis
    intensity = focus.div(focus.sum(1), axis=0) # Total focus on an exercise for a given week

    # Add noise
    noise_ = pd.DataFrame(2e-5 * (np.random.random(week.shape) - .5), index=week.index, columns=week.columns)
    intensity[intensity > 0] += noise_[intensity > 0]
    intensity = intensity.clip(0, 1)
    rel_intensity = intensity.copy()

    for _, week_ in intensity.iterrows():
        # Distribute exercise intensity in ascending order, 0 is excluded
        week_[week_ > 0] = week_[week_ > 0].argsort() + 1

    intensity = intensity.astype(int)
    if not return_relative_intensity:
        return focus.to_dict(), intensity.to_dict()
    else:
        return focus.to_dict(), intensity.to_dict(), rel_intensity.to_dict()



def format_program(program_dict):
    raise NotImplementedError



class TrainingExercise:

    def __init__(self, cfg, program_cfg, mesh):
        self.cfg = cfg
        self.program_cfg = program_cfg
        self.xs, self.ys, self.zs = mesh

    def test_top_set(self, rep, weight):
        difficulty = self.cfg.get_estimator(self.cfg.optimal_estimator)(weight, rep) / self.cfg.one_rm
        return (difficulty > .95) or (rep <= LOW_REP_LIMIT)

    def get_adaptive_mesh(self, week_i, week_total, increase_rate = None):
        if increase_rate is None:
            increase_rate = self.program_cfg.increase_rate
        scale_ = (2 - week_total + week_i) * increase_rate
        xs = round_to_closest(self.xs * (1 + scale_), config.SMALLEST_WEIGHT_STEP)
        return xs, self.ys, self.zs

    def sample(self, focus, intensity, week_i, overload_period):
        xs, ys, zs = self.get_adaptive_mesh(week_i, overload_period)
        range_ = (np.array((intensity - 1, intensity)) / (self.cfg.frequency))
        rep_norm = (zs / zs.sum(1)[:,None]).cumsum(1)
        argmin_range = np.array(tuple(np.abs(rep_norm - extremum_).argmin(1) for extremum_ in range_)).T
        argmin_range = argmin_range.clip(0, xs.shape[1]-1)

        for i, (low_, high_) in enumerate(argmin_range):
            zs[i][:low_] = 0.
            zs[i][high_+1:] = 0.

        idx_ = np.array([[f"({i},{j})" for j in range(zs.shape[1])] for i in range(zs.shape[0])])
        sample_ = literal_eval(np.random.choice(idx_.flatten(), p = zs.flatten()/zs.sum()))
        out_ = [(ys[sample_[0],0], round_to_closest(xs[0,sample_[1]], config.WEIGHT_STEP))]

        if self.test_top_set(*out_[0]):
            # If heavy low-rep set, add dropdown
            new_ps = zs.copy()
            new_ps[:LOW_REP_LIMIT] = 0.
            dropdown_set = literal_eval(np.random.choice(
                idx_.flatten(),
                p = new_ps.flatten()/new_ps.sum()
            ))
            out_.append((ys[dropdown_set[0],0], round_to_closest(xs[0,dropdown_set[1]], config.WEIGHT_STEP)))
        return out_



class TrainingProgram:

    def __init__(self, cfg, exercise_cfgs, meshes):
        self.cfg = cfg
        self.exercises = {}
        for exercise_cfg, (exercise, mesh) in zip(exercise_cfgs, meshes.items()):
            self.exercises[exercise] = TrainingExercise(exercise_cfg, self.cfg, mesh)

    def sample_program(self, fname, **kwargs):
        program = {}
        vol_ = {k: 0 for k in self.exercises.keys()}
        for week_i in range(self.cfg.progressive_overload_period):
            program[f"week_{week_i}"], vol_ = self.sample_week(week_i, vol_, return_volume = True)
        program = format_program(program)
        program.to_csv(fname)

    def sample_week(self, week_i, prev_volume, return_volume = False):
        week_dist = {}
        days_remaining = {k: v.cfg.frequency for k, v in self.exercises.items()}

        # Distribute exercises over the week
        for day_i in range(self.cfg.n_training_sessions):
            # A given training day
            ex_incl = {k: False for k in self.exercises.keys()}
            shuffled_dict = dict(random.sample(list(days_remaining.items()), len(days_remaining)))
            
            while not any(ex_incl.values()): # Ensure at least one excercise for each training day
                # FIX THIS
                for ex_, ex_rem in shuffled_dict.items(): # Randomise sample order
                    remaining_sessions = (self.cfg.n_training_sessions - day_i)
                    assert ex_rem <= remaining_sessions
                    if random.random() <= ex_rem / remaining_sessions:
                        ex_incl[ex_] = True
                        days_remaining[ex_] -= 1
                        if sum(days_remaining.values()) == remaining_sessions: continue # Ensure no "empty" remaining days
            week_dist[f"day_{day_i}"] = ex_incl

        focus, intensity, rel_intensity = get_session_parameters(week_dist, return_relative_intensity = True)

        # Sample days
        week = {}
        weekly_volume = {k: 0 for k in self.exercises.keys()}
        for (day_i, train_), focus_i, intensity_i, rel_intensity_i in zip(
            week_dist.items(),
            focus.values(),
            intensity.values(),
            rel_intensity.values()
        ):
            ordered_intensities = dict(sorted(rel_intensity_i.items(), key=lambda item: item[1], reverse = True))
            week[day_i] = {ex_: self.exercises[ex_].sample(focus_i[ex_], intensity_i[ex_], week_i, self.cfg.progressive_overload_period) for ex_ in ordered_intensities.keys() if train_[ex_]}
        raise NotImplementedError

        assert all(weekly_volume[k] > prev_volume[k] for k in weekly_volume.keys()), f"Progressive overload for week {week} not sufficient."
        assert True, "Exercise frequency correpond to config"
        if return_volume:
            return week, weekly_volume
        else:
            return week