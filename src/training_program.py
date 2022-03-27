

class TrainingExercise:

    def __init__(self, cfg, xs, ys, zs):
        self.cfg = cfg
        self.xs, self.ys, self.zs = xs, ys, zs

    def sample(self, overload):
        raise NotImplementedError


class TrainingProgram:

    def __init__(self, cfgs, meshes):
        self.exercises = {}
        for cfg, (exercise, (xs, ys, zs)) in zip(cfgs, meshes.items()):
            self.exercises[exercise] = TrainingExercise(cfg, xs, ys, zs)

    def sample_program(self, fname, cfg = None, **kwargs):
        raise NotImplementedError
        # for week i in progressive_overload
            # for exercise in exercises
                # sample week i training
