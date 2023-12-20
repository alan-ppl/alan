class Moment:
    def __init__(self):
        raise Exception("Moment objects should be used directly, and not instantiated")

class RawMoment(Moment):
    """
    Must be overwritten with self.f as a static method.
    """
    @staticmethod
    def from_sample(sample, Ndim):
        return t.mean(self.f(sample), Ndim)

    @staticmethod
    def from_marginals(sample, weights, Kdim):
        return t.sum(self.f(sample) * weights, Kdim)

    @classmethod
    def all_raw_moments(cls):
        return cls



class CompoundMoment(Moment):
    """
    Must be overwritten with:
    raw_moments as a list of RawMoments
    combiner as a function that combines the raw moments
    """
    @classmethod
    def from_sample(cls, sample, Ndim):
        moments = [raw_moment.from_sample(sample, Ndim) for raw_moment in cls.raw_moments]
        return cls.combiner(*moments)

    @classmethod
    def from_marginals(cls, sample, weights, Kdim):
        moments = [raw_moment.from_marginals(sample, weights, Kdim) for raw_moment in cls.raw_moments]
        return cls.combiner(*moments)

    @classmethod
    def all_raw_moments(cls):
        return cls.raw_moments



class Mean(RawMoment)
    @staticmethod
    def f(x):
        return x

class Mean2(RawMoment)
    @staticmethod
    def f(x):
        return x*x

class Var(CompoundMoment):
    raw_moments = (Mean, Mean2)
    @staticmethod
    def combiner(mean, mean2):
        return mean2 - mean**2
