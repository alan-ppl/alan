from .utils import *

class Moment:
    def __init__(self):
        raise Exception("Moment objects should be used directly, and not instantiated")

class RawMoment(Moment):
    """
    Must be overwritten with self.f as a static method.
    """
    @classmethod
    def from_samples(cls, samples:tuple[Tensor], Ndim:Dim):
        return cls.f(*samples).mean(Ndim)

    @classmethod
    def from_marginals(cls, samples:tuple[Tensor], weights:Tensor, all_platedims:dict[str, Dim]):
        assert isinstance(samples, tuple)
        assert isinstance(weights, Tensor)

        set_all_platedims = set(all_platedims.values())
        f = cls.f(*samples)
        f_Kdims = set(generic_dims(f)).difference(set_all_platedims)
        w_Kdims = set(generic_dims(weights)).difference(set_all_platedims)
        assert f_Kdims.issubset(w_Kdims)
        tuple_w_Kdims = tuple(w_Kdims)
        assert 0 < len(tuple_w_Kdims)
        return (f * weights).sum(tuple_w_Kdims)

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
    def from_samples(cls, samples:tuple[Tensor], Ndim):
        moments = [raw_moment.from_samples(samples, Ndim) for raw_moment in cls.raw_moments]
        return cls.combiner(*moments)

    @classmethod
    def from_marginals(cls, samples:tuple[Tensor], weights:Tensor, all_platedims:dict[str, Dim]):
        moments = [raw_moment.from_marginals(samples, weights, all_platedims) for raw_moment in cls.raw_moments]
        return cls.combiner(*moments)

    @classmethod
    def all_raw_moments(cls):
        return cls.raw_moments



class Mean(RawMoment):
    @staticmethod
    def f(x):
        return x

class Mean2(RawMoment):
    @staticmethod
    def f(x):
        return x*x

class Var(CompoundMoment):
    raw_moments = (Mean, Mean2)
    @staticmethod
    def combiner(mean, mean2):
        return mean2 - mean**2

def uniformise_moment_args(args):
    """
    moment can be called in a bunch of different ways.  For a single variable/set of variables:
    * `sample.moments("a", Mean)`
    * `sample.moments("b", (Mean, Var))`
    * `sample.moments(("a", "b"), Cov)`

    For multiple variables:
    sample.moments({
        "a": Mean,
        "b": (Mean, Var),
        ("a", "b"): Cov
    })

    This function converts all these argument formats into a uniform dictionary, mapping tuples of input variables to tuples of moments.
    """
    assert isinstance(args, tuple)

    mom_args_exception = Exception(".moment must be called as ...")

    #Converts everthing to a dict.
    if   1 == len(args):
        args = args[0]
        if not isinstance(args, dict):
            raise mom_args_exception
    elif 2 == len(args):
        args = {args[0]: args[1]}
    else:
        raise mom_args_exception

    uniform_arg_dict = {}
    for k, v in args.items():
        if not isinstance(k, (tuple, str)):
            raise mom_args_exception
        if not (isinstance(v, tuple) or issubclass(v, Moment)):
            raise mom_args_exception

        if not isinstance(k, tuple):
            k = (k,)
        if not isinstance(v, tuple):
            v = (v,)

        uniform_arg_dict[k] = v

    return uniform_arg_dict


def postproc_moment_outputs(result, raw_moms):
    #If we weren't given a dict, we should just return a value, _not_ a dict.
    if 2==len(raw_moms):
        result = next(iter(result.values()))
        assert isinstance(result, tuple)

        #If we weren't given a tuple of moments, just a single moment, we should return a single tensor
        if not isinstance(raw_moms[1], tuple):
            assert len(result) == 1
            result = result[0]
    return result
