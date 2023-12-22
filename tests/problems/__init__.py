from .test_problem import TestProblem
from .model1 import tp as model1
from .bernoulli_no_plate import tp as bernoulli_no_plate

problems = {
    'model1': model1,
    'bernoulli_no_plate': bernoulli_no_plate,
}

__all__ = problems, TestProblem
