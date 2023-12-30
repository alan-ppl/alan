import torch as t
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from .moments import mean, mean2, mean_log, mean_log1m, mean_xxT, cov_x, vec_square

import alan.postproc as pp

Tensor = (functorch.dim.Tensor, t.Tensor)

def grad_digamma(x):
    return t.special.polygamma(1, x)

def inverse_digamma(y):
    """
    Solves y = digamma(x)
    or computes x = digamma^{-1}(y)
    Appendix C in https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    Works very well assuming the x's you start with are all positive
    """
    x_init_for_big_y = y.exp()+0.5
    x_init_for_small_y = -t.reciprocal(y-t.digamma(t.ones(())))
    x = t.where(y>-2.22, x_init_for_big_y, x_init_for_small_y)
    for _ in range(6):
        x = x - (t.digamma(x) - y)/grad_digamma(x)
    return x

def tuple_assert_allclose(xs, ys):
    for (x, y) in zip(xs, ys):
        assert t.allclose(x, y, atol=1E-5)

def dict_assert_allclose(xs, ys):
    assert set(xs.keys()) == set(ys.keys())
    for key in xs:
        assert t.allclose(xs[key], ys[key], atol=1E-5)

class AbstractConversion():
    @staticmethod
    def canonical_conv(**kwargs)
        return kwargs


class BernoulliConversion(AbstractConversion):
    dist = t.distributions.Bernoulli
    sufficient_stats = (mean,)
    @staticmethod
    def conv2mean(probs):
        return (probs,)
    @staticmethod
    def mean2conv(mean):
        return {'probs': mean}
    @staticmethod
    def canonical_conv(logits=None, probs=None):
        assert (probs is None) != (logits is None)
        return {'probs': sigmoid(logits) if (logits is not None) else probs}
    @staticmethod
    def test_conv(N):
        return {'logits': t.randn(N)}


class PoissonConversion(AbstractConversion):
    dist = t.distributions.Poisson
    sufficient_stats = (mean,)
    @staticmethod
    def conv2mean(rate):
        return (mean,)
    @staticmethod
    def mean2conv(mean):
        return {'rate': mean}
    @staticmethod
    def test_conv(N):
        return {'rate': t.randn(N)}


class NormalConversion(AbstractConversion):
    dist = t.distributions.Normal
    sufficient_stats = (mean, mean2)
    @staticmethod
    def conv2mean(loc, scale):
        Ex  = loc
        Ex2 = loc**2 + scale**2
        return Ex, Ex2
    @staticmethod
    def mean2conv(Ex, Ex2):
        loc   = Ex
        var = Ex2 - loc**2
        scale = scale.floor(min=0.).sqrt()
        return {'loc': loc, 'scale': scale}
    @staticmethod
    def test_conv(N):
        return {'loc': t.randn(N), 'scale': t.randn(N).exp()}
    
class ExponentialConversion(AbstractConversion):
    dist = t.distributions.Exponential
    sufficient_stats = (mean,)
    @staticmethod
    def conv2mean(rate):
        return (t.reciprocal(rate),)
    @staticmethod
    def mean2conv(mean):
        return {'rate': t.reciprocal(mean)}
    @staticmethod
    def test_conv(N):
        return {'rate': t.randn(N).exp()}


class DirichletConversion(AbstractConversion):
    dist = t.distributions.Dirichlet
    sufficient_stats = (mean_log)

    @staticmethod
    def conv2mean(concentration):
        return (t.digamma(concentration) - t.digamma(concentration.sum(-1, keepdim=True)),)
    @staticmethod
    def mean2conv(logp):
        """
        Methods from https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
        """
        alpha = t.ones_like(logp)
        #Initialize with fixed point iterations from Eq. 9 that are slow, but guaranteed to converge
        for _ in range(5):
            alpha = inverse_digamma(t.digamma(alpha.sum(-1, keepdim=True)) + logp)

        #Clean up with a few fast but unstable Newton's steps (Eq. 15-18)
        for _ in range(6):
            sum_alpha = alpha.sum(-1, keepdim=True)
            g = (t.digamma(sum_alpha) - t.digamma(alpha) + logp) #Eq. 6
            z = grad_digamma(sum_alpha)
            q = - grad_digamma(alpha)
            b = (g/q).sum(-1, keepdim=True) / (1/z + (1/q).sum(-1, keepdim=True))
            alpha = alpha - (g - b)/q
        return {'concentration': alpha}

    @staticmethod
    def test_conv(N):
        return {'concentration': t.randn(N, 4).exp()}

    @staticmethod
    def canonical_conv(concentration):
        return {'concentration': concentration}

class BetaConversion(AbstractConversion):
    dist = t.distributions.Beta
    sufficient_stats = (mean_log, mean_log1m)

    @staticmethod
    def conv2mean(concentration1, concentration0):
        norm = t.digamma(concentration1 + concentration0)
        return (t.digamma(concentration1) - norm, t.digamma(concentration0) - norm)
    @staticmethod
    def mean2conv(Elogx, Elog1mx):
        logp = t.stack([Elogx, Elog1mx], -1)
        c = DirichletConversion.mean2conv(logp)['concentration']
        return {'concentration1': c[..., 0], 'concentration0': c[..., 1]}
    @staticmethod
    def test_conv(N):
        return {'concentration1': t.randn(N).exp(),'concentration0': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(concentration1, concentration0):
        return {'concentration1': concentration1, 'concentration0': concentration0}



class GammaConversion(AbstractConversion):
    """
    concentration == alpha
    rate == beta
    """
    dist = t.distributions.Gamma
    sufficient_stats = (mean_log, mean)

    @staticmethod
    def conv2mean(concentration, rate):
        #Tested by sampling
        alpha = concentration
        beta = rate
        return (-t.log(beta) + t.digamma(alpha), alpha/(beta))
    @staticmethod
    def mean2conv(Elogx, Ex):
        """
        Generalised Newton's method from Eq. 10 in https://tminka.github.io/papers/minka-gamma.pdf
        Rewrite as:
        1/a^new = 1/a (1 + num / a (1/a + grad_digamma(a)))
        1/a^new = 1/a (1 + num / (1 + a grad_digamma(a)))
        a^new   = a / (1 + num / (1 + a grad_digamma(a)))
        """
        logEx = (Ex).log()
        diff = (Elogx - logEx)
        alpha = - 0.5 / diff
        for _ in range(6):
            num = diff + alpha.log() - t.digamma(alpha)
            denom = 1 - alpha * grad_digamma(alpha)
            alpha = alpha * t.reciprocal(1 + num/denom)
        beta = alpha / Ex
        return {'concentration': alpha, 'rate': beta}

    @staticmethod
    def test_conv(N):
        return {'concentration': t.randn(N).exp(), 'rate': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(concentration, rate):
        return {'concentration': concentration, 'rate': rate}

#class InverseGammaConversion(AbstractConversion):
#PyTorch doesn't seem to have an Inverse Gamma distribution
#    dist = staticmethod(InverseGamma)
#    sufficient_stats = (t.log, t.reciprocal)
#
#    @staticmethod
#    def conv2nat(alpha, beta):
#        return (-alpha-1, -beta)
#    @staticmethod
#    def nat2conv(nat0, nat1):
#        return (-nat0-1, -nat1)
#
#    @staticmethod
#    def conv2mean(alpha, beta):
#        #From Wikipedia (Inverse Gamma: Properties)
#        return (t.log(beta) - t.digamma(alpha), alpha/beta)
#    @staticmethod
#    def mean2conv(mean_0, mean_1):
#        return GammaConversion.mean2conv(-mean_0, mean_1)
#
#    @staticmethod
#    def test_conv(N):
#        return (t.randn(N).exp(),t.randn(N).exp())

def posdef_matrix_inverse(x):
    return t.cholesky_inverse(t.linalg.cholesky(x))
class MvNormalConversion(AbstractConversion):
    dist = t.distributions.MultivariateNormal
    sufficient_stats = (mean, mean_xxT)

    @staticmethod
    def conv2mean(loc, covariance_matrix):
        return (loc, covariance_matrix + vec_square(loc))
    @staticmethod
    def mean2conv(Ex, Ex2):
        return {'loc': Ex, 'covariance_matrix': Ex2 - vec_square(Ex)}

    @staticmethod
    def test_conv(N):
        mu = t.randn(N, 2)
        V = t.randn(N, 2, 4)
        S = V @ V.mT / 4
        return {'loc': mu, 'covariance_matrix': S}

    @staticmethod
    def canonical_conv(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        assert 1 == sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril])
        if precision_matrix is not None:
            covariance_matrix = posdef_matrix_inverse(precision_matrix)
        elif scale_tril is not None:
            covariance_matrix = scale_tril @ scale_tril.mT
        return {'loc': loc, 'covariance_matrix': covariance_matrix}
