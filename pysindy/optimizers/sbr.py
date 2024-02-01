from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
from diffrax import diffeqsolve
from diffrax import ODETerm
from diffrax import SaveAt
from diffrax import Tsit5
from jax import random
from numpyro.distributions import Exponential
from numpyro.distributions import HalfCauchy
from numpyro.distributions import InverseGamma
from numpyro.distributions import Normal
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer import Predictive
from numpyro.infer.initialization import init_to_value

from .base import BaseOptimizer


class SBR(BaseOptimizer):
    """
    Sparse Bayesian Regression (SBR) optimizer. This uses the regularised
    horseshoe prior over the SINDy coefficients to achieve sparsification.

    The horseshoe prior contains a "spike" of nonzero probability at the
    origin, and a "slab" of distribution in cases where a coefficient is nonzero.

    The SINDy coefficients are set as the posterior means of the MCMC NUTS samples.
    Additional statistics can be computed from the MCMC samples stored in
    the mcmc_ attribute using e.g. ArviZ.

    This implementation differs from the method described in Hirsh et al. (2021)
    by imposing the error model directly on the derivatives, rather than on the
    states, circumventing the need to integrate the equation to evaluate the
    posterior density.

    TODO: Implement the data-generating model described in eq. 2.4 of Hirsh
    et al. (2021). This could be achieved using the JAX-based solver "diffrax".
    Se discussion in https://github.com/dynamicslab/pysindy/pull/440 for more
    details.

    See the following reference for more details:

        Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
        Sparsifying Priors for Bayesian Uncertainty Quantification in
        Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107

    Parameters
    ----------
    sparsity_coef_tau0 : float, optional (default 0.1)
        Sparsity coefficient for regularised horseshoe hyper-prior. Lower
        value increases the sparsity of the SINDy coefficients.

    slab_shape_nu : float, optional (default 4)
        Controls spread of slab.  For values less than 4,
        the kurtosis of of nonzero coefficients is undefined.  As  the value
        increases past 4, for higher values, the variance and kurtosis approach
        :math:`s` and :math:`s^2`, respectively

    slab_shape_s : float, optional (default 2)
        Controls spread of slab.  Higher values lead to more spread
        out nonzero coefficients.

    noise_hyper_lambda : float, optional (default 1)
        Rate hyperparameter for the exponential prior distribution over
        the noise standard deviation.

    num_warmup : int, optional (default 1000)
        Number of warmup (or "burnin") MCMC samples to generate. These are
        discarded before analysis and are not included in the posterior samples.

    num_samples : int, optional (default 5000)
        Number of posterior MCMC samples to generate.

    mcmc_kwargs : dict, optional (default None)
        Instructions for MCMC sampling.
        Keyword arguments are passed to numpyro.infer.MCMC

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Posterior means of the SINDy coefficients.

    mcmc_ : numpyro.infer.MCMC
        Complete traces of the posterior samples.
    """

    def __init__(
        self,
        sparsity_coef_tau0: float = 0.1,
        slab_shape_nu: float = 4,
        slab_shape_s: float = 2,
        noise_hyper_lambda: float = 1,
        num_warmup: int = 1000,
        num_samples: int = 5000,
        mcmc_kwargs: Optional[dict] = None,
        unbias: bool = False,
        **kwargs,
    ):

        if unbias:
            raise ValueError("SBR is incompatible with unbiasing. Set unbias=False")
        self.init = kwargs.pop("init", None)
        super().__init__(unbias=unbias, **kwargs)

        self.integrator = DiffraxModel()

        # set the hyperparameters
        self.sparsity_coef_tau0 = sparsity_coef_tau0
        self.slab_shape_nu = slab_shape_nu
        self.slab_shape_s = slab_shape_s
        self.noise_hyper_lambda = noise_hyper_lambda

        # set MCMC sampling parameters.
        self.num_warmup = num_warmup
        self.num_samples = num_samples

        # set the MCMC kwargs.
        if mcmc_kwargs is not None:
            self.mcmc_kwargs = mcmc_kwargs
        else:
            self.mcmc_kwargs = {}

    def _reduce(self, x, y):
        # set up a sparse regression and sample.
        self.mcmc_ = self._run_mcmc(x, y, **self.mcmc_kwargs)

        # set the mean values as the coefficients.
        self.coef_ = np.array(self.mcmc_.get_samples()["beta"].mean(axis=0))

    def _numpyro_model(self, x, y):
        # get the dimensionality of the problem.
        n_features = x.shape[1]
        n_targets = y.shape[1]

        # sample the horseshoe hyperparameters.
        tau = numpyro.sample("tau", HalfCauchy(self.sparsity_coef_tau0))
        c_sq = numpyro.sample(
            "c_sq",
            InverseGamma(
                self.slab_shape_nu / 2, self.slab_shape_nu / 2 * self.slab_shape_s**2
            ),
        )

        # sample the parameters compute the predicted outputs.
        beta = _sample_reg_horseshoe(tau, c_sq, (n_targets, n_features))
        mu = self.integrator.solve(x[0, :n_targets], jnp.linspace(0, 30, 50), beta)

        # compute the likelihood.
        sigma = numpyro.sample("sigma", Exponential(self.noise_hyper_lambda))
        numpyro.sample("obs", Normal(mu, sigma), obs=x[:, :n_targets])

    def _run_mcmc(self, x, y, **kwargs):

        # set up a jax random key.
        seed = kwargs.pop("seed", 0)
        key = random.PRNGKey(seed)

        # init = self.init
        key, subkey = random.split(key)
        init = Predictive(self._numpyro_model, num_samples=1, batch_ndims=1)(
            subkey, x, y
        )
        init = {key: value[0] for key, value in init.items()}
        init["beta"] = self.init["beta"]
        # init["beta"] = jnp.linalg.lstsq(x, y)[0].T

        # run the MCMC
        kernel = NUTS(self._numpyro_model, init_strategy=init_to_value(values=init))
        mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, **kwargs
        )

        key, subkey = random.split(key)
        mcmc.run(subkey, x=x, y=y)

        # extract the MCMC samples and compute the UQ-SINDy parameters.
        return mcmc


def _sample_reg_horseshoe(tau, c_sq, shape):
    lamb = numpyro.sample("lambda", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    beta = numpyro.sample(
        "beta",
        Normal(jnp.zeros_like(lamb_squiggle), jnp.sqrt(lamb_squiggle**2 * tau**2)),
    )
    return beta


class DiffraxModel:
    def __init__(self, **kwargs):

        self.term = ODETerm(self.dxdt)
        self.solver = Tsit5()

        self.dt = kwargs.pop("dt", 0.1)
        self.kwargs = kwargs

    def solve(self, y0, t, beta):
        saveat = SaveAt(ts=t)
        return diffeqsolve(
            self.term,
            self.solver,
            t0=t[0],
            t1=t[-1],
            dt0=self.dt,
            y0=y0,
            args=beta,
            saveat=saveat,
            **self.kwargs,
        ).ys

    def dxdt(self, t, x, args):
        _x = jnp.array([x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2])
        return jnp.dot(_x, args.T) - 1e-5 * x**3
