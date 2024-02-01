from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
from diffrax import ConstantStepSize
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
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.initialization import init_to_value

import pysindy
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
        integrator_kwargs: Optional[dict] = None,
        unbias: bool = False,
        **kwargs,
    ):

        if unbias:
            raise ValueError("SBR is incompatible with unbiasing. Set unbias=False")

        super().__init__(unbias=unbias, **kwargs)

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

        # set the integrator kwargs.
        if integrator_kwargs is not None:
            self.integrator_kwargs = integrator_kwargs
        else:
            self.integrator_kwargs = {}

    def _pre_fit_hook(self, t, feature_library):
        self.t = jnp.array(t[0])
        self.feature_library = feature_library

    def _reduce(self, x, y):

        # set up the initial value problem solver.
        self.integrator = DiffraxModel(self.feature_library, **self.integrator_kwargs)

        # extract the columns containing the state variables.
        y = self._sanitize_data(x)

        # set up a sparse regression and sample.
        self.mcmc_ = self._run_mcmc(x, y, **self.mcmc_kwargs)

        # set the mean values as the coefficients.
        self.coef_ = np.array(self.mcmc_.get_samples()["beta"].mean(axis=0))

    def _sanitize_data(self, x):
        if isinstance(
            self.feature_library,
            pysindy.feature_library.polynomial_library.PolynomialLibrary,
        ):
            # extract the columns containing the state variables.
            if self.feature_library.include_bias:
                return x[:, 1 : self.feature_library.n_features_in_ + 1]
            else:
                return x[:, : self.feature_library.n_features_in_]

    def _numpyro_model(self, x, y, t):
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
        mu = self.integrator.solve(y[0, :], t, beta)

        # compute the likelihood.
        sigma = numpyro.sample("sigma", Exponential(self.noise_hyper_lambda))
        numpyro.sample("obs", Normal(mu, sigma), obs=y)

    def _run_mcmc(self, x, y, **kwargs):

        # set up a jax random key.
        seed = kwargs.pop("seed", 0)
        key = random.PRNGKey(seed)

        # get the initial values if they were provided.
        initial_values = kwargs.pop("initial_values", None)
        if initial_values is None:
            init_strategy = init_to_sample()
        else:
            # sample the remaining random variables.
            key, subkey = random.split(key)
            _initial_values = Predictive(
                self._numpyro_model, num_samples=1, batch_ndims=1
            )(subkey, x, y, self.t)
            _initial_values = {key: value[0] for key, value in _initial_values.items()}
            _initial_values["beta"] = jnp.array(initial_values)
            init_strategy = init_to_value(values=_initial_values)

        # run the MCMC
        kernel = NUTS(self._numpyro_model, init_strategy=init_strategy)
        mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, **kwargs
        )
        mcmc.run(key, x=x, y=y, t=self.t)

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
    def __init__(self, feature_library, **kwargs):

        # set the feature library
        self.feature_library = feature_library
        self._initialize_feature_library()

        self.term = ODETerm(self.dxdt)
        self.solver = kwargs.pop("solver", Tsit5())
        self.dt = kwargs.pop("dt", 0.1)

        # there is a strange bug that raises a XlaRuntimeError when
        # using PIDController in conjunction with numpyro.infer.MCMC.
        self.stepsize_controller = kwargs.pop("stepsize_controller", ConstantStepSize())

    def _initialize_feature_library(self):
        # handle PolynomialLibrary.
        if isinstance(
            self.feature_library,
            pysindy.feature_library.polynomial_library.PolynomialLibrary,
        ):
            # get the feature combinations
            _combinations = self.feature_library._combinations(
                self.feature_library.n_features_in_,
                self.feature_library.degree,
                self.feature_library.include_interaction,
                self.feature_library.interaction_only,
                self.feature_library.include_bias,
            )
            self.combinations = list(_combinations)

            # set the degree of the stability term.
            if self.feature_library.degree % 2 == 0:
                self.stability_degree = self.feature_library.degree + 1
            else:
                self.stability_degree = self.feature_library.degree + 2

            self._transform = self._transform_polynomial_features

    def solve(self, y0, t, beta):

        # saveat are the timesteps with measurements.
        saveat = SaveAt(ts=t)

        # run the Diffrax solver and return the y-values.
        return diffeqsolve(
            self.term,
            self.solver,
            t0=t[0],
            t1=t[-1],
            dt0=self.dt,
            y0=y0,
            args=beta,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        ).ys

    def _transform_polynomial_features(self, x):
        # construct features from the state variables.
        _x = jnp.array(
            [
                x[..., self.combinations[i]].prod(-1)
                for i in range(self.feature_library.n_output_features_)
            ]
        )
        return _x

    def dxdt(self, t, x, args):
        _x = self._transform(x)
        return jnp.dot(_x, args.T) - 1e-6 * x**self.stability_degree
