from typing import Optional
from typing import Tuple

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

from ..feature_library import PolynomialLibrary
from .base import BaseOptimizer

COMPATIBLE_FEATURE_LIBRARIES = PolynomialLibrary


class SBR(BaseOptimizer):
    """
    Sparse Bayesian Regression (SBR) optimizer. This uses the regularised
    horseshoe prior over the SINDy coefficients to achieve sparsification.

    The horseshoe prior contains a "spike" of nonzero probability at the
    origin, and a Student's-T-shaped "slab" of distribution in cases where a
    coefficient is nonzero.


    The SINDy coefficients are set as the posterior means of the MCMC NUTS samples.
    Additional statistics can be computed from the MCMC samples stored in
    the mcmc attribute using e.g. ArviZ.

    This implementation differs from the method described in Hirsh et al. (2021)
    by imposing the error model directly on the derivatives, rather than on the
    states, circumventing the need to integrate the equation to evaluate the
    posterior density. One consequence of this is that the noise standard
    deviation "sigma" is with respect to the derivatives instead of the states
    and hence should not be interpreted.

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
        Along with ``slab_shape_s``, controls tails of nonzero coefficients.
        Specifically, degrees of freedom for Student's-T-shaped slab.
        Higher values decrease excess kurtosis to zero, lower values >= 4
        increase kurtosis to infinity.

    slab_shape_s : float, optional (default 2)
        Along with ``slab_shape_nu``, controls standard deviation of nonzero
        coefficients.

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
        exact: bool = True,
        unbias: bool = False,
        **kwargs,
    ):

        if unbias:
            raise ValueError("SBR is incompatible with unbiasing. Set unbias=False")

        super().__init__(unbias=unbias, **kwargs)

        # check that hyperparameters are positive.
        if sparsity_coef_tau0 <= 0:
            raise ValueError("sparsity_coef_tau0 must be positive")
        if slab_shape_nu <= 0:
            raise ValueError("slab_shape_nu must be positive")
        if slab_shape_s <= 0:
            raise ValueError("slab_shape_s must be positive")
        if noise_hyper_lambda <= 0:
            raise ValueError("noise_hyper_lambda must be positive")

        # check that samples are positive integers.
        if not isinstance(num_warmup, int) or num_warmup < 0:
            raise ValueError("num_warmup must be a positive integer")
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError("num_samples must be a positive integer")

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

        self.exact = exact

    def _pre_fit_hook(self, t, estimator):

        if self.exact:
            # internalise the time vector for integrations.
            self.t = jnp.array(t[0])

            # internalise the feature library for identifying the state variables.
            self.feature_library = estimator.feature_library

            # get the number of control features.
            self.n_control_features_ = estimator.n_control_features_

            if not isinstance(self.feature_library, COMPATIBLE_FEATURE_LIBRARIES):
                raise TypeError(
                    f"{self.feature_library.__class__.__name__} is not compatible"
                    " with exact SBR optimizer"
                )

        else:
            self.t = None

    def _reduce(self, x, y):

        if self.exact:
            # set up the initial value problem solver.
            self.integrator = DiffraxModel(
                self.feature_library, **self.integrator_kwargs
            )

            # extract the columns containing the state and control variables.
            self.n_state_features_ = (
                self.feature_library.n_features_in_ - self.n_control_features_
            )
            y = self._extract_state(x, self.n_state_features_)
            u = self._extract_control(
                x, self.n_state_features_, self.n_control_features_
            )

            # set the forward method to use the integrator.
            self.forward = self._forward_exact

        else:
            # set the forward method to bypass the integrator.
            self.forward = self._forward_approximate
            u = None

        # set up a sparse regression and sample.
        self.mcmc_ = self._run_mcmc(x, y, u, **self.mcmc_kwargs)

        # set the mean values as the coefficients.
        self.coef_ = np.array(self.mcmc_.get_samples()["beta"].mean(axis=0))

    def _extract_state(self, x, ns):

        # extract the columns containing the state variables.
        if self.feature_library.include_bias:
            return x[:, 1 : 1 + ns]
        else:
            return x[:, :ns]

    def _extract_control(self, x, ns, nc):

        if self.n_control_features_ == 0:
            return None

        # extract the columns containing the control variables.
        if self.feature_library.include_bias:
            return jnp.array(x[:, 1 + ns : 1 + ns + nc])
        else:
            return jnp.array(x[:, ns : ns + nc])

    def _forward_exact(self, x, t, y, u, beta, sigma):
        y0 = numpyro.sample("y0", Normal(y[0, :], sigma))
        return self.integrator.solve(t, y0, u, beta)

    def _forward_approximate(self, x, t, y, u, beta, sigma):
        return jnp.dot(x, beta.T)

    def _numpyro_model(self, x, t, y, u):
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

        # sample the SINDy coefficients and the measurement noise standard deviation.
        beta = _sample_reg_horseshoe(tau, c_sq, (n_targets, n_features))
        sigma = numpyro.sample("sigma", Exponential(self.noise_hyper_lambda))

        # run the respective forward model.
        mu = self.forward(x, t, y, u, beta, sigma)

        # compute the likelihood.
        numpyro.sample("obs", Normal(mu, sigma), obs=y)

    def _run_mcmc(self, x, y, u, **kwargs):

        # set up a jax random key.
        seed = kwargs.pop("seed", 0)
        key = random.PRNGKey(seed)

        # get the initial values if they were provided.
        given_initial_values = kwargs.pop("initial_values", None)
        if given_initial_values is None:
            init_strategy = init_to_sample()
        else:
            # sample the remaining random variables.
            key, subkey = random.split(key)
            sampled_initial_values = Predictive(
                self._numpyro_model, num_samples=1, batch_ndims=1
            )(subkey, x, self.t, y, u)
            sampled_initial_values = {
                key: value[0] for key, value in sampled_initial_values.items()
            }
            initial_values = {**sampled_initial_values, **given_initial_values}
            init_strategy = init_to_value(values=initial_values)

        # run the MCMC
        kernel = NUTS(self._numpyro_model, init_strategy=init_strategy)
        mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, **kwargs
        )
        mcmc.run(key, x=x, t=self.t, y=y, u=u)

        # extract the MCMC samples and compute the UQ-SINDy parameters.
        return mcmc


def _sample_reg_horseshoe(tau: float, c_sq: float, shape: Tuple[int, ...]):
    """Create a regularized horseshoe distribution

    The regularized horseshoe distribution behaves like a horseshoe prior when
    shrinkage ``lamb`` is small, but behaves like a gaussian slab of variance
    ``c_sq`` when ``lamb`` is big or a Student T-slab when ``c_sq`` is itself
    an inverse Gamma.

    For original work, including interpretation of the coefficients, see:

    Piironen, J., and Vehtari, A. (2017). Sparsity Information and
    Regularization in the Horseshoe and Other Shrinkage Priors. Electronic Journal
    of Statistics Vol. 11 pp 5018-5051. https://doi.org/10.1214/17-EJS1337SI
    """
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
        self.feature_library = ProxyFeatureLibrary(feature_library)

        self.term = ODETerm(self.dxdt)
        self.solver = kwargs.pop("solver", Tsit5())
        self.dt = kwargs.pop("dt", 0.1)

        # there is a strange bug that raises a XlaRuntimeError when
        # using PIDController in conjunction with numpyro.infer.MCMC.
        self.stepsize_controller = kwargs.pop("stepsize_controller", ConstantStepSize())

    def solve(self, t, y0, u, beta):

        # saveat are the timesteps with measurements.
        saveat = SaveAt(ts=t)

        if u is not None:
            self.u = [lambda _t: jnp.interp(_t, t, _u) for _u in u.T]
        else:
            self.u = []

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

    def dxdt(self, t, x, args):
        u = jnp.array([_u(t) for _u in self.u])
        _x = jnp.hstack((x, u))
        _x = self.feature_library.transform(_x)
        return jnp.dot(_x, args.T) - 1e-6 * x**self.feature_library.stability_order


class ProxyFeatureLibrary:
    def __init__(self, feature_library):

        self.feature_library = feature_library

        # handle PolynomialLibrary.
        if isinstance(self.feature_library, PolynomialLibrary):
            # get the feature combinations
            _combinations = self.feature_library._combinations(
                self.feature_library.n_features_in_,
                self.feature_library.degree,
                self.feature_library.include_interaction,
                self.feature_library.interaction_only,
                self.feature_library.include_bias,
            )
            self.combinations = list(_combinations)

            self.order = self.feature_library.degree
            self.transform = self._transform_polynomial_features

        # set the degree of the stability term.
        if self.order % 2 == 0:
            self.stability_order = self.order + 1
        else:
            self.stability_order = self.order + 2

    def _transform_polynomial_features(self, x):
        # construct features from the state variables.
        _x = jnp.array(
            [
                x[..., self.combinations[i]].prod(-1)
                for i in range(self.feature_library.n_output_features_)
            ]
        )
        return _x
