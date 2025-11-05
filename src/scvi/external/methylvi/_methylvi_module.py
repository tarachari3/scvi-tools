"""PyTorch module for methylVI for single cell methylation data."""

from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Binomial, Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.distributions import BetaBinomial
from scvi.external.methylvi import METHYLVI_REGISTRY_KEYS, DecoderMETHYLVI
from scvi.external.methylvi._base_components import BSSeqModuleMixin
from scvi.external.methylvi._utils import _context_cov_key, _context_mc_key
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder

TensorDict = dict[str, torch.Tensor]


class METHYLVAE(BaseModuleClass, BSSeqModuleMixin):
    """PyTorch module for methylVI.

    Parameters
    ----------
    n_input
        Total number of input genomic regions
    contexts
        List of methylation contexts (e.g. ["mCG", "mCH"])
    num_features_per_context
        Number of features corresponding to each context
    n_batch
        Number of batches, if 0, no batch correction is performed
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers_enc
        Number of hidden layers used for encoder NNs
    n_layers_dec
        Number of hidden layers used for decoder NNs
    dropout_rate_enc
        Dropout rate for neural networks in encoder
    dropout_rate_dec
        Dropout rate for neural networks in decoder
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    likelihood
        One of
        * ``'betabinomial'`` - BetaBinomial distribution
        * ``'binomial'`` - Binomial distribution
    dispersion
        One of the following
        * ``'region'`` - dispersion parameter of BetaBinomial is constant per region across cells
        * ``'region-cell'`` - dispersion can differ for every region in every cell
        * ``'nu'`` - dispersion of BetaBinomial(alpha,beta) for every region in every cell
    nu_params
        nu_max, m, and b for relationship between concentration and read depth
    mu_glob
        Boolean to include global means of PSI across events
    lin_decoder
        Boolean to use linear decoder
    """

    def __init__(
        self,
        n_input: int,
        contexts: Iterable[str],
        num_features_per_context: Iterable[int],
        n_batch: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_enc: int = 1,
        n_layers_dec: int = 1,
        dropout_rate_enc: float = 0.1,
        dropout_rate_dec: float = 0.0,
        log_variational: bool = True,
        likelihood: Literal["betabinomial", "binomial"] = "betabinomial",
        dispersion: Literal["region", "region-cell", "nu"] = "region",
        nu_params: dict = None,
        mu_glob: bool = False,
        mu_inits: float = None,
        lin_decoder: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch

        self.latent_distribution = "normal"
        self.dispersion = dispersion
        self.likelihood = likelihood
        self.contexts = contexts
        self.log_variational = log_variational
        self.num_features_per_context = num_features_per_context

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        self.z_encoder = Encoder(
            n_input * 2,  # Methylated counts and coverage for each feature --> x2
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers_enc,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_enc,
            return_dist=True,
            var_activation=torch.nn.functional.softplus,  # Better numerical stability than exp
        )

        # ----- add mu_glob (shared mu across features) --> INIT FROM DATA? or just N(0,1) nn.Parameter(torch.randn(num_features)) --------
        self.mu_glob = mu_glob
        if self.mu_glob:
            self.mu_vals = nn.Parameter(mu_inits)  #torch.randn(num_features_per_context[0])
        else:
            self.mu_vals = None
        # else:
        #     if len(mu_glob) != num_features:
        #         raise ValueError(f"`mu_glob` must have length {num_features} (events)")
        #     self.mu_glob = mu_glob 


        self.decoders = nn.ModuleDict()
        for context, num_features in zip(contexts, num_features_per_context, strict=False):
            self.decoders[context] = DecoderMETHYLVI(
                n_latent,
                num_features,
                n_cat_list=cat_list,
                n_layers=n_layers_dec,
                n_hidden=n_hidden,
                dropout_rate = dropout_rate_dec,
                linear = lin_decoder,
                mu_glob = mu_glob,
                mu_vals = self.mu_vals,
            )

        if self.dispersion == "region":
            self.px_gamma = torch.nn.ParameterDict(
                {
                    context: nn.Parameter(torch.randn(num_features))
                    for (context, num_features) in zip(
                        contexts, num_features_per_context, strict=False
                    )
                }
            )

        if nu_params is None:
            nu_params = {
                "nu_max": 1.0, # --- initialize as nn.Parameter(torch.randn()) ------
                "m": 1.0,
                "b": 1.0,
            }
        else:
            expected_keys = {"nu_max", "m", "b"} # ------ in future, if given, use values only as inits unless specified ------
            if set(nu_params.keys()) != expected_keys:
                raise ValueError(f"`nu_params` must have keys {expected_keys}, but got {nu_params.keys()}")
            
        self.nu_params = nu_params


    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        mc = torch.cat(
            [tensors[_context_mc_key(context)] for context in self.contexts],
            dim=1,
        )
        cov = torch.cat(
            [tensors[_context_cov_key(context)] for context in self.contexts],
            dim=1,
        )

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = {
            METHYLVI_REGISTRY_KEYS.MC_KEY: mc,
            METHYLVI_REGISTRY_KEYS.COV_KEY: cov,
            "batch_index": batch_index,
            "cat_covs": cat_covs,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = {
            "z": z,
            "batch_index": batch_index,
            "cat_covs": cat_covs,
        }
        return input_dict

    @auto_move_data
    def inference(self, mc, cov, batch_index, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the inputs to the variational distribution for numerical stability
        mc_ = torch.log(1 + mc)
        cov_ = torch.log(1 + cov)

        # get variational parameters via the encoder networks
        # we input both the methylated reads (mc) and coverage (cov)
        methylation_input = torch.cat((mc_, cov_), dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz, z = self.z_encoder(methylation_input, batch_index, *categorical_input)
        if n_samples > 1:
            z = qz.sample((n_samples,))

        outputs = {"z": z, "qz": qz}
        return outputs

    @auto_move_data
    def generative(self, z, batch_index, cat_covs=None):
        """Runs the generative model."""
        # form the parameters of the BetaBinomial likelihood
        px_mu, px_gamma = {}, {}
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        for context in self.contexts:
            px_mu[context], px_gamma[context] = self.decoders[context](
                self.dispersion, z, batch_index, *categorical_input
            )

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {"px_mu": px_mu, "px_gamma": px_gamma, "pz": pz}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Loss function."""
        qz = inference_outputs["qz"]
        pz = generative_outputs["pz"]
        kl_divergence_z = kl(qz, pz).sum(dim=1)

        kl_local_for_warmup = kl_divergence_z

        weighted_kl_local = kl_weight * kl_local_for_warmup

        minibatch_size = qz.loc.size()[0]
        reconst_loss = self._compute_minibatch_reconstruction_loss(
            minibatch_size=minibatch_size,
            tensors=tensors,
            generative_outputs=generative_outputs,
        )
        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {"kl_divergence_z": kl_divergence_z}
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
        )

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> dict[torch.Tensor]:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell

        Returns
        -------
        x_new
            tensor with shape (n_cells, n_regions, n_samples)
        """
        inference_kwargs = {"n_samples": n_samples}
        (
            _,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        exprs = {}
        for context in self.contexts:
            px_mu = generative_outputs["px_mu"][context] 
            px_gamma = generative_outputs["px_gamma"][context]
            cov = tensors[f"{context}_{METHYLVI_REGISTRY_KEYS.COV_KEY}"]

            if self.dispersion == "region":
                px_gamma = torch.sigmoid(self.px_gamma[context])
            elif self.dispersion == "nu":
                a = self.nu_params["m"]*torch.log2(cov +1) + self.nu_params["b"] # a = 1.14*torch.log2(cov +1) - 2.21
                px_gamma = self.nu_params["nu_max"] * (1/(1+torch.exp(-a))) #2.14, nu_max * inv_logit(a)

            if self.likelihood == "binomial":
                dist = Binomial(probs=px_mu, total_count=cov)
            elif self.likelihood == "betabinomial":
                if self.dispersion == "nu":
                    px_gamma = 1 / (px_gamma + 1)
                dist = BetaBinomial(mu=px_mu, gamma=px_gamma, total_count=cov)
                

            if n_samples > 1:
                exprs_ = dist.sample()
                exprs[context] = exprs_.permute(
                    [1, 2, 0]
                ).cpu()  # Shape : (n_cells_batch, n_regions, n_samples)
            else:
                exprs[context] = dist.sample().cpu()

        return exprs
