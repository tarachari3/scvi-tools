from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

from torch import nn
from torch.distributions import Binomial

from scvi.distributions import BetaBinomial
from scvi import REGISTRY_KEYS
from scvi.external.psivi._utils import PSIVI_REGISTRY_KEYS
from scvi.nn import FCLayers

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from mudata import MuData

    from scvi._types import Number

import numpy as np
import pandas as pd
import torch

from scvi import settings
from scvi.model.base._de_core import (
    _de_core,
)

from ._utils import scmc_raw_counts_properties

logger = logging.getLogger(__name__)


class PSIMixin:
    """General purpose methods for splicing analysis."""

    @torch.inference_mode()
    def get_psi(
        self,
        mdata: MuData | None = None,
        indices: Sequence[int] | None = None,
        event_list: Sequence[str] | None = None,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        context: str | None = None,
        **importance_weighting_kwargs,
    ) -> (np.ndarray | pd.DataFrame) | dict[str, np.ndarray | pd.DataFrame]:
        r"""Returns the normalized (decoded) PSI.

        This is denoted as :math:`\mu_n` in the paper.

        Parameters
        ----------
        mdata
            MuData object with equivalent structure to initial Mudata.
            If `None`, defaults to the MuData object used to initialize the model.
        indices
            Indices of cells in mdata to use. If `None`, all cells are used.
        event_list
            Return PSI for subset of events.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`.
            DataFrame includes event names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        context
            If not `None`, returns normalized PSI for the specified
            mudata modality/context. Otherwise, a dictionary with contexts as keys and normalized
            PSIs as values is returned.

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_event).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_events).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_events).

        If model was set up using a MuData object, a dictionary is returned with keys
        corresponding to splicing contexts with values determined as
        described above.
        """
        mdata = self._validate_anndata(mdata)

        if context is not None and context not in self.contexts:
            raise ValueError(
                f"{context} is not a valid mudata modality for this model. "
                f"Valid modalities are {self.contexts}."
            )

        if indices is None:
            indices = np.arange(mdata[self.contexts[0]].n_obs)
        if n_samples_overall is not None:
            assert n_samples == 1  # default value
            n_samples = n_samples_overall // len(indices) + 1
        scdl = self._make_data_loader(adata=mdata, indices=indices, batch_size=batch_size)

        #region_mask = slice(None) if event_list is None else mdata.var_names.isin(event_list)
        region_masks = {
            ctxt: slice(None) if event_list is None else mdata[ctxt].var_names.isin(event_list)
            for ctxt in self.contexts
        }

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        exprs = defaultdict(list)

        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                generative_kwargs={},
                compute_loss=False,
            )

            for ctxt in self.contexts:
                #exp_ = generative_outputs["px_mu"][ctxt] #exp_ = exp_[..., region_mask]
                exp_ = generative_outputs["px_mu"][ctxt][..., region_masks[ctxt]]
                exprs[ctxt].append(exp_.cpu())

        cell_axis = 1 if n_samples > 1 else 0

        for ctxt in self.contexts:
            exprs[ctxt] = np.concatenate(exprs[ctxt], axis=cell_axis)

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            for ctxt in self.contexts:
                exprs[ctxt] = exprs[ctxt].reshape(-1, exprs[ctxt].shape[-1])
                n_samples_ = exprs[ctxt].shape[0]
                ind_ = np.random.choice(n_samples_, n_samples_overall, replace=True)
                exprs[ctxt] = exprs[ctxt][ind_]
                return_numpy = True

        elif n_samples > 1 and return_mean:
            for ctxt in self.contexts:
                exprs[ctxt] = exprs[ctxt].mean(0)

        if return_numpy is None or return_numpy is False:
            exprs_dfs = {}
            for ctxt in self.contexts:
                exprs_dfs[ctxt] = pd.DataFrame(
                    exprs[ctxt],
                    columns=mdata[ctxt].var_names[region_masks[ctxt]],
                    index=mdata[ctxt].obs_names[indices],
                )
            exprs_ = exprs_dfs
        else:
            exprs_ = exprs

        if context is not None:
            return exprs_[context]
        else:
            return exprs_

    @torch.inference_mode()
    def get_specific_psi(
        self,
        mdata: MuData | None = None,
        context: str = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        event_list: Sequence[str] | None = None,
        n_samples: int = 1,
        n_samples_overall: int = None,
        weights: Literal["uniform", "importance"] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        **importance_weighting_kwargs,
    ) -> (np.ndarray | pd.DataFrame) | dict[str, np.ndarray | pd.DataFrame]:
        r"""Convenience function to obtain normalized methylation values for a single context.

        Parameters
        ----------
        mdata
            MuData object with equivalent structure to initial MuData. If `None`, defaults to the
            MuData object used to initialize the model.
        context
            Mudata modality for which to obtain normalized methylation levels.
        indices
            Indices of cells in mdata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        event_list
            Return PSI for a subset of events.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        weights
            Weights to use for sampling. If `None`, defaults to `"uniform"`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`.
            DataFrame includes event names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        importance_weighting_kwargs
            Keyword arguments passed into
            :meth:`~scvi.model.base.RNASeqMixin._get_importance_weights`.

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_events).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_event).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_events).
        """
        exprs = self.get_psi(
            mdata=mdata,
            indices=indices,
            transform_batch=transform_batch,
            event_list=event_list,
            n_samples=n_samples,
            n_samples_overall=n_samples_overall,
            weights=weights,
            batch_size=batch_size,
            return_mean=return_mean,
            return_numpy=return_numpy,
            **importance_weighting_kwargs,
        )
        return exprs[context]

    def differential_psi(
        self,
        mdata: MuData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | str | None = None,
        idx2: Sequence[int] | Sequence[bool] | str | None = None,
        mode: Literal["vanilla", "change"] = "vanilla",
        delta: float = 0.05,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        two_sided: bool = True,
        **kwargs,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        r"""\.

        A unified method for differential inclusion/PSI analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18`. and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_mdata)s
        %(de_modality)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        two_sided
            Whether to perform a two-sided test, or a one-sided test.
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential inclusion DataFrame with the following columns:
        proba_de
            the probability of the event being differentially included
        is_de_fdr
            whether the event passes a multiple hypothesis correction procedure
            with the target_fdr threshold
        bayes_factor
            Bayes Factor indicating the level of significance of the analysis
        effect_size
            the effect size, computed as (PSI in population 2) -
            (PSI in population 1)
        emp_effect
            the empirical effect, based on observed detection rates instead of the estimated
            PSIs from the psiVI model
        scale1
            the estimated PSI in population 1
        scale2
            the estimated PSI in population 2
        emp_mean1
            the empirical (observed) PSI in population 1
        emp_mean2
            the empirical (observed) PSI in population 2

        """
        mdata = self._validate_anndata(mdata)
        kwargs["pseudocounts"] = 0

        def change_fn(a, b,pseudocounts=None): # need to use same change-based DE as scvi
            return a - b


        if two_sided:
            def m1_domain_fn(samples):
                samples_plus = samples >= delta
                samples_minus = samples < -delta
                return samples_plus, samples_minus
        else:
            def m1_domain_fn(samples):
                samples_plus = samples >= delta
                samples_minus = np.zeros_like(samples_plus, dtype=bool)
                return samples_plus, samples_minus

        # Temporarily add context's obs columns on the global mdata.obs
        added_cols = []
        for ctxt in self.contexts:
            mod_obs = mdata[ctxt].obs
            for col in mod_obs.columns:
                if col not in mdata.obs.columns:
                    # align by cell barcode
                    mdata.obs[col] = mod_obs[col].reindex(mdata.obs_names).values
                    added_cols.append(col)
        try:
            result = {}
            for context in self.contexts:
                col_names = mdata[context].var_names
                model_fn = partial(
                    self.get_specific_psi,
                    batch_size=batch_size,
                    context=context,
                )
                all_stats_fn = partial(scmc_raw_counts_properties, context=context)

                result[context] = _de_core(
                    adata_manager=self.get_anndata_manager(mdata, required=True),
                    model_fn=model_fn,
                    representation_fn=None,
                    groupby=groupby,
                    group1=group1,
                    group2=group2,
                    idx1=idx1,
                    idx2=idx2,
                    all_stats=all_stats,
                    all_stats_fn=all_stats_fn,
                    col_names=col_names,
                    mode=mode,
                    batchid1=batchid1,
                    batchid2=batchid2,
                    delta=delta,
                    batch_correction=batch_correction,
                    fdr=fdr_target,
                    silent=silent,
                    change_fn=change_fn,
                    m1_domain_fn=m1_domain_fn,
                    **kwargs,
                )
        finally:
            # clean up, even if DE fails
            mdata.obs.drop(columns=added_cols, inplace=True)

        return result
    
    @torch.inference_mode()
    def get_nu_params(self, as_dataframe: bool = True):
        """Learned capture-dispersion parameters (one value per batch).

        ``nu_max`` is log scale (the model uses ``exp(nu_max)``), so df has ``exp_nu_max``.
        """
        if self.module.dispersion != "capture":
            warnings.warn(
                f"dispersion='{self.module.dispersion}': nu_params are not used by this model.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        nu = {k: v.detach().cpu().numpy() for k, v in self.module.nu_params.items()}
        if not as_dataframe:
            return nu

        n_nu = len(nu["m"])
        batches = list(
            self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
        )
        if len(batches) != n_nu:
            batches = [str(i) for i in range(n_nu)]

        df = pd.DataFrame(nu, index=pd.Index(batches, name="batch"))
        df["exp_nu_max"] = np.exp(df["nu_max"])
        return df

    @torch.inference_mode()
    def get_px_gamma(
        self,
        mdata: MuData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        context: str | None = None,
        return_numpy: bool = False,
    ):
        """Dispersion parameter ``px_gamma`` per context.

        For ``dispersion='capture'`` this is ``exp(nu_max) * sigmoid(m * log2(cov + 1) + b)``,
        i.e. *before* the ``1 / (x + 1)`` transform used inside the BetaBinomial.

        Returns
        -------
        Dict keyed by context (or a single object if ``context`` is given).
        Shape is (n_events,) for ``dispersion='event'`` and (n_cells, n_events) otherwise.
        """
        mdata = self._validate_anndata(mdata)
        if context is not None and context not in self.contexts:
            raise ValueError(f"{context} not in model contexts {self.contexts}.")
        contexts = [context] if context is not None else list(self.contexts)

        if indices is None:
            indices = np.arange(mdata[self.contexts[0]].n_obs)

        out = {}
        if self.module.dispersion == "event":
            for ctxt in contexts:
                vals = torch.sigmoid(self.module.px_gamma[ctxt]).cpu().numpy()
                out[ctxt] = (
                    vals if return_numpy
                    else pd.Series(vals, index=mdata[ctxt].var_names, name="px_gamma")
                )
            return out[context] if context is not None else out

        scdl = self._make_data_loader(adata=mdata, indices=indices, batch_size=batch_size)
        chunks = defaultdict(list)
        for tensors in scdl:
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs={"n_samples": 1},
                generative_kwargs={},
                compute_loss=False,
            )
            for ctxt in contexts:
                g = self.module.compute_px_gamma(ctxt, tensors, generative_outputs)
                chunks[ctxt].append(g.cpu())

        for ctxt in contexts:
            arr = torch.cat(chunks[ctxt], dim=0).numpy()
            out[ctxt] = (
                arr if return_numpy
                else pd.DataFrame(
                    arr,
                    index=mdata[ctxt].obs_names[indices],
                    columns=mdata[ctxt].var_names,
                )
            )
        return out[context] if context is not None else out

    @torch.inference_mode()
    def get_decoder_weights(
        self,
        mdata: MuData | None = None,
        context: str | None = None,
        which: Literal["mu", "gamma"] = "mu",
        as_dataframe: bool = True,
    ):
        """Linear decoder weights W, shape (n_events, n_latent).

        Only defined when the model was trained with ``lin_decoder=True``.
        ``which="mu"`` gives the PSI decoder (``psi = sigmoid(W z + mu_vals)`` when a global
        mean is used). ``which="gamma"`` gives the dispersion decoder, which is only
        used when ``dispersion='event-cell'``.
        """
        mdata = self._validate_anndata(mdata)
        context = self.contexts[0] if context is None else context
        if context not in self.contexts:
            raise ValueError(f"{context} not in model contexts {self.contexts}.")

        dec = self.module.decoders[context]
        if not dec.linear:
            raise ValueError("Decoder weights W are only available with lin_decoder=True.")

        layer = dec.px_mu_decoder[0] if which == "mu" else dec.px_gamma_decoder[0]
        W = layer.weight.detach().cpu().numpy()  # (n_events, n_latent)

        if not as_dataframe:
            return W
        return pd.DataFrame(
            W,
            index=mdata[context].var_names,
            columns=[f"z_{i}" for i in range(W.shape[1])],
        )


class PSIModuleMixin:
    """Shared methods for splicing VAE modules."""

    data_input_keys = [PSIVI_REGISTRY_KEYS.MC_KEY, PSIVI_REGISTRY_KEYS.COV_KEY]
    def _nu_params_for_cells(self, tensors):
        """Return (nu_max, m, b), broadcastable against (n_cells, n_events)."""
        names = ("nu_max", "m", "b")
        if self.n_batch == 0:
            return tuple(self.nu_params[k][0] for k in names)

        batch_index = (
            tensors[REGISTRY_KEYS.BATCH_KEY]
            .squeeze(-1)
            .long()
            .to(self.nu_params["m"].device)
        )
        return tuple(self.nu_params[k][batch_index].unsqueeze(-1) for k in names)

    def compute_px_gamma(self, context, tensors, generative_outputs):
        """Dispersion parameter for one context, before any transform for the likelihood.

        Returns
        -------
        - ``dispersion == "event"``: sigmoid(raw per-event param), shape (n_events,)
        - ``dispersion == "event-cell"``: decoder output, shape ([n_samples,] n_cells, n_events)
        - ``dispersion == "capture"``: ``exp(nu_max) * sigmoid(m * log2(cov + 1) + b)``,
          shape (n_cells, n_events)
        """
        if self.dispersion == "event":
            return torch.sigmoid(self.px_gamma[context])

        if self.dispersion == "event-cell":
            return generative_outputs["px_gamma"][context]

        if self.dispersion == "capture":
            nu_max, m, b = self._nu_params_for_cells(tensors)
            cov = tensors[f"{context}_{PSIVI_REGISTRY_KEYS.COV_KEY}"].to(m.device)
            a = m * torch.log2(cov + 1) + b
            return torch.exp(nu_max) * torch.sigmoid(a)

        raise ValueError(f"Unknown dispersion: {self.dispersion}")

    def _compute_minibatch_reconstruction_loss(self, minibatch_size, tensors, generative_outputs):
        reconst_loss = torch.zeros(minibatch_size).to(self.device)
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)  # (n_cells,)


        for context in self.contexts:
            px_mu = generative_outputs["px_mu"][context]
            px_gamma = generative_outputs["px_gamma"][context]
            mc = tensors[f"{context}_{PSIVI_REGISTRY_KEYS.MC_KEY}"]
            cov = tensors[f"{context}_{PSIVI_REGISTRY_KEYS.COV_KEY}"]

            px_gamma = self.compute_px_gamma(context, tensors, generative_outputs)

            if self.likelihood == "betabinomial":
                if self.dispersion == "capture":
                    px_gamma = 1 / (px_gamma + 1)
                dist = BetaBinomial(mu=px_mu, gamma=px_gamma, total_count=cov)

            reconst_loss += -dist.log_prob(mc).sum(dim=-1)


        return reconst_loss


class DecoderPSIVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        linear: bool = False,
        mu_glob: bool = False,
        mu_vals: float = None,
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        self.linear = linear 
        self.mu_glob = mu_glob
        self.mu_vals = mu_vals

        if self.linear:
            if self.mu_glob: #No sigmoid() yet
                    self.px_mu_decoder = nn.Sequential(
                    nn.Linear(n_input, n_output,bias=False),
                )
            else:
                self.px_mu_decoder = nn.Sequential(
                    nn.Linear(n_input, n_output,bias=False),
                    nn.Sigmoid(),
                )

            self.px_gamma_decoder = nn.Sequential(
                nn.Linear(n_input, n_output,bias=False),
                nn.Sigmoid(),
            )
        else:
            if self.mu_glob:
                self.px_mu_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output),
                )
            else:
                self.px_mu_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output),
                    nn.Sigmoid(),
                )
            self.px_gamma_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_output),
                nn.Sigmoid(),
            )

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        *cat_list: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the beta-binomial distribution of methylation
         #. If ``dispersion != 'event-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'event'`` - dispersion parameter constant per event across cells
            * ``'event-cell'`` - dispersion can differ for every event in every cell
            * ``'capture'`` - capture-based concentration param per event in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            parameters for the Beta distribution of mean PSI values

        """
        px = self.px_decoder(z, *cat_list)
        
        if self.linear:
            px_mu = self.px_mu_decoder(z)
            if self.mu_glob:
                px_mu = torch.sigmoid(px_mu + self.mu_vals)
                
            px_gamma = self.px_gamma_decoder(z) if dispersion == "event-cell" else None
        else:
            px_mu = self.px_mu_decoder(px)
            if self.mu_glob:
                px_mu = torch.sigmoid(px_mu + self.mu_vals)

            px_gamma = self.px_gamma_decoder(px) if dispersion == "event-cell" else None

        return px_mu, px_gamma
