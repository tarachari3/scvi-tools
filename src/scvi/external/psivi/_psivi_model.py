from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from mudata import MuData

import numpy as np
import sparse
import torch

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.external.psivi._base_components import PSIMixin
from scvi.external.psivi._utils import _context_cov_key, _context_mc_key
from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.utils import setup_anndata_dsp

from ._psivi_module import PSIVAE

logger = logging.getLogger(__name__)


class PSIVI(VAEMixin, PSIMixin, UnsupervisedTrainingMixin, ArchesMixin, BaseModelClass):
    """
    Model class for psiVI

    Parameters
    ----------
    mdata
        MuData object that has been registered via :meth:`~scvi.external.PSIVI.setup_mudata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers_enc
        Number of hidden layers used for encoder NNs.
    n_layers_dec
        Number of hidden layers used for decoder NNs.
    tech_only_batch
        If ``True``, only allow batch to influence dispersion and library size parameters. Otherwise default behavior.
    **model_kwargs
        Keyword args for :class:`~scvi.external.psivi.PSIVAE`

    Examples
    --------
    >>> mdata = mudata.read_h5mu(path_to_mudata)
    >>> PSIVI.setup_mudata(mdata, batch_key="batch")
    >>> vae = PSIVI(mdata)
    >>> vae.train()
    >>> mdata.obsm["X_psiVI"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        mdata: MuData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_enc: int = 1,
        n_layers_dec: int = 1,
        tech_only_batch: bool = False,
        **model_kwargs,
    ):
        super().__init__(mdata)

        n_batch = self.summary_stats.n_batch
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)[
                fields.CategoricalJointObsField.N_CATS_PER_KEY
            ]
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.contexts = self.get_anndata_manager(mdata, required=True).registry[_SETUP_ARGS_KEY][
            "mudata_mod"
        ]
        self.num_features_per_context = [mdata[context].shape[1] for context in self.contexts]
        self.get_normalized_function_name = "get_psi"

        n_input = np.sum(self.num_features_per_context)

        # -- initialize mu with data ---
        ctxt = self.contexts[0]
        mc_attr = self.adata_manager.data_registry[_context_mc_key(ctxt)]
        cov_attr = self.adata_manager.data_registry[_context_cov_key(ctxt)]

        inc_layer = mc_attr.attr_key   # e.g. "inc"
        cov_layer = cov_attr.attr_key  # e.g. "cov"
        inc = mdata[ctxt].layers[inc_layer]
        cov = mdata[ctxt].layers[cov_layer]
        if hasattr(inc, "todense"):
            inc = inc.todense()
        if hasattr(cov, "todense"):
            cov = cov.todense()
        inc = np.asarray(inc, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)


        mean_ps = np.nanmean((inc/cov),axis=0)
        mean_ps = np.clip(mean_ps,1e-6,1-1e-6)
        mu_inits = np.log(mean_ps/(1-mean_ps))


        self.module = PSIVAE(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_enc=n_layers_enc,
            n_layers_dec=n_layers_dec,
            n_batch=n_batch,
            n_cats_per_cov=n_cats_per_cov,
            contexts=self.contexts,
            num_features_per_context=self.num_features_per_context,
            tech_only_batch = tech_only_batch,
            mu_inits = torch.tensor(mu_inits, dtype=torch.float32),
            **model_kwargs,
        )
        self._model_summary_string = (
            "Overwrite this attribute to get an informative representation for your model"
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        **kwargs,
    ) -> AnnData | None:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s

        Returns
        -------
        %(returns)s
        """
        raise NotImplementedError("PSIVI must be used with a MuData object.")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        inc_layer: str,
        cov_layer: str,
        mudata_mod: str,
        batch_key: str | None = None,
        user_z_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        modalities=None,
        **kwargs,
    ):
        """%(summary_mdata)s.

        Parameters
        ----------
        %(param_mdata)s
        inc_layer
            Layer containing inclusion counts for events or features.
        cov_layer
            Layer containing total coverage counts for each event or feature.
        mudata_mod
            Modality field in `mdata` object to be used for psi inference.
            A mod has layer containing the number of inclusion counts
            (specified by `inc_layer`) and total number of counts (specified by `cov_layer`) for
            each event or feature.
        %(param_batch_key)s
        %(param_cat_cov_keys)s
        %(param_modalities)s

        Examples
        --------
        PSIVI.setup_mudata(
            mdata,
            inc_layer="inc",
            cov_layer="cov",
            batch_key="sample",
            mudata_mod='SJ',
            user_z_key='X_scvi',
            modalities={
                "batch_key": "SJ",
                "user_z_key": "GEX"
            },
        )

        """
        if modalities is None:
            modalities = {}


        if mudata_mod is None:
            raise ValueError("A mod name for the mudata must be entered.")
        
        if isinstance(mudata_mod, str):
            mudata_mod = [mudata_mod]
        else:
            mudata_mod = list(mudata_mod)
        
        setup_method_args = PSIVI._get_setup_method_args(**locals())

        modalities_ = cls._create_modalities_attr_dict(modalities, setup_method_args)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities_.batch_key,
        )

        cat_cov_field = fields.MuDataCategoricalJointObsField(
            REGISTRY_KEYS.CAT_COVS_KEY,
            categorical_covariate_keys,
            mod_key=modalities_.categorical_covariate_keys,
        )


        mc_fields = []
        cov_fields = []

        for context in mudata_mod:
            mc_fields.append(
                fields.MuDataLayerField(
                    _context_mc_key(context),
                    inc_layer,
                    mod_key=context,
                    is_count_data=True,
                    mod_required=True,
                )
            )

            cov_fields.append(
                fields.MuDataLayerField(
                    _context_cov_key(context),
                    cov_layer,
                    mod_key=context,
                    is_count_data=True,
                    mod_required=True,
                )
            )

        cont_cov_field = None
        if user_z_key is not None:
            cont_cov_field = fields.MuDataObsmField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                user_z_key, # X_scvi
                mod_key=modalities_.user_z_key
            )

        mudata_fields = mc_fields + cov_fields + [batch_field] + [cat_cov_field]
        if cont_cov_field is not None:
            mudata_fields = mudata_fields + [cont_cov_field]
        
        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)

        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def posterior_predictive_sample(
        self,
        mdata: MuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> dict[str, sparse.GCXS] | sparse.GCXS:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        mdata
            MuData object with equivalent structure to initial MuData. If `None`, defaults to the
            MuData object used to initialize the model.
        n_samples
            Number of samples for each cell.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_regions, n_samples)
        """
        mdata = self._validate_anndata(mdata)

        scdl = self._make_data_loader(adata=mdata, batch_size=batch_size)

        x_new = defaultdict(list)
        for tensors in scdl:
            samples = self.module.sample(
                tensors,
                n_samples=n_samples,
            )

            for context in self.contexts:
                x_new[context].append(sparse.GCXS.from_numpy(samples[context].numpy()))

        for context in self.contexts:
            x_new[context] = sparse.concatenate(
                x_new[context]
            )  # Shape (n_cells, n_regions, n_samples)

        return x_new
