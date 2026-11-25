from typing import NamedTuple


class _PSIVI_REGISTRY_KEYS_NT(NamedTuple):
    MC_KEY: str = "mc"
    COV_KEY: str = "cov"


PSIVI_REGISTRY_KEYS = _PSIVI_REGISTRY_KEYS_NT()