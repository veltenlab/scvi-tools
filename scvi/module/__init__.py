from ._autozivae import AutoZIVAE
from ._classifier import Classifier
from ._mrdeconv import MRDeconv
from ._peakvae import PEAKVAE
from ._scanvae import SCANVAE
from ._totalvae import TOTALVAE
from ._vae import LDVAE, VAE
from ._vaec import VAEC
from ._methvae import MethVAE

__all__ = [
    "VAE",
    "LDVAE",
    "TOTALVAE",
    "AutoZIVAE",
    "SCANVAE",
    "Classifier",
    "PEAKVAE",
    "VAEC",
    "MRDeconv",
    "MethVAE"
]
