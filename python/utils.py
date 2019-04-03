import logging, logging.config
from scipy.special import logsumexp
import numpy as np


# set standard logging configurations
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
})


class Logged:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)


def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    :param a: non-normalized input array
    :param axis: dimension along which normalization is to be performed
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.

    :param a: non-normalized input array
    :param axis: dimension along which normalization is to be performed
    """
    with np.errstate(under="ignore"):
        a_lse = logsumexp(a, axis)
        # print('a: {}'.format(a))
        # print('logsumexp: {}'.format(a_lse))
    a -= a_lse[:, np.newaxis]


def iter_from_X_lengths(X, lengths):
    """Iterate through the start and end indexes of subsequences in input array where the subsequences are explicitly specified.

    :param X: input array
    :param lengths: the lengths of the subsequences
    """
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError('More than {:d} samples in lengths array {!s}'.format(n_samples, lengths))
    for i in range(len(lengths)):
        yield start[i], end[i]


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.

    """
    a = np.asarray(a)
    with np.errstate(divide='ignore'):
        return np.log(a)
