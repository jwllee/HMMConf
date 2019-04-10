import logging, logging.config
from scipy.special import logsumexp
import numpy as np


np.set_printoptions(precision=10)


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
            'formatter': 'standard'
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


def assert_bounded(name, a, lower, upper):
    msg = """Values of {name} are not in [{lower},{upper}]
             [array]: {a}
             <{lower}: {less}
             >{upper}: {more}"""
    less = np.sum(a < lower, axis=None)
    more = np.sum(a > upper, axis=None)
    if a.size == 1:
        msg = msg.format(name=name, a=np.ravel(a)[0], 
                        lower=lower, upper=upper,
                        less=less, more=more)
    else:
        msg = msg.format(name=name, a=a, 
                        lower=lower, upper=upper,
                        less=less, more=more)
    bounded = (a >= lower).all() and (a <= upper).all()
    assert bounded.all(), msg


def assert_no_negatives(name, a):
    msg = """{} negative values in {}, expected 0:
             [array]:   {}"""
    n_neg = np.count_nonzero(a < 0)
    msg = msg.format(n_neg, name, a)
    assert n_neg == 0, msg


def assert_shape(name, expected, actual):
    msg = """Shape difference in {name}
             [expected]: {expected}
             [actual]:   {actual}"""
    msg = msg.format(name=name, 
                     expected=expected, 
                     actual=actual)
    assert expected == actual, msg


def make_logger(name):
    return logging.getLogger(name)


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

    # summed = a.sum(axis)
    # almost_one = np.isclose(summed, np.ones(summed.shape))
    # msg = 'Normalize not summing close to 1: {}'
    # msg = msg.format(summed[np.invert(almost_one)])
    # assert almost_one.all(), msg


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

    # check it's actually normalized
    # summed = logsumexp(a, axis=axis)
    # almost_one = np.isclose(summed, np.ones(summed.shape))
    # msg = 'Log normalize not summing close to 1: {}'
    # msg = msg.format(summed[np.invert(almost_one)])
    # assert almost_one.all(), msg

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
