import numpy as np
from hmmconf.utils import *


def test_assert_ndim():
    a = np.random.rand(3)
    assert_ndim('Test 1', 1, a.ndim)
