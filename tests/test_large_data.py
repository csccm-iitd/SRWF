import urllib.request

import numpy as np
import pytest

from srwf.data import large


def _online() -> bool:
    try:
        urllib.request.urlopen(large.BASE_URL.split("/raw/")[0], timeout=10)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online() and not (large.data_root() / "pol" / "data.npy").exists(), reason="offline")
def test_pol_splits_are_fixed_90_10_partitions():
    n, d = large.DATASETS["pol"]
    xy, mask = large.load("pol")
    assert xy.shape == (n, d + 1) and mask.shape == (n, 10)
    s = large.split("pol", 3)
    assert s.x_train.shape[1] == d and len(s.y_train) + len(s.y_test) == n
    assert abs(len(s.y_test) / n - 0.1) < 0.01
    np.testing.assert_array_equal(s.y_test, np.asarray(xy)[np.asarray(mask[:, 3]), -1])
