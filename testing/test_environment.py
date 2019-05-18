from __future__ import absolute_import
import pytest


@pytest.mark.run(order=0)
def test_import():
    import morphs


@pytest.mark.run(order=0)
def test_ols_import():
    from statsmodels.formula.api import ols


if __name__ == "__main__":
    test_import()
