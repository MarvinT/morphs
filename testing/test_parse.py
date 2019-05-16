from __future__ import absolute_import
import morphs
import pytest


@pytest.mark.run(order=0)
def test_p_value():
    assert morphs.data.parse.p_value(.0499) == "*"
    assert morphs.data.parse.p_value(.0008) == "***"
    assert morphs.data.parse.p_value(.051) == ""
    assert morphs.data.parse.p_value(.04, bonferroni_n=2) == ""
    assert morphs.data.parse.p_value(.02, bonferroni_n=2) == "*"
    assert morphs.data.parse.p_value(.01, bonferroni_n=8) == ""
