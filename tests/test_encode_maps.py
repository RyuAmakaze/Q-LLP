import sys
import os
import sys
import pytest

qiskit = pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from encode import get_feature_map
from qiskit.circuit.library import ZZFeatureMap


def test_get_feature_map_returns_correct_class():
    fm = get_feature_map("zz", 2)
    assert isinstance(fm, ZZFeatureMap)


def test_get_feature_map_invalid():
    with pytest.raises(ValueError):
        get_feature_map("unknown", 2)

