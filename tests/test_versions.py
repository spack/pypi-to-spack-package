import pytest
from spack.version import Version

import package


def test_best_lowerbound():
    assert package._best_lowerbound(Version("1.0"), Version("2.0")) == Version("2")
    assert package._best_lowerbound(Version("1.0"), Version("1.1.1")) == Version("1.1")
    assert package._best_lowerbound(Version("1.0"), Version("1.0.1")) == Version("1.0.1")
    assert package._best_lowerbound(Version("1.0a1"), Version("1.0")) == Version("1.0")
    assert package._best_lowerbound(Version("1.0a1"), Version("1.0b2")) == Version("1.0b2")


def test_best_upperbound():
    assert package._best_upperbound(Version("1.0"), Version("2.0")) == Version("1")
    assert package._best_upperbound(Version("1.0b1"), Version("1.0")) == Version("1.0b1")