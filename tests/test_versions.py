import pytest
from spack.version import Version

import package


def test_best_lowerbound():
    package._best_lowerbound(Version("1.0"), Version("2.0")) == Version("2")
    package._best_lowerbound(Version("1.0"), Version("1.1.1")) == Version("1.1")
    package._best_lowerbound(Version("1.0"), Version("1.0.1")) == Version("1.0.1")
    package._best_lowerbound(Version("1.0a1"), Version("1.0")) == Version("1.0")
    package._best_lowerbound(Version("1.0a1"), Version("1.0b2")) == Version("1.0b2")
