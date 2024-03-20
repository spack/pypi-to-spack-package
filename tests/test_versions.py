import pytest
from packaging.version import Version as PVersion
from spack.version import Version as SVersion
from spack.version import ver

import package


def test_best_lowerbound():
    assert package._best_lowerbound(SVersion("1.0"), SVersion("2.0")) == SVersion("2")
    assert package._best_lowerbound(SVersion("1.0"), SVersion("1.1.1")) == SVersion("1.1")
    assert package._best_lowerbound(SVersion("1.0"), SVersion("1.0.1")) == SVersion("1.0.1")
    assert package._best_lowerbound(SVersion("1.0a1"), SVersion("1.0")) == SVersion("1.0")
    assert package._best_lowerbound(SVersion("1.0a1"), SVersion("1.0b2")) == SVersion("1.0b2")


def test_best_upperbound():
    assert package._best_upperbound(SVersion("1.0"), SVersion("2.0")) == SVersion("1")
    assert package._best_upperbound(SVersion("1.0b1"), SVersion("1.0")) == SVersion("1.0b1")


def _pversions(*versions):
    return [PVersion(v) for v in versions]


def test_condensed_versions():
    all_versions = _pversions("1.0", "2.0", "3.0", "3.1", "4.0", "5.0")
    condense = package._condensed_version_list
    assert condense(_pversions("2.0", "4.0", "5.0"), all_versions) == ver(["2,4:"])
    assert condense(_pversions("2.0", "4.0"), all_versions) == ver(["2,4"])
    assert condense(_pversions("2.0", "3.0", "3.1", "4.0"), all_versions) == ver(["2:4"])
    assert condense(_pversions("1.0", "2.0"), all_versions) == ver([":2"])
    assert condense(all_versions, all_versions) == ver([":"])


def test_condensed_versions_with_prereleases():
    all_versions = _pversions("1.0", "2.0a1", "2.0b2", "2.0")
    condense = package._condensed_version_list
    assert condense(_pversions("1.0", "2.0a1"), all_versions) == ver([":2.0a1"])
    assert condense(_pversions("2.0"), all_versions) == ver(["2.0:"])
