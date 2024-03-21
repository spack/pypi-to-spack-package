import pytest
from packaging.markers import Marker
from packaging.version import Version as PVersion
from spack.spec import Spec
from spack.version import Version as SVersion
from spack.version import ver

import package


def test_best_lowerbound():
    assert package._best_lowerbound(SVersion("1.0"), SVersion("2.0")) == SVersion("2")
    assert package._best_lowerbound(SVersion("1.0"), SVersion("1.1.1")) == SVersion("1.1")
    assert package._best_lowerbound(SVersion("1.0"), SVersion("1.0.1")) == SVersion("1.0.1")
    assert package._best_lowerbound(SVersion("1.0a1"), SVersion("1.0")) == SVersion("1.0")
    assert package._best_lowerbound(SVersion("1.0a1"), SVersion("1.0beta2")) == SVersion(
        "1.0beta2"
    )


def test_best_upperbound():
    assert package._best_upperbound(SVersion("1.0"), SVersion("2.0")) == SVersion("1")
    assert package._best_upperbound(SVersion("1.0beta1"), SVersion("1.0")) == SVersion("1.0beta1")


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
    assert condense(_pversions("1.0", "2.0a1"), all_versions) == ver([":2.0-alpha1"])
    assert condense(_pversions("2.0"), all_versions) == ver(["2.0:"])


@pytest.mark.parametrize(
    "marker, expected",
    [
        ("python_version < '3.9'", [Spec("^python@:3.8")]),
        ("python_version < '3.9' and extra == 'foo'", [Spec("+foo ^python@:3.8")]),
        ("python_version < '3.9' or extra == 'foo'", [Spec("+foo"), Spec("^python@:3.8")]),
        (
            "python_version <= '3.9' and python_version > '3.8.4' or extra == 'foo' and extra == 'bar'",
            [Spec("+bar+foo"), Spec("^python@3.8.5:3.9.0")],
        ),
        (
            "python_version <= '3.9' and (python_version > '3.8.4' or extra == 'foo') and extra == 'bar'",
            [Spec("+bar ^python@3.8.5:3.9.0"), Spec("+bar +foo ^python@:3.9.0")],
        ),
        # python_version >= 3.6 should be statically evaluated as true since older is unsupported.
        ("python_version >= '3.6'", True),
        ("python_version >= '3.6' and extra == 'foo'", [Spec("+foo")]),
        ("python_version >= '3.6' or extra == 'foo'", True),
        ("python_version >= '3.6' and python_version <= '3.10'", [Spec("^python@:3.10.0")]),
    ],
)
def test_marker(marker, expected):
    lookup = package.VersionsLookup(None)
    out = package._evaluate_marker(Marker(marker), lookup)
    if expected in (True, False, None):
        assert out == expected
    else:
        assert set(out) == set(expected)
