"""
Integrity check for the committed Earth system grids.

The nine ``data/earth_*.npy`` files that ship with the repository are the
reference for all published results. This test pins their SHA-256 so that an
accidental regeneration (``python -m oikoumene.datagen.generate_earth_data``
overwrites them) or any other edit is caught.

Regenerating with the current code and library versions does not reproduce
these files bit for bit: the seven float grids differ by 1-2 ULP in 0.1-2.7 %
of cells; ``biome`` and ``terrain`` are identical. If the reference is changed
on purpose, update the hashes below in the same commit and say why.
"""
import hashlib

import pytest

from oikoumene.paths import DATA_DIR

EXPECTED_SHA256: dict[str, str] = {
    "earth_biome.npy": "b18bcd2bf798b63772e4e06dfd077eca001276adbeaceb0d5fa9998a0c222c09",
    "earth_elevation.npy": "e91a212e85e7e5dd941107d5e43540a2d68dbc22f36958e2419addabfff20ea1",
    "earth_fertility.npy": "31ee29bd3bcac637f8c69fc558caa3040bc489c9a2d8b05a2a425f8a17c42389",
    "earth_fossil_fuels.npy": "dd9eaed6852c7a063d89e5ee672db815944742b0ce993b90e2dade62a4c6506f",
    "earth_freshwater.npy": "d80b443a9b9000c564be0d82ac4ba38b7fff322df377fa742a5fb4068e16678d",
    "earth_minerals.npy": "d7caf0180fb7d9e089096c0a4a0b2fabbcce0de52012ca277065ae50224cb6e7",
    "earth_precipitation.npy": "8c085771dd9c42d02093987c0a8ed92a6aa39f7ec036f9a87b264764daa77a42",
    "earth_temperature.npy": "1670580bc8e30fe246d0fe87e831a548053774d9274fe2509816de7c009bee72",
    "earth_terrain.npy": "07918240b8b45b4556710567470ee03b8b5ac891bfd1560cc9335d7175aab639",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA256))
def test_earth_grid_matches_reference_hash(name: str) -> None:
    path = DATA_DIR / name
    assert path.is_file(), f"{path} is missing"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256[name]
