"""
fixtures/corpus.py — pff_corpus session fixture.

Returns a PFFCorpus instance backed by the Lick observatory package data
(or PSETI_V2_CORPUS_PATH / qa.toml override).  Session-scoped so discovery
runs once per pytest session.
"""

from __future__ import annotations

import pytest

from ci.software_only_v2.infra.corpus import PFFCorpus


@pytest.fixture(scope="session")
def pff_corpus() -> PFFCorpus:
    """Session-scoped PFFCorpus backed by the panoseti_grpc package data corpus."""
    return PFFCorpus()
