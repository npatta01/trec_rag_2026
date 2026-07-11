import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_V2_1 = REPO_ROOT / "reports" / "experiments" / "query_planner_v2_1_synthetic_smoke_001"


def test_new_deterministic_arm_cannot_silently_mutate_v2_1_ledger():
    expected = {
        "artifacts/_run.json": "a705b8ab28d09d43e6e73c04929409e4dfe90badbe64775e0591605cb7a9dd9c",
        "artifacts/outcome.json": "6f1842f038e17d7464110dc8b4e3161abb93e87902a94aea1150a80b49f198ac",
        "artifacts/raw_response.json": "81a0a5be70491d9bc896eb1ac4af1943224189470665876a3fb3debdd0bbb3f4",
        "artifacts/plans.manifest.json": "c1d71b5568ab53ab09669c9b05d6b9e72e2061cb9174e73f4be0f8758a7cb150",
    }

    actual = {
        relative: hashlib.sha256((FROZEN_V2_1 / relative).read_bytes()).hexdigest()
        for relative in expected
    }

    assert actual == expected
