# v1 Sunset Checklist

This file documents the sunset gate for the v1→v2 migration.

## Prerequisite: 7-day soak

Before running any deletion commands:
- [ ] Both `v1-tier{1..5}` and `v2-tier{1..5}` CI jobs pass for 7 consecutive days
- [ ] Zero v2-only flakes in that window
- [ ] All v1 tier{1..5} tests appear in `parity_coverage_report()` (run: `python -c "from ci.software_only.infra.parity import parity_coverage_report; print(parity_coverage_report())"`)

## What to delete

After the 7-day soak is confirmed, run the following from the repository root.

### 1. Delete the v1 software-only test tree

```bash
git rm -r control/src/ci/software_only/
```

### 2. Delete redundant v1 fixture modules from ci/fixtures/

```bash
git rm \
  control/src/ci/fixtures/factories.py \
  control/src/ci/fixtures/rsync_fixtures.py \
  control/src/ci/fixtures/data_fixtures.py \
  control/src/ci/fixtures/chaos_fixtures.py \
  control/src/ci/fixtures/workspace_fixtures.py \
  control/src/ci/fixtures/transfer_fixtures.py \
  control/src/ci/fixtures/state_probe.py \
  control/src/ci/fixtures/client_fixtures.py \
  control/src/ci/fixtures/mocks.py \
  control/src/ci/fixtures/fleet.py \
  control/src/ci/fixtures/network_fixtures.py
```

**DO NOT delete** `control/src/ci/fixtures/topology_fixtures.py`:
hardware_software/ imports `ObservatoryTopology` from it in 5 files.
Either keep it or migrate those imports to a hw-sw-specific module first.

### 3. Delete the original chaos modules (now in software_only/fixtures/chaos/)

```bash
git rm -r control/src/ci/fixtures/chaos/
```

### 4. Update shared/ imports (if any still reference the deleted modules)

```bash
grep -r "from ci\.fixtures\." control/src/ci/shared/ | grep -v "__pycache__"
```

Fix any remaining imports before committing.

### 5. Commit

```bash
git commit -m "chore: sunset v1 software_only test suite

v2 has passed 7-day soak with zero flakes. All v1 tier{1..5} tests
have passing v2 parity equivalents. Deleting v1 tree and redundant
fixture modules.

hardware_software/ is untouched: topology_fixtures.py is preserved.
"
```

## What to keep forever

- `control/src/ci/fixtures/topology_fixtures.py` — used by hardware_software/
- `control/src/ci/fixtures/__init__.py` — package marker
- `control/src/ci/hardware_software/` — entirely untouched
- `control/src/ci/shared/` — transfer helpers used by tier5
- `control/src/ci/fixtures/configs/` — static config files (used by both hw-sw and tier5)
