# PyAutoCTI Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoCTI**, run in CI to verify the core library works
end-to-end. It is **not** a user-facing workspace — see `../autocti_workspace` for examples and
tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autocti`, `autofit`, `autoarray`, and **arcticpy** (source-only C++ sdist; the
full install recipe is in `PyAutoCTI/AGENTS.md` §arcticpy, and CI runs it from
`PyAutoHeart/.github/actions/install-arcticpy`). Short form:

```bash
sudo apt-get install -y libgsl-dev
pip install --upgrade pip setuptools wheel     # setuptools is a BUILD dep; 3.12+ venvs omit it
pip install numpy cython scipy matplotlib
pip install arcticpy==2.6 --no-build-isolation --no-deps
```

## Repository Structure

```
scripts/                     Integration-test scripts run in CI
  dataset_1d/model_fit.py    1D calibration: simulate -> factor-graph fit -> aggregator round-trip
  imaging_ci/model_fit.py    2D charge injection: simulate -> factor-graph fit -> result inspection
  plot/subplots.py           Drives the autocti.plot function surface to files
legacy/                      Euclid VIS heritage (2022-2023, pre-resurrection API — not runnable;
                             see legacy/README.md)
config/                      Current PyAutoCTI config (mirrors autocti_workspace)
config/build/profile_smoke.yaml   Per-script env for smoke runs (PYAUTO_TEST_MODE=2 defaults)
smoke_tests.txt              The curated smoke list (small on purpose)
```

## Running

```bash
python .github/scripts/run_smoke.py     # the smoke list, with profile_smoke.yaml applied
python scripts/dataset_1d/model_fit.py  # one script, real search (no env applied)
```

CI runs the smoke list through PyAutoHeart's reusable smoke workflow (thin caller in
`.github/workflows/smoke_tests.yml`, chain `PyAutoNerves PyAutoFit PyAutoArray PyAutoCTI`). The
arcticpy build is **not** in `.github/scripts/smoke_install.sh` any more: the caller passes
`arcticpy: true` and Heart runs its own `install-arcticpy` action before the epilogue, so the
recipe has one owner instead of a copy per repo. `smoke_install.sh` now holds only the
workspace-specific chain install.

## Conventions

- Keep `smoke_tests.txt` a **small curated subset** — do not mass-promote scripts.
- Integration scripts are self-contained (simulate their own data, small shapes) and single-trap.
- The test-mode knob is `PYAUTO_TEST_MODE` (`2` bypasses sampling); `PYAUTOFIT_TEST_MODE`
  does not exist.
- Never edit `legacy/` — it is preserved Euclid VIS history.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->
