# PyAutoCTI Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoCTI**, run in CI to verify the core library works
end-to-end. It is **not** a user-facing workspace — see `../autocti_workspace` for examples and
tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autocti`, `autofit`, `autoarray`, and **arcticpy** (source-only C++ sdist — install
with `pip install arcticpy==2.6 --no-build-isolation --no-deps` after numpy+cython, with
`libgsl-dev` present; see `PyAutoCTI/AGENTS.md`).

## Repository Structure

```
scripts/                     Integration-test scripts run in CI
  dataset_1d/model_fit.py    1D calibration: simulate -> factor-graph fit -> aggregator round-trip
  imaging_ci/model_fit.py    2D charge injection: simulate -> factor-graph fit -> result inspection
  plot/subplots.py           Drives the autocti.plot function surface to files
legacy/                      Euclid VIS heritage (2022-2023, pre-resurrection API — not runnable;
                             see legacy/README.md)
config/                      Current PyAutoCTI config (mirrors autocti_workspace)
config/build/env_vars.yaml   Per-script env for smoke runs (PYAUTO_TEST_MODE=2 defaults)
smoke_tests.txt              The curated smoke list (small on purpose)
```

## Running

```bash
python .github/scripts/run_smoke.py     # the smoke list, with env_vars.yaml applied
python scripts/dataset_1d/model_fit.py  # one script, real search (no env applied)
```

CI runs the smoke list through PyAutoHeart's reusable smoke workflow (thin caller in
`.github/workflows/smoke_tests.yml`, chain `PyAutoNerves PyAutoFit PyAutoArray PyAutoCTI`; the
arcticpy build lives in `.github/scripts/smoke_install.sh`).

## Conventions

- Keep `smoke_tests.txt` a **small curated subset** — do not mass-promote scripts.
- Integration scripts are self-contained (simulate their own data, small shapes) and single-trap
  (identical-prior ordered-trap models tie at prior medians under the `PYAUTO_TEST_MODE=2`
  bypass and raise their own assertion — a filed autofit issue; avoid the pattern here).
- The test-mode knob is `PYAUTO_TEST_MODE` (`2` bypasses sampling); `PYAUTOFIT_TEST_MODE`
  does not exist.
- Never edit `legacy/` — it is preserved Euclid VIS history.

## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
