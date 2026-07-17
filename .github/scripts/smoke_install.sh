#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`).
set -e

# arcticpy (the C++ arctic clocking code) is a hard import of autocti but not
# a pip dependency: its sdist is source-only (needs libgsl-dev + cython) and
# its own requirements downgrade numpy below 2.0.
sudo apt-get update && sudo apt-get install -y libgsl-dev
pip install numpy cython
pip install arcticpy==2.6 --no-build-isolation --no-deps

pip install ./PyAutoConf ./PyAutoFit ./PyAutoArray ./PyAutoCTI
pip install "./PyAutoArray[optional]"
# The re-resolution above can upgrade autoconf to the stale PyPI release;
# pin the local source last so recent autoconf APIs are importable.
pip install --force-reinstall --no-deps ./PyAutoConf
