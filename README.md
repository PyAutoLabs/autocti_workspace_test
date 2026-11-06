# PyAutoCTI Workspace Test

Integration, smoke and regression test workspace for **PyAutoCTI** example scripts and release validation.

Every script simulates its own data and asserts the result:

- **Recover the input**: the CTI model that made the data fits it perfectly, and a likelihood scan over trap density peaks at the true value (1D, and 2D parallel / serial / both).
- **CTI correction**: removing CTI from an image recovers the image from before CTI was added.
- **Cosmic rays**: injected cosmic rays are flagged, and the calibration mask covers their CTI trails.
- **Model fits**: a full fit round-trips through the results database.
- **Likelihood pins**: exact log-likelihood values, so any numerical change is noticed.

Run them all with `python .github/scripts/run_smoke.py`. Details are in `AGENTS.md`.

<sub><i><a href="https://open.spotify.com/track/4pok08MvDsJzOM6kMpWHmc">we gon' make it through</a></i></sub>
