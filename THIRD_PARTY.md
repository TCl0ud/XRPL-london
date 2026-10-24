# What we did not write ourselves

- **Organiser material**: the synthetic dataset, its README, the starter notebook and the
  organisers' `payments.py` loader (none of these are included). Amount decoding and hourly USD
  valuation in `detector/data.py` and `xrpl-detector/aml_ranker/data.py` follow `payments.py` and
  the organisers' README.
- **Libraries** (`requirements.txt`, `xrpl-detector/requirements.txt`): NumPy, pandas, PyArrow,
  scikit-learn (`HistGradientBoostingClassifier`, `StratifiedGroupKFold`, metrics), SciPy
  (`connected_components`, `rankdata`), NetworkX, Matplotlib, joblib.
- **AI assistance**: the code and documentation were written with Claude (Anthropic) via Claude Code.
  The team members' prompts and what they produced are in [`chat-history/`](chat-history/README.md).
- No pre-trained models, external data, external labels or outside services are used.
