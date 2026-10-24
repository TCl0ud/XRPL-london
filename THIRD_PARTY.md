# What we did not write ourselves

- **Organiser material**: the synthetic dataset and its README. `detector/data.py` decodes amounts
  and values them in USD the way the organisers' README and `payments.py` describe (first transfer
  leg = actual source cost, hourly reference price from `market_rates.csv`); no organiser code is copied.
- **Libraries** (`requirements.txt`): pandas, NumPy, PyArrow, scikit-learn
  (`HistGradientBoostingClassifier`, `StratifiedGroupKFold`, `average_precision_score`) and SciPy
  (`connected_components`, `rankdata`).
- **AI assistance**: the code and documentation were written with Claude (Anthropic) via Claude Code.
- No pre-trained models, external data, external labels or outside services are used.
