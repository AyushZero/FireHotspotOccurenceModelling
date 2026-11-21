# Wildfire hotspot model research v2

This is a separate follow-up. The original Wildfire_Ignition_Paper_Handoff is unchanged.

## Deliverables

- `research_paper.pdf`: new six-page IEEE manuscript.
- `research_paper.tex`, `research_paper_template.tex.j2`: generated and editable sources.
- `research_protocol.json`: partitions, search scope, primary metric and limitations.
- `outputs/research/leaderboard.json`: all 61 candidate selection scores.
- `outputs/research/frozen_selection.json` and `frozen_calibration.json`: decisions made before benchmark scoring.
- `outputs/research/research_results.json`: raw/calibrated metrics and paired AUC/AP intervals.
- `outputs/research/benchmark_predictions.npz`: labels and reported 2023 probabilities.
- `models/research/`: serialized candidate models and probability mappings.
- `logs/`: experiment logs; candidate JSONs also retain settings and elapsed time.

Selected model: XGBoost, 300 trees, depth 4, log-compressed histories, no weighting or resampling. The selected calibration is identity (no adjustment). On 2023: AUC 0.85915, AP 0.28596, Brier 0.018487, top-decile capture 59.96%. Original RF: AUC 0.86011, AP 0.27709, Brier 0.018759, capture 60.58%. Improvements are mixed across metrics.

2023 was previously examined. This is retrospective development, not fresh independent validation. Complete included MODIS inputs end in December 2023. A finite candidate search does not prove global optimality. The target is hotspot cell-days, not confirmed new ignitions or advance forecasts.

## Reproduction on Windows

1. Install Python 3.11 and optionally Tectonic for PDF compilation. Open PowerShell here.
2. Run `py -3.11 -m venv .venv`, then `.\.venv\Scripts\Activate.ps1` and `python -m pip install -r requirements.txt`.
3. Run `python audit_real_inputs.py`. Prepared features and all raw sources are included.
4. Run `powershell -ExecutionPolicy Bypass -File .\run_research_pipeline.ps1`. This resumes candidates, freezes selection, selects calibration, evaluates 2023, runs bootstraps, creates figures/TeX, checks TeX and compiles if Tectonic is installed. Logs are in `logs/`. Allow several minutes and several GB of RAM. Tectonic needs internet for its initial bundle.
5. To refit every candidate from scratch, move `outputs/research` and `models/research` to backup locations first. The scripts recreate these directories. Do not reuse caches with changed features; cached specifications are checked but the feature data must also remain identical.
6. To rebuild features and original comparator results from raw data first, run `powershell -ExecutionPolicy Bypass -File .\run_empirical_pipeline.ps1`, clear research caches as above, then run step 4. That original workflow reproduces comparators; it does not replace the new paper.

Individual stages:

```powershell
python run_research.py
python evaluate_research.py
python make_research_paper.py
python check_tex_static.py research_paper.tex
tectonic research_paper.tex
```

Apply the saved selected model without refitting using `python predict_research.py --year 2023`. It writes dates, grid coordinates and probabilities to `outputs/research/predictions_2023.csv`. This uses prepared retrospective inputs, not live forecast data.

## Protocol

Train on 2021. Select on 2022 days 1-14 of each month; calibrate on days 22-month-end. Days 15-21 are unused. Calibration candidates fit on days 22-25 and are assessed on remaining calibration days, then the chosen mapping refits all calibration rows. Month-boundary dependence remains and is disclosed.

Search: 30 matched logistic fits, eight spline/interaction fits, five multiplicative offsets and 18 tree/class-handling combinations. Selection uses AP. Paired bootstrap retains all benchmark rows, with circular 7-day and 30-day blocks and 300 replicates each, reporting AUC and AP differences.

The original preprocessing/model scripts, src modules, outputs and model artifacts reproduce comparator results. `baseline/` preserves the preceding paper as historical context. Its included PDF is ready to read; the archived TeX refers to original root-relative figures. Candidate artifacts are retained for auditing and reproduction.

## Sharing

The observed ERA5, FIRMS, MODIS and SRTM inputs, source audits, boundary, features, results and models are included. No API key is included. Existing acquisition scripts are retained; reacquisition is unnecessary. FIRMS downloads require an external FIRMS_MAP_KEY.

Run `git lfs install` before adding this as a GitHub repository. The supplied .gitattributes covers large data/model binaries. .gitignore excludes environments, caches and logs; the local folder retains experiment logs. Publication outputs and data are included. Ensure the destination has sufficient Git LFS storage.

## Verification

Run `python verify_research.py` to recompute all 17 reported metric rows, check the validation winner, compare weighted AUC/AP calculations with scikit-learn, and refit the selected model from 2021. The completed audit reproduced selected probabilities exactly. `outputs/research/verification.json` records the checks. The PDF has six pages; all figures are 300 dpi and the rendered pages were inspected.
