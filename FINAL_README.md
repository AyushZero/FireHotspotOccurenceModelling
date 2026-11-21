# California daily hotspot occurrence — final paper and reproducible study

This folder is the standalone final study. Start with **final_paper.pdf** and its editable **final_paper.tex**. The manuscript presents the proposed method directly; it is not a revision report about an earlier paper.

## Study design

Observed ERA5, VIIRS/FIRMS, MODIS MOD13A1.061 and SRTM inputs cover 2021–2024. Models train on 2021–2022. Model selection uses days 1–14 of each month of 2023; separate calibration uses days 22–month-end. Calibration candidates fit on days 22–25 and are assessed on the remaining calibration days, then the chosen mapping refits all calibration rows. Days 15–21 are unused in selection/calibration. Month-boundary dependence remains and is discussed in the paper.

2024 is the later-year test. Model and probability-mapping decisions were frozen before calculating its performance. Same-day weather and satellite hotspot labels make this retrospective occurrence modelling, not advance forecasting or verified new-ignition prediction.

Earlier papers used different years and vegetation inputs. Their headline scores are not a controlled comparator for this study. Use the same-population baselines and paired intervals in this paper when assessing improvements.

The mathematical search contains 276 fitted configurations: 12 matched directional/linear fits, 27 smooth additive fits, 192 joint multiplicative configurations (96 parameter draws, each paired with raw and log-compressed histories), 36 tree refinements and nine matched class-handling configurations. The ensemble search adds 128 convex weight draws and four vertices. The criterion is validation average precision, with AUC as the tie breaker. This is a bounded empirical search, not proof of global optimality.

## What is included

The selected convex ensemble achieves 2024 AUC **0.846934**, AP **0.256396**, Brier **0.018066**, and **58.04%** positive capture in the highest-scored 10% of rows. Test prevalence is **2.13595%**, with 4,959 positives among 232,168 complete cell-days. Its AP gain over XGBoost is small and the 30-day paired interval includes zero. The additional spline roughness penalty did not improve validation AP; the selected spline uses ridge alone. Compressed history improved AP in 94/96 paired multiplicative candidates on validation. These findings are all reported, including the slight test Brier deterioration after the validation-selected calibration.

The six-page PDF and all three 300-dpi figures were visually reviewed. Static TeX checks, all 11 reported metric rows, frozen inference, spline-formula reconstruction and numerical gradient checks passed. Inside this handoff, rebuilding from raw inputs reproduced the exact feature hash; regenerating the manuscript and figures reproduced identical files. The original project folders were preserved.

- `final_paper.pdf`, `final_paper.tex`, `standalone_paper_template.tex.j2`: final paper and regenerable sources.
- `outputs/standalone/results.json`: test metrics, probability mappings and paired bootstrap intervals.
- `outputs/standalone/protocol.json`, `frozen_selection.json`, `frozen_calibration.json`: reproducible decision record.
- `outputs/standalone/leaderboard.json`, `ensemble_search.json`: every candidate score and blend weight.
- `outputs/standalone/test_predictions.npz` and `selected_test_predictions.csv`: predictions and labels used by the paper.
- `outputs/standalone/verification.json`: numerical and frozen-inference checks.
- `outputs/standalone/smooth_formula.json`, `multiplicative_formula.json`, `selected_method.json`: readable fitted coefficients, knot vectors, index parameters, ensemble weights and probability mapping.
- `outputs/stage3/data_audit.json`: source coverage, missing observations, sample counts and hashes.
- `models/standalone/`: all family winners, selected ensemble and required probability mappings. Unused candidate model binaries are omitted to keep the handoff smaller; their full settings/results and validation predictions remain, and training recreates them.
- `data/raw/`: all required observed input subsets and source metadata. MODIS samples include raw integer bands and quality flags, with unsigned source URLs and hashes. Source granule gaps are recorded explicitly, never replaced with synthetic observations.
- `data/standalone/`: raw daily cell observations, training-fitted scaling limits and prepared model inputs.
- `logs/`: acquisition, fitting, evaluation and verification logs.

The previous papers and obsolete model artifacts are intentionally not included in this final deliverable.

## Step-by-step reproduction (Windows PowerShell)

1. Install Python 3.11. Install Tectonic if you want to compile the PDF. The included PDF can be read without installing anything.
2. Open PowerShell in this folder and run:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Verify saved results and inference: `python verify_standalone.py`.
4. Regenerate figures and paper from saved results:

   ```powershell
   python make_standalone_paper.py
   python check_tex_static.py
   tectonic final_paper.tex
   ```

   Tectonic may need internet access for its initial bundle. All publication figures are generated at 300 dpi.

5. Apply the saved model to included observations without fitting: `python predict_standalone.py --year 2024`. This writes `outputs/standalone/predictions_2024.csv`.
6. For complete reproduction from included raw inputs, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_standalone_pipeline.ps1
   python verify_standalone.py
   tectonic final_paper.tex
   ```

   This rebuilds daily inputs and scaling, fits missing candidates, repeats selection/calibration, evaluates 2024, runs 7-day and 30-day paired bootstraps (300 replicates each), and rebuilds figures/TeX. Expect a sustained run and several GB of available RAM. Each stage writes a log. Existing matching candidate fits resume; the protocol checks the feature hash and candidate specifications.

7. To refit every candidate from scratch, move `models/standalone` and `outputs/standalone` to backup locations before step 6. The scripts recreate them. Keep `outputs/cb_2024_us_state_500k.zip` and the source-data audit directories. Do not mix caches from a different dataset or protocol.

## Optional reacquisition

Reacquisition is unnecessary because required subsets are included. For MODIS, run `python fetch_modis_public.py` with internet access; it uses a public read token, not an Earthdata credential, and samples native 500 m pixels. It retains source metadata and missing-granule records. Quality filtering is applied later by the daily builder.

FIRMS: supply `FIRMS_MAP_KEY` through your environment and run `python fetch_firms_archive.py --year 2024` (repeat with another required year as needed). No user credential is included or printed by the downloader.

ERA5 acquisition uses a separate Python 3.12 environment because Icechunk v2 requires it:

```powershell
py -3.12 -m venv .venv_era5
.\.venv_era5\Scripts\python.exe -m pip install icechunk pcodec xarray zarr numpy pandas netcdf4 dask
.\.venv_era5\Scripts\python.exe fetch_era5_earthmover.py --year 2024
```

The included SRTM elevation tiles can rebuild slopes with `python build_srtm_slope.py`. Source products have their own attribution and usage terms; consult the citations in the paper and recorded source metadata.

## GitHub sharing

Run `git lfs install` before adding the folder as a repository. `.gitattributes` covers NetCDF, GeoTIFF, joblib and NPZ binaries. Keep the included results, figures, source observations and models. `.gitignore` excludes caches, environments, temporary files and logs from ordinary Git tracking; logs remain available locally. Check that the destination account has enough LFS storage.

`MANIFEST.json` records hashes for the delivered files, including data and models. Numerical results can be checked from saved predictions independently of the paper renderer. The study's data coverage and validation limitations are stated in the manuscript.

The parent project's LICENSE is retained. Source-data attribution is provided in the manuscript and acquisition metadata.
