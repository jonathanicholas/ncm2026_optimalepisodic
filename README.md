# Eye-tracking Episodic Memory Decision Making (EMDM-Eye)

This repository contains data and analysis code for an experiment testing whether eye tracking can serve as a process-tracing measure of episodic memory retrieval during decision making. Participants encoded item-reward associations, then made take/leave decisions based on auditorily cued features while eye gaze was recorded at 1000 Hz.

---

## Repository Structure

```
ncm2026_optimalepisodic/
│
├── analysis/                          # Behavioral and eye-tracking analysis
│   ├── run_behavior.sh                #   Pipeline: behavior -> Figure 1 + stats
│   ├── run_eyetracking.sh             #   Pipeline: eye-tracking -> Figure 2 + stats
│   ├── run_analyze_recall.sh           #   Helper: per-subject recall analysis
│   ├── analyze_behavior.py             #   Compile behavioral data, generate Figure 1
│   ├── analyze_eyetracking.py          #   Read precomputed CSVs, generate Figure 2
│   ├── run_mixed_effects.R             #   Bayesian mixed-effects models for behavior (brms)
│   ├── run_mixed_effects_eye.R         #   Bayesian mixed-effects models for eye-tracking (brms)
│   └── lib/                            #   Core analysis modules
│       ├── analyze_recall.py           #     Per-subject recall fixation analysis (reads EDF data)
│       ├── analyze_recall_group.py     #     Group time course + cluster-based permutation tests
│       ├── prepare_choice_fixations.py #     Aggregate and validate choice-phase fixations
│       ├── choice_fixation_proportions.py    # Fixation proportions by relevance and valence
│       ├── generate_wedge_aligned_fixations.py  # Rotate fixations into circular wedge coordinates
│       ├── predict_choice_from_item_prop_time_interactions.py  # CV logistic regression: fixations -> choice
│       ├── compute_revisits_count_and_duration.py  # Revisit statistics for items during choice
│       ├── analyze_fixation_duration_by_position.py  # Fixation duration by location/relevance
│       ├── recalled_valence.py         #     Map items to recalled outcome valence
│       ├── visualize_recall_fixation_wedges_group.py   # Group heatmaps of recall fixations
│       ├── visualize_choice_fixation_wedges_group.py   # Choice heatmaps + contrast maps
│       ├── visualize_first_fixations_relevance_and_magnitude.py  # First fixation patterns
│       ├── plot_fixation_advantage_violin.py  # Violin plots of relevant vs irrelevant fixation time
│       ├── analyze_choice_fixation_sweeps.py  # Fixation sequence/transition sweep analysis
│       ├── compute_recall_drop_fraction.py  # Recall-calibrated fixation drop fraction
│       └── pygazeanalyser/             #     Third-party EyeLink EDF reading library
│
├── addm/                              # Attentional Drift Diffusion Model (aDDM)
│   ├── run_addm.sh                    #   Pipeline: CV fitting + comparison + recovery (modes: fix, rtTrans)
│   ├── run_kfold10_rt_transition_models.sh  # 10-fold CV with 5-seed sweep
│   ├── compare_cv_fits.py             #   Merge per-fold log-likelihoods, compute model comparisons
│   ├── plot_kfold_parameter_bars.py   #   Plot fitted parameter estimates (d, theta, sigma)
│   ├── plot_addm_ddm_comparison.py    #   Posterior predictive checks: aDDM vs DDM
│   ├── plot_recovery_sweep.py         #   Visualize parameter recovery results
│   ├── plot_addm_supplement.py        #   Combine PPC, fit params, and recovery into Figure S3
│   └── lib/                           #   aDDM core implementation
│       ├── adapted_addm_simulation.py #     Discrete-time aDDM simulator (drift + noise + boundaries)
│       ├── addm_fitting.py            #     Group-level fitting: gaze stats, generative model, pyBADS
│       ├── generative_gaze.py         #     Generate synthetic fixation sequences from empirical stats
│       ├── prepare_fixations_for_modeling.py  # Preprocess raw fixations for model input
│       ├── run_kfold_cv.py            #     K-fold cross-validation framework
│       └── parameter_recovery_sweep.py #    Systematic parameter recovery (500 combos)
│
├── metarnn/                           # Neural network model comparisons
│   ├── run_nn_pipeline.sh             #   Pipeline: process NNs -> compare -> Bayesian stats
│   ├── create_nn_figures.sh           #   Sub-pipeline: JSON simulations -> human-like CSVs + figures
│   ├── plot_NN_NN_comparison.py       #   Compare two NN variants (Figure 4)
│   ├── plot_NN_H_next_fixation_gen.py #   Human vs NN next-fixation generation (Figure 5, S5, S6)
│   ├── run_mixed_effects_human_vs_nn.R     # Bayesian models: human vs NN
│   ├── run_mixed_effects_nn_nn_comparison.R  # Bayesian models: NN vs NN
│   ├── run_mixed_effects_next_fixation_gen.R  # Bayesian models: fixation advantage + transitions
│   ├── docs/                          #   Analysis documentation
│   ├── lib/                           #   NN analysis tools
│   │   ├── compile_nn_to_human_fixations.py  # Convert NN JSON output to human-format CSVs
│   │   ├── analyze_NN_behavior.py     #     Compile NN behavioral summaries
│   │   ├── create_eyeplot_NN.py       #     Eye-tracking style plots from NN data
│   │   ├── plot_NN_overview.py        #     Overview visualization of NN behavior (Figure 3)
│   │   ├── plot_NN_H_comparison.py    #     Direct human-NN comparison plots
│   │   ├── plot_NN_sweep_transitions.py  #  Fixation transition patterns
│   │   ├── plot_prop_drop_supplement.py  #  Fixation drop fraction supplement (Figure S4)
│   │   └── export_nn_nn_comparison_data.py  # Export data for R statistical analysis
│   └── simulations/                   #   NN simulation data
│       ├── simulation_04_04_input0/   #     Raw JSON simulations (baseline NN, no episodic input)
│       ├── simulation_04_04_input5/   #     Raw JSON simulations (episodic-input NN)
│       ├── human_like_04_04_input0/   #     Processed baseline: human-format CSVs + outputs
│       └── human_like_04_04_input5/   #     Processed episodic-input: human-format CSVs + outputs
│
├── data/                              # Raw participant data (43 subjects: 101-150)
│   └── {SUBID}/                       #   Per-subject directory
│       ├── {SUBID}_MAIN_logfile_7.csv #     Master behavioral logfile (all task phases)
│       ├── {SUBID}_freerecall.csv     #     Free recall verbal responses + timestamps
│       ├── {SUBID}_valuerecall.csv    #     Reward recall data + timestamps
│       └── {SUBID}_fixations_*.csv    #     Preprocessed fixation dataframes
│
├── output/                            # All analysis results
│   ├── figures/                       #   Manuscript figures (Figures 1-5)
│   │   └── supplementary/            #   Supplementary figures (Figures S2-S6)
│   ├── behavior/                      #   Behavioral results
│   │   ├── Figure1.pdf                #     Figure 1
│   │   ├── FigureS2.pdf               #     Figure S2
│   │   └── stats/                     #     CSVs: accuracy, choice, recall, RT, model summaries
│   ├── eyegaze/                       #   Eye-tracking results
│   │   ├── Figure2.pdf                #     Figure 2
│   │   ├── recall/                    #     Group recall time courses + recall drop fraction
│   │   └── stats/                     #     CSVs: fixation proportions, mixed-effects summaries
│   ├── addm/                          #   aDDM modeling results
│   │   ├── FigureS3.pdf               #     Figure S3
│   │   ├── kfold/                     #     Per-seed, per-fold cross-validation outputs
│   │   ├── kfold_compare/             #     Merged CV comparisons
│   │   ├── ppc/                       #     Posterior predictive check data
│   │   └── parameter_recovery_sweep/  #     Recovery analysis results
│   └── next_fixation_gen/             #   Human fixation generation analysis cache
│
├── task/
│   └── emdm-eyetracking/
│       └── game_info/                 #     Trial configurations (items, rewards, sequences)
│
├── environment.yml                    # Conda environment specification
└── LICENSE                            # MIT License
```

---

## Pipelines

There are four analysis pipelines, each orchestrated by a top-level shell script. All scripts assume they are run from the repository root with the `analysis` conda environment active.

### 1. Behavioral Analysis (`analysis/run_behavior.sh`)

Compiles behavioral data and fits Bayesian mixed-effects models.

```
analyze_behavior.py  -->  run_mixed_effects.R
```

**Outputs:** `output/behavior/Figure1.pdf`, `output/behavior/FigureS2.pdf`, `output/behavior/stats/*.csv`

### 2. Eye-tracking Analysis (`analysis/run_eyetracking.sh`)

Eight-step pipeline from raw fixation data to publication figure and statistical models.

```
analyze_recall.py (per subject, parallelized)
  --> analyze_recall_group.py
  --> generate_wedge_aligned_fixations.py
  --> prepare_choice_fixations.py
  --> choice_fixation_proportions.py
  --> predict_choice_from_item_prop_time_interactions.py
  --> analyze_eyetracking.py  (Figure2.pdf)
  --> run_mixed_effects_eye.R
```

**Outputs:** `output/eyegaze/Figure2.pdf`, `output/eyegaze/stats/*.csv`, `output/eyegaze/recall/*.csv`

### 3. aDDM Modeling (`addm/run_addm.sh <fix|rtTrans>`)

Fits the attentional drift diffusion model via cross-validation, compares to a baseline DDM, and runs parameter recovery. Two modes: `fix` (fixation time) and `rtTrans` (reaction time with transitions).

```
run_kfold10_rt_transition_models.sh  (10-fold CV, 5 seeds)
  --> compare_cv_fits.py + plot_kfold_parameter_bars.py
  --> plot_addm_ddm_comparison.py  (posterior predictive checks)
  --> parameter_recovery_sweep.py + plot_recovery_sweep.py
  --> plot_addm_supplement.py  (rtTrans only; FigureS3.pdf)
```

**Outputs:** `output/addm/kfold/`, `output/addm/kfold_compare/`, `output/addm/ppc/`, `output/addm/parameter_recovery_sweep/`, `output/addm/FigureS3.pdf`

### 4. Neural Network Comparisons (`metarnn/run_nn_pipeline.sh <SIM_NAME> <NINPUTS>`)

Processes NN simulation JSON files into human-format CSVs, generates comparison figures, and fits Bayesian mixed-effects models comparing human vs NN and NN vs NN behavior.

```
create_nn_figures.sh (input0 baseline)
  --> create_nn_figures.sh (inputN target)
  --> run_mixed_effects_human_vs_nn.R
  --> plot_NN_NN_comparison.py
  --> export_nn_nn_comparison_data.py
  --> run_mixed_effects_nn_nn_comparison.R
  --> plot_NN_H_next_fixation_gen.py
  --> run_mixed_effects_next_fixation_gen.R
  --> copy figures to output/figures/
```

`create_nn_figures.sh` internally runs: JSON compilation, fixation preparation, choice fixation proportions, choice prediction (with recall-calibrated drop), prop-drop supplement, NN overview, and human-NN comparison figures.

**Outputs:** `metarnn/simulations/human_like_*/output/`, `output/figures/Figure3-5.pdf`, `output/figures/supplementary/FigureS4-S6.pdf`

---

## Sample

43 participants total. Subjects 107 and 131 are excluded from all eye-tracking and aDDM analyses (N=41) due to excessive data loss. All 43 are included in purely behavioral analyses.

---

## Setup

### Environment

Create the conda environment from the included specification:

```bash
conda env create -f environment.yml
```

After creating the environment, install the required R packages:

```r
install.packages(c("brms", "readr", "dplyr", "tidyr", "broom.mixed"))
```

### Reproducing Results

Pipelines 1-3 (behavioral, eye-tracking, aDDM) are independent and can be run in any order. Pipeline 4 (NN) depends on outputs from Pipeline 2 (eye-tracking), specifically files in `output/eyegaze/stats/` and `output/eyegaze/recall/`.

Pre-computed outputs are included in `output/` and `metarnn/simulations/human_like_*/output/`, so results can be inspected without re-running.

```bash
conda activate analysis

# Pipeline 1: Behavioral analyses
bash analysis/run_behavior.sh

# Pipeline 2: Eye-tracking analyses
bash analysis/run_eyetracking.sh

# Pipeline 3: aDDM modeling (run both modes)
bash addm/run_addm.sh fix
bash addm/run_addm.sh rtTrans

# Pipeline 4: NN comparisons (comparing input0 vs input5)
bash metarnn/run_nn_pipeline.sh 04_04 5
```

### Notes

- All scripts should be run from the repository root directory.
- The eye-tracking pipeline parallelizes per-subject analysis (default: 4 concurrent jobs).
- The aDDM parameter recovery sweep runs 1000 parameter combinations and may take several hours.
- Bayesian mixed-effects models (brms) use 4 chains with 2000 iterations each.

---

## Figures

All manuscript figures are collected in `output/figures/` (main) and `output/figures/supplementary/` (supplements). Each figure is also saved alongside its pipeline outputs in the original output directory.

| Figure | Description | Generated by |
|--------|-------------|--------------|
| Figure 1 | Behavioral results | `analysis/analyze_behavior.py` |
| Figure 2 | Eye-tracking results | `analysis/analyze_eyetracking.py` |
| Figure 3 | NN overview | `metarnn/lib/plot_NN_overview.py` |
| Figure 4 | NN-NN comparison | `metarnn/plot_NN_NN_comparison.py` |
| Figure 5 | Next-fixation generation | `metarnn/plot_NN_H_next_fixation_gen.py` |
| Figure S2 | Behavioral supplement | `analysis/analyze_behavior.py` |
| Figure S3 | aDDM supplement | `addm/plot_addm_supplement.py` |
| Figure S4 | Fixation drop supplement | `metarnn/lib/plot_prop_drop_supplement.py` |
| Figure S5 | Transition supplement | `metarnn/plot_NN_H_next_fixation_gen.py` |
| Figure S6 | Advantage supplement | `metarnn/plot_NN_H_next_fixation_gen.py` |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
