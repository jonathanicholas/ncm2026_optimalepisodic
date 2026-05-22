# Flexible decisions arise from optimally sampling memories

This repository contains data and analysis code for the following manuscript [eventual bioRxiv link].

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
│   ├── plot_addm_supplement.py        #   Combine PPC, fit params, and recovery into Figure S2
│   └── lib/                           #   aDDM core implementation
│       ├── adapted_addm_simulation.py #     Discrete-time aDDM simulator (drift + noise + boundaries)
│       ├── addm_fitting.py            #     Group-level fitting: gaze stats, generative model, pyBADS
│       ├── generative_gaze.py         #     Generate synthetic fixation sequences from empirical stats
│       ├── prepare_fixations_for_modeling.py  # Preprocess raw fixations for model input
│       ├── run_kfold_cv.py            #     K-fold cross-validation framework
│       └── parameter_recovery_sweep.py #    Systematic parameter recovery (1000 combos)
│
├── training/                          # RNN training pipeline
│   ├── run_train.sh                   #   SLURM: train network (jobid 0-9, one seed per job)
│   ├── run_simulate.sh                #   SLURM: run simulation with trained network
│   ├── run_transform.sh               #   SLURM: convert simulation pickle to JSON
│   ├── submit_train.sh                #   Submit training jobs to cluster
│   ├── submit_simulate.sh             #   Submit simulation jobs to cluster
│   ├── submit_transform.sh            #   Submit transform jobs to cluster
│   ├── train.py                       #   Train RNN via A2C
│   ├── simulate.py                    #   Simulate trained network (100k trials)
│   ├── transform.py                   #   Convert simulation pickle to JSON format
│   └── modules/                       #   Core training modules
│       ├── environment.py             #     Gymnasium environment
│       ├── network.py                 #     SharedGRURecurrentActorCriticPolicy
│       ├── a2c.py                     #     A2C trainer
│       ├── replaybuffer.py            #     Replay buffer for rollout storage
│       ├── simulation.py              #     Simulation utilities
│       ├── argument.py                #     Argument parsing and serialization
│       └── utils.py                   #     Miscellaneous utilities
│
├── metarnn/                           # RNN model comparisons
│   ├── run_nn_pipeline.sh             #   Pipeline: process NNs -> compare -> Bayesian stats
│   ├── create_nn_figures.sh           #   Sub-pipeline: JSON simulations -> human-like CSVs + figures
│   ├── plot_NN_NN_comparison.py       #   Compare two NN variants (Figure 4)
│   ├── run_mixed_effects_human_vs_nn.R     # Bayesian models: human vs NN
│   ├── run_mixed_effects_nn_nn_comparison.R  # Bayesian models: NN vs NN
│   ├── next_fixation/                 #   Next-fixation conditional-logit sub-pipeline
│   │   ├── run_next_fixation.sh       #     Pipeline: build data -> conditional-logit fits -> Figures 5, S5
│   │   ├── build_next_fixation_data.py  #   Build long-form candidate datasets (human, NN, nulls)
│   │   ├── conditional_logit.stan     #     Conditional-logit model (flat)
│   │   ├── conditional_logit_re.stan  #     Conditional-logit model (hierarchical)
│   │   ├── fit_conditional_logit.R    #     Flat fit (NN + null oracles)
│   │   ├── fit_conditional_logit_re.R #     Hierarchical fit (human)
│   │   ├── plot_next_fixation_forest.py        # Next-fixation forest (Figure 5)
│   │   ├── plot_next_fixation_nulls_forest.py  # Null-oracle forest (Figure S5)
│   │   └── lib/                       #     Data loaders, candidate features, null oracles
│   ├── lib/                           #   NN analysis tools
│   │   ├── compile_nn_to_human_fixations.py  # Convert NN JSON output to human-format CSVs
│   │   ├── analyze_NN_behavior.py     #     Compile NN behavioral summaries
│   │   ├── create_eyeplot_NN.py       #     Eye-tracking style plots from NN data
│   │   ├── plot_NN_overview.py        #     Overview visualization of NN behavior (Figure 3)
│   │   ├── plot_NN_H_comparison.py    #     Direct human-NN comparison plots
│   │   ├── plot_NN_sweep_transitions.py  #  Fixation transition patterns
│   │   ├── plot_prop_drop_supplement.py  #  Fixation drop fraction supplement (Figure S4)
│   │   ├── export_nn_nn_comparison_data.py  # Export data for R statistical analysis
│   │   ├── plot_evidence_figure.py    #     Evidence accumulation + decoding figure (Figure 3BC)
│   │   └── run_belief_decoding.py     #     Decode belief states from hidden states (OLS, GroupKFold by trial)
│   └── simulations/                   #   NN simulation data
│       ├── simulation_04_04_input0/   #     Raw JSON simulations (baseline NN, no episodic input)
│       │   └── with_hidden/           #       JSON files with hidden states + logits (gitignored)
│       ├── simulation_04_04_input5/   #     Raw JSON simulations (episodic-input NN)
│       ├── human_like_04_04_input0/   #     Processed baseline: human-format CSVs + outputs
│       │   └── output/evidence/       #       Evidence accumulation + belief decoding outputs
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
│   │   └── supplementary/            #   Supplementary figures (Figures S1-S7)
│   ├── behavior/                      #   Behavioral results
│   │   ├── Figure1.pdf                #     Figure 1
│   │   ├── FigureS1.pdf               #     Figure S1
│   │   └── stats/                     #     CSVs: accuracy, choice, recall, RT, model summaries
│   ├── eyegaze/                       #   Eye-tracking results
│   │   ├── Figure2.pdf                #     Figure 2
│   │   ├── recall/                    #     Group recall time courses + recall drop fraction
│   │   └── stats/                     #     CSVs: fixation proportions, mixed-effects summaries
│   ├── addm/                          #   aDDM modeling results
│   │   ├── FigureS2.pdf               #     Figure S2
│   │   ├── kfold/                     #     Per-seed, per-fold cross-validation outputs
│   │   ├── kfold_compare/             #     Merged CV comparisons
│   │   ├── ppc/                       #     Posterior predictive check data
│   │   └── parameter_recovery_sweep/  #     Recovery analysis results
│   ├── next_fixation/                 #   Next-fixation conditional-logit datasets + beta CSVs
│   └── next_fixation_gen/             #   Fixation-transition sweep cache (Figure S6)
│
├── supplemental_analysis/             # Supplementary robustness checks
│   ├── feature_analysis/              #   Per-feature-dimension robustness
│   │   ├── run_feature_analysis.sh    #     Pipeline: data -> 8 brms fits -> FigureS7
│   │   ├── scripts/                   #     build, compute, plot scripts
│   │   ├── data/                      #     Intermediate per-trial CSVs
│   │   └── output/
│   │       ├── FigureS7.pdf           #       Figure S7 (also copied to output/figures/supplementary/)
│   │       ├── feature_deviation_from_mean.csv
│   │       └── model_summaries/
│   ├── block_analysis/                #   Block-number (round) interaction check
│   │   ├── run_block_analysis.sh      #     Pipeline: data -> 9 brms fits -> TableS8
│   │   ├── scripts/                   #     build, compute scripts
│   │   ├── data/                      #     Intermediate per-trial CSVs
│   │   └── output/
│   │       ├── block_interaction_table.csv   #  Table S8 rows (one per analysis)
│   │       └── model_summaries/
│   └── fixation_transitions/          #   Fixation-transition structure (Figure S6)
│       ├── run_fixation_transitions.sh         # Pipeline: transition figure + Bayesian stats
│       ├── plot_fixation_transitions.py        # Transition-matrix + sequence-length figure (Figure S6)
│       └── run_mixed_effects_fixation_transitions.R  # Bayesian models: delta similarity + sequence length
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

There are seven analysis pipelines, each orchestrated by a top-level shell script. All scripts assume they are run from the repository root with the `analysis` conda environment active.

### 1. Behavioral Analysis (`analysis/run_behavior.sh`)

Compiles behavioral data and fits Bayesian mixed-effects models.

```
analyze_behavior.py  -->  run_mixed_effects.R
```

**Outputs:** `output/behavior/Figure1.pdf`, `output/behavior/FigureS1.pdf`, `output/behavior/stats/*.csv`

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
  --> plot_addm_supplement.py  (rtTrans only; FigureS2.pdf)
```

**Outputs:** `output/addm/kfold/`, `output/addm/kfold_compare/`, `output/addm/ppc/`, `output/addm/parameter_recovery_sweep/`, `output/addm/FigureS2.pdf`

### 4. RNN Training (`training/` — HPC cluster)

Trains a GRU-based recurrent actor-critic policy from scratch via A2C, then simulates behavior and converts the output to JSON for downstream NN analysis. Each job corresponds to one random seed; 5 seeds are run per condition via SLURM array (indices 0–9).

```
train.py  -->  simulate.py  -->  transform.py
```

The key condition variable is `init_num_items`, which controls how many episodic memory items are available at the start of each trial (e.g., 0 for the no-memory baseline, 5 for the partial-memory agent).

**Key training hyperparameters** (defaults in `training/modules/argument.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 100 | GRU hidden units |
| `num_episodes` | 25,000,000 | Total training episodes |
| `batch_size` | 40 | Parallel environments per update |
| `lr` | 1e-3 | Adam learning rate |
| `gamma` | 1.0 | Temporal discount (undiscounted) |
| `lamda` | 1.0 | GAE λ coefficient |
| `beta_v` | 0.05 | Value loss coefficient |
| `beta_e` | 0.05 | Entropy regularization coefficient |
| `stay_cost` | 0.008 | Per-step cost for fixating the same item |
| `saccade_cost` | 0.04 | Cost for shifting fixation |

**Outputs:** `training/results/exp_{init_num_items}_{jobid}/net.pth`, `data_training.p`, `data_simulation.p`; JSON files written to `training/results/data_json/data_{init_num_items}_{jobid}.json`.

The JSON output from `transform.py` matches the format consumed by the `metarnn/` pipeline. Copy or symlink the JSON files into `metarnn/simulations/` before running Pipeline 5.

To run on a SLURM cluster:

```bash
cd training

# Train 5 seeds for each condition
sbatch submit_train.sh

# After training completes: simulate then transform
sbatch submit_simulate.sh
sbatch submit_transform.sh
```

### 5. Neural Network Comparisons (`metarnn/run_nn_pipeline.sh <SIM_NAME> <NINPUTS>`)

Processes NN simulation JSON files into human-format CSVs, generates comparison figures, and fits Bayesian mixed-effects models comparing human vs NN and NN vs NN behavior.

```
create_nn_figures.sh (input0 baseline)
  --> create_nn_figures.sh (inputN target)
  --> run_mixed_effects_human_vs_nn.R
  --> plot_NN_NN_comparison.py
  --> export_nn_nn_comparison_data.py
  --> run_mixed_effects_nn_nn_comparison.R
  --> next_fixation/run_next_fixation.sh
  --> copy figures to output/figures/
```

`create_nn_figures.sh` internally runs: JSON compilation, fixation preparation, choice fixation proportions, choice prediction (with recall-calibrated drop), prop-drop supplement, NN overview, and human-NN comparison figures. The pipeline also runs belief decoding (decoding metalevel MDP belief states from the network's hidden states via ordinary least squares linear regression, with 5-fold GroupKFold by trial) and generates the evidence accumulation figure (Figure 3, panels B-C) and the belief-decoding supplement (Figure S3) if simulation files with hidden states are available in `simulation_*/with_hidden/`. Pre-computed outputs are cached so the figure can be regenerated without the large JSON files.

`next_fixation/run_next_fixation.sh` builds long-form candidate datasets for humans, the prior-memory network, and two null oracles (adjacent-walk, uniform-random), fits conditional-logit models (hierarchical for humans, flat for the network and nulls), and produces the next-fixation forest (Figure 5) and the null-oracle forest (Figure S5). MCMC fits are cached and skipped if their beta CSVs already exist.

**Outputs:** `metarnn/simulations/human_like_*/output/`, `output/next_fixation/`, `output/figures/Figure3-5.pdf`, `output/figures/Figure3BC.pdf`, `output/figures/supplementary/FigureS3-S5.pdf`

### 6. Feature-dimension robustness (`supplemental_analysis/feature_analysis/run_feature_analysis.sh`)

Fits per-feature-dimension reparameterizations of eight analyses from Pipelines 1 and 2 and produces Figure S7, summarising how each effect varies across the four feature dimensions.

```
build_choices.py
  --> build_eye_fixation.py
  --> compute_feature_stats.R   (8 brms fits; ~30-40 min)
  --> plot_feature_deviation.py (FigureS7.pdf)
```

**Outputs:** `supplemental_analysis/feature_analysis/output/FigureS7.pdf`, `.../feature_deviation_from_mean.csv`, `.../model_summaries/*.{txt,csv}`, `output/figures/supplementary/FigureS7.pdf`

### 7. Block-number (round) robustness (`supplemental_analysis/block_analysis/run_block_analysis.sh`)

Refits effects from Pipelines 1 and 2 with an added block-number (round) interaction as both a fixed effect and a by-subject random slope, plus three memory-performance checks (item recall, value recall fidelity, location recall). Produces Table S8, a 9-row summary of interaction coefficients and 95% HDIs.

```
build_choices.py
  --> build_eye_fixation.py
  --> compute_block_stats.R   (9 brms fits; ~30-45 min)
```

**Outputs:** `supplemental_analysis/block_analysis/output/block_interaction_table.csv`, `.../model_summaries/*.{txt,csv}`

### 8. Fixation-transition structure (`supplemental_analysis/fixation_transitions/run_fixation_transitions.sh <SIM_NAME> <NINPUTS>`)

Compares the transition-matrix structure and consecutive-fixation-sequence lengths of humans against the prior-memory network, with accompanying Bayesian mixed-effects statistics. Depends on the `human_like_*` exports, so run it after Pipeline 4.

```
plot_fixation_transitions.py            (FigureS6.pdf)
  --> run_mixed_effects_fixation_transitions.R   (delta similarity + sequence length)
  --> copy figure to output/figures/supplementary/
```

**Outputs:** `output/figures/supplementary/FigureS6.pdf`, `metarnn/simulations/human_like_*/output/next_fixation_gen/stats/*.{csv,txt}`

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

Pipelines 1-3 (behavioral, eye-tracking, aDDM) are independent and can be run in any order. Pipeline 4 (NN) depends on outputs from Pipeline 2 (eye-tracking), specifically files in `output/eyegaze/stats/` and `output/eyegaze/recall/`. Pipelines 6 and 7 (feature-dimension and block-number robustness) depend on the eye-tracking CSV produced by Pipeline 2, and Pipeline 7 additionally reads the behavioral memory CSVs produced by Pipeline 1. Pipeline 8 (fixation-transition structure) depends on the `human_like_*` exports produced by Pipeline 4.

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

# Pipeline 6: Feature-dimension robustness
bash supplemental_analysis/feature_analysis/run_feature_analysis.sh

# Pipeline 7: Block-number (round) robustness
bash supplemental_analysis/block_analysis/run_block_analysis.sh

# Pipeline 8: Fixation-transition structure (run after Pipeline 4)
bash supplemental_analysis/fixation_transitions/run_fixation_transitions.sh 04_04 5
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
| Figure 3 (D-J) | NN overview | `metarnn/lib/plot_NN_overview.py` |
| Figure 3 (B-C) | Evidence accumulation + belief decoding | `metarnn/lib/plot_evidence_figure.py` + `metarnn/lib/run_belief_decoding.py` |
| Figure 4 | NN-NN comparison | `metarnn/plot_NN_NN_comparison.py` |
| Figure 5 | Next-fixation conditional-logit forest | `metarnn/next_fixation/plot_next_fixation_forest.py` |
| Figure S1 | Behavioral supplement | `analysis/analyze_behavior.py` |
| Figure S2 | aDDM supplement | `addm/plot_addm_supplement.py` |
| Figure S3 | Belief decoding supplement | `metarnn/lib/plot_evidence_figure.py` + `metarnn/lib/run_belief_decoding.py` |
| Figure S4 | Fixation drop supplement | `metarnn/lib/plot_prop_drop_supplement.py` |
| Figure S5 | Next-fixation null-oracle forest | `metarnn/next_fixation/plot_next_fixation_nulls_forest.py` |
| Figure S6 | Fixation-transition structure | `supplemental_analysis/fixation_transitions/plot_fixation_transitions.py` |
| Figure S7 | Per-feature-dimension robustness | `supplemental_analysis/feature_analysis/scripts/plot_feature_deviation.py` |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
