# AODMA --- Agentic Optimization of Diffusion Models

**Working research title:** A Budget-Aware Multi-Agent Framework for
Adaptive and Resource-Efficient Diffusion Model Inference\
**Core research component:** BQRPS --- Budgeted Quality-Risk Portfolio
Selection\
**Status:** End-to-end multi-agent smoke workflow implemented. Novelty
investigation and statistical validation remain in progress.

## 1. Project overview

AODMA explores resource-aware optimization of diffusion-model inference.
It combines:

-   A diffusion runtime with multiple samplers.
-   Benchmarking and image-quality evaluation.
-   A portfolio of pre-profiled inference configurations.
-   BQRPS for budget-aware candidate selection.
-   A multi-agent workflow for research, strategy, execution,
    evaluation, and reporting.

The intended contribution is BQRPS; the agents provide orchestration.
Agentic orchestration alone is not the claimed algorithmic contribution.

## 2. Research formulation

A candidate configuration is:

\[ c=(s,`\tau`{=tex},n,p) \]

where (s) is sampler, (`\tau`{=tex}) is timestep schedule, (n) is NFE,
and (p) is precision.

The intended selection problem is:

\[
c\^\*=`\arg`{=tex}`\min`{=tex}\_{c`\in`{=tex}`\mathcal `{=tex}C}R_Q(c)
\]

subject to:

\[
R_T(c)`\leq `{=tex}T\_{`\max`{=tex}},`\qquad `{=tex}M_c`\leq `{=tex}M\_{`\max`{=tex}},`\qquad `{=tex}NFE(c)`\leq `{=tex}B\_{`\max`{=tex}}.
\]

Here (R_Q) is estimated quality risk, (R_T) is estimated latency risk,
(M_c) is memory use, and (B\_{`\max`{=tex}}) is the NFE budget.

The current design has used estimates of the form:

\[
R_Q(c)=`\hat `{=tex}Q_c+`\beta`{=tex}*Q`\sigma`{=tex}*{Q,c}+`\text{calibration offset}`{=tex}
\]

\[
R_T(c)=`\hat `{=tex}T_c+`\beta`{=tex}*T`\sigma`{=tex}*{T,c}+`\text{calibration offset}`{=tex}.
\]

**Statistical caution:** these are currently risk heuristics unless
validated on held-out observations. They must not be described as
guaranteed confidence bounds until coverage is demonstrated. FID and KID
are distribution-level metrics, not per-image quality scores.

## 3. Project structure

This is the consolidated structure recorded during development, not a
fresh filesystem listing. Some directories may contain additional files.

``` text
AODMA/
├── agents/
│   ├── llm/
│   │   ├── config.py
│   │   ├── provider.py
│   │   └── __init__.py
│   ├── orchestrator.py
│   ├── state.py
│   ├── research_agent.py
│   ├── strategy_agent.py
│   ├── execution_agent.py
│   ├── evaluation_agent.py
│   ├── reporting_agent.py
│   └── diffusion_runtime.py
├── diffusion/
│   ├── models/
│   │   ├── unet.py
│   │   └── __init__.py
│   ├── noise_schedule.py
│   ├── forward_process.py
│   ├── datasets.py
│   ├── training.py
│   └── __init__.py
├── samplers/
│   ├── ddpm.py
│   ├── ddim.py
│   ├── dpm_solver.py
│   ├── unipc.py
│   └── __init__.py
├── bqrps/
│   ├── portfolio.py
│   ├── profiler.py
│   ├── calibration.py
│   ├── uncertainty.py
│   ├── selector.py
│   ├── baselines.py
│   └── build_portfolio.py
├── evaluation/
│   ├── reproducibility.py
│   ├── statistics.py
│   ├── metrics.py
│   ├── benchmark.py
│   ├── fid.py
│   ├── kid.py
│   ├── quality_metrics.py
│   └── feature_extractor.py
├── experiments/
│   ├── configs/
│   │   ├── ddpm_mnist.yaml
│   │   ├── ddpm_cifar10.yaml
│   │   ├── benchmark.yaml
│   │   ├── bqrps_ablation.yaml
│   │   └── bqrps_budget_sensitivity.yaml
│   ├── baseline/
│   │   ├── train_ddpm.py
│   │   ├── train_cifar10.py
│   │   ├── generate_samples.py
│   │   ├── run_benchmark.py
│   │   ├── evaluate_quality.py
│   │   └── run_multi_agent.py
│   └── ablations/
│       ├── run_bqrps_ablation.py
│       └── run_budget_sensitivity.py
├── checkpoints/
│   ├── ddpm_mnist/
│   └── ddpm_cifar10/
├── results/
│   ├── cifar10/
│   │   └── bqrps/
│   │       └── unified_portfolio.json
│   └── agentic/
│       ├── generated/
│       ├── aodma_report_*.md
│       └── aodma_result_*.json
├── .env
├── .env.example
├── requirements.txt
└── [other project files]
```

## 4. Development environment

  Component              Recorded setup
  ---------------------- -------------------------------
  OS                     Windows
  Python                 3.13.14
  Environment            `.venv`
  PyTorch                2.6.0+cu124
  TorchVision            0.21.0+cu124
  NumPy                  2.2.6
  Diffusers              0.32.2
  GPU                    NVIDIA GeForce GTX 1650, 4 GB
  RAM                    8 GB
  NVIDIA driver          555.97
  LLM runtime            Ollama
  Local model            Mistral
  Alternative provider   OpenRouter

The machine is resource-constrained, so use conservative batch sizes and
avoid unnecessary large-scale runs.

### LLM configuration

The provider abstraction is in `agents/llm/config.py` and
`agents/llm/provider.py`. Provider selection is configured through
`.env`.

Example:

``` env
LLM_PROVIDER=ollama
LLM_TEMPERATURE=0.2
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

The local Mistral/Ollama path has been tested; OpenRouter is available
as an alternative provider path.

## 5. Diffusion implementation

Implemented:

-   Forward diffusion process and noise schedule.
-   Time-conditioned residual U-Net.
-   MNIST and CIFAR-10 data loaders.
-   Training with AdamW, AMP support, EMA, gradient clipping, checkpoint
    save/resume, and JSONL logging.
-   DDPM, DDIM, DPM-Solver, and UniPC sampler implementations.

Training recorded:

-   MNIST DDPM: 20 epochs, approximately 37,500 steps, final loss around
    0.02268.
-   CIFAR-10 DDPM: 20 epochs; checkpoint at
    `checkpoints/ddpm_cifar10/latest.pt`.

## 6. Benchmarking completed

Benchmark infrastructure includes reproducibility utilities, statistics,
metrics, benchmark execution, and a baseline runner.

### MNIST pilot latency

  Candidate      NFE   Mean latency
  ----------- ------ --------------
  DDPM          1000   13,066.91 ms
  DDIM            10      105.98 ms
  DDIM            20      217.35 ms
  DDIM            50      698.18 ms
  DDIM           100    1,349.04 ms

The pilot showed DDIM-10 was about 123× faster than DDPM in that setup.
This is a latency result, not evidence of equivalent image quality.

### Four-sampler MNIST pilot

  Sampler           NFE   Mean latency
  -------------- ------ --------------
  DDPM             1000   13,517.03 ms
  DDIM               10      122.86 ms
  DDIM               20      244.47 ms
  DDIM               50      707.38 ms
  DPM-Solver++       10      145.93 ms
  DPM-Solver++       20      238.87 ms
  DPM-Solver++       50      682.63 ms
  UniPC              10      141.34 ms
  UniPC              20      263.52 ms
  UniPC              50      704.62 ms

These are pilot measurements. Warm-up, batch size, repetitions, device,
and timing protocol need standardization before strong comparisons.

## 7. CIFAR-10 quality evaluation

A quality pipeline was implemented using TorchMetrics FID/KID and
Inception features. The reported evaluation used 1,000 generated images
per candidate and 1,000 real CIFAR-10 test images.

  Candidate                FID   KID mean
  ----------------- ---------- ----------
  DDIM-10             186.1239   0.126757
  DDIM-20             140.9002   0.081827
  DDIM-50             101.1855   0.044305
  DPM-Solver++-10     248.0945   0.168546
  DPM-Solver++-20     232.6024   0.149596
  DPM-Solver++-50     222.6362   0.135675
  UniPC-10            244.9410   0.165340
  UniPC-20            229.9143   0.147086
  UniPC-50            219.1024   0.131497
  DDPM-1000           108.7219   0.026004

Caveats:

-   One reported FID result per candidate is present in the portfolio.
-   KID subset standard deviation is not independent-seed uncertainty.
-   FID/KID are distribution-level metrics.
-   Reproduction and independent-seed evaluation are needed before
    publication claims.

## 8. BQRPS implementation and preliminary results

BQRPS means **Budgeted Quality-Risk Portfolio Selection**.

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `portfolio.py`                      Candidate configuration, profiles,
                                      portfolio

  `profiler.py`                       Summarizes measurements and builds
                                      profiles

  `calibration.py`                    Residual calibration utilities

  `uncertainty.py`                    Quality and latency risk estimation

  `selector.py`                       Budget-aware candidate selection

  `baselines.py`                      Baseline policies

  `build_portfolio.py`                Builds portfolio from benchmark and
                                      quality JSON
  -----------------------------------------------------------------------

Portfolio file:

``` text
results/cifar10/bqrps/unified_portfolio.json
```

The portfolio contains 10 candidates. Candidate records include
configuration ID, sampler, schedule, NFE, precision, FID, KID, latency
samples and summaries, memory, throughput, actual NFE, benchmark
repetitions, and quality sample count.

Baselines implemented:

-   BQRPS
-   Best-FID
-   Fastest
-   Lowest-NFE

### Preliminary ablation

  Method                   Selected candidate         FID       Latency   NFE   Feasible
  ------------------------ -------------------- --------- ------------- ----- ----------
  BQRPS                    DDIM-20                140.900     269.96 ms    20       5/10
  Best-FID                 DDIM-20                140.900     269.96 ms    20       5/10
  Fastest                  DDIM-10                186.124     132.75 ms    10       5/10
  Lowest-NFE               DDIM-10                186.124     132.75 ms    10       5/10
  Full BQRPS               DDIM-20                140.900     269.96 ms    20       5/10
  No latency uncertainty   DDIM-20                140.900     269.96 ms    20       6/10
  No latency constraint    DDIM-50                101.185   1,087.87 ms    50       9/10
  No memory constraint     DDIM-20                140.900     269.96 ms    20       5/10
  No NFE constraint        DDIM-20                140.900     269.96 ms    20       5/10

### Preliminary budget sensitivity

    Latency budget BQRPS / Best-FID   Fastest / Lowest-NFE
  ---------------- ------------------ ----------------------
            150 ms DDIM-10            DDIM-10
            250 ms DDIM-10            DDIM-10
            350 ms DDIM-20            DDIM-10
            500 ms DDIM-20            DDIM-10
          1,200 ms DDIM-20            DDIM-10

Interpretation:

-   BQRPS and Best-FID tie in the reported main ablation because there
    is one FID estimate per candidate and the quality safety margin is
    zero.
-   The latency constraint is binding in the reported ablation; removing
    it allows DDIM-50.
-   These results show policy sensitivity, not statistical superiority.
-   A single observation cannot provide empirical variance; fallback
    margins are heuristic.
-   The uncertainty implementation's confidence-level handling needs
    review: the recorded implementation supports 95% critical values
    despite exposing a confidence-level parameter.
-   Three latency repetitions are a pilot, not a reliable future-latency
    or p95 guarantee.
-   Held-out calibration and coverage testing remain necessary.

## 9. Multi-agent architecture

The current workflow uses LangGraph-style orchestration.

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  Research Agent                      Summarizes portfolio and research
                                      context

  Strategy Agent                      Adapts flat portfolio records and
                                      invokes deterministic BQRPS

  Execution Agent                     Executes the selected real
                                      diffusion sampler

  Evaluation Agent                    Validates generated samples

  Reporting Agent                     Creates a Markdown report

  Orchestrator                        Connects workflow stages and shared
                                      state

  Diffusion Runtime                   Loads checkpoint and exposes
                                      sampler execution

  LLM provider                        Local Ollama/Mistral or OpenRouter
  -----------------------------------------------------------------------

**Design principle:** LLMs can summarize and coordinate, but the final
numerical selection must remain with deterministic BQRPS logic.

### Successful end-to-end smoke test

Command:

``` powershell
python -m experiments.baseline.run_multi_agent --samples 32 --batch-size 4 --seed 42
```

Observed:

-   CUDA runtime and CIFAR-10 checkpoint loaded.
-   LangGraph workflow completed.
-   32 images generated using DDIM-20.
-   Smoke evaluation passed.
-   Markdown report and JSON result were saved.

Selected configuration:

``` text
config_id: ddim_linear_nfe20_fp32
sampler: ddim
schedule: linear
nfe: 20
precision: fp32
stored profile FID: 140.9002
stored profile latency mean: 269.96 ms
stored profile latency upper risk: 332.68 ms
stored profile peak memory: 36.479 MB
```

Execution measurements from that run:

``` text
sample_count: 32
batch_size: 4
number_of_batches: 8
seed: 42
mean batch latency: 411.75 ms
total generation latency: 3293.98 ms
throughput: 9.71 images/second
peak allocated memory: 20.39 MB
peak reserved memory: 56.00 MB
device: CUDA
```

The first batch was much slower than later batches, indicating warm-up
effects.

### What the smoke test proves

It validates integration of portfolio loading, BQRPS selection, real
sampler execution, sample saving, smoke evaluation, and report
generation.

It does **not** validate current-run FID/KID, image quality, BQRPS
superiority, risk calibration, or comparability between stored profile
latency and the new execution measurement.

The selected candidate's FID came from the earlier 1,000-image portfolio
evaluation. It was not computed from the 32 images generated in the
smoke run.

## 10. Current status

### Implemented / completed

-   [x] Diffusion forward process and noise schedule.
-   [x] Time-conditioned U-Net.
-   [x] MNIST and CIFAR-10 data pipelines.
-   [x] DDPM training and checkpointing.
-   [x] DDPM, DDIM, DPM-Solver, and UniPC samplers.
-   [x] Initial latency benchmark infrastructure.
-   [x] CIFAR-10 FID/KID pipeline.
-   [x] Candidate portfolio construction.
-   [x] Initial BQRPS selector and baseline policies.
-   [x] Preliminary ablation and budget sensitivity.
-   [x] Multi-agent workflow integration.
-   [x] Real DDIM execution from the agent workflow.
-   [x] Smoke evaluation and report generation.

### Pending

-   [ ] Establish novelty through structured literature review.
-   [ ] Compare at least 15 closely related works.
-   [ ] Obtain independent-seed quality measurements.
-   [ ] Correct confidence-level handling.
-   [ ] Implement held-out residual calibration and coverage
    diagnostics.
-   [ ] Evaluate selection stability.
-   [ ] Standardize warm-up, timing, batch size, device, precision, and
    memory protocols.
-   [ ] Measure orchestration overhead separately.
-   [ ] Add full FID/KID evaluation to the agent workflow.
-   [ ] Validate agent evaluation against the standalone evaluation
    pipeline.
-   [ ] Produce a complete evidence-grounded report.

## 11. Agreed research priority order

1.  **Establish novelty.**
2.  **Strengthen BQRPS statistically.**
3.  **Make experimental measurements defensible.**
4.  **Validate agent orchestration.**
5.  **Write the paper.**

### Phase 1 --- Establish novelty

Create:

``` text
research/
└── novelty_matrix.md
```

Review approximately 15 closely related papers. For each, record:

-   Title, authors, year, venue, publication status, and link.
-   Problem and methodology.
-   Optimization objectives and resource constraints.
-   Uncertainty estimation and calibration guarantees.
-   Datasets, hardware, and experimental setup.
-   Limitations, overlap with AODMA, and potential research gap.

Initial shortlist:

1.  AdaDiff.
2.  QoS-Diff.
3.  Adaptive Time-Stepping Schedules.
4.  SDM (2026 preprint).
5.  Xema (2026 preprint).
6.  Adaptive Second-Order Solvers (2026 preprint).
7.  DiffPro.
8.  Hierarchical Schedule Optimization.

This is a starting list, not a completed or verified literature review.
Verify bibliographic details and claims from primary sources before
citing them.

Questions to resolve:

1.  Do existing methods select among multiple pre-profiled sampler
    configurations using predicted quality and latency?
2.  Does any method combine quality-risk estimation, latency-risk
    estimation, and hard memory/NFE constraints in one selection
    procedure?
3.  Does any method calibrate those risk estimates on held-out
    observations and evaluate constraint-violation coverage?
4.  Does any framework coordinate research, optimization, execution, and
    evaluation with agents while retaining deterministic numerical
    selection?

Do not claim novelty until the closest competing work has been examined.

### Phase 2 --- Strengthen BQRPS statistically

-   Obtain independent-seed quality measurements.
-   Estimate quality uncertainty with a defensible protocol.
-   Correct confidence-level support.
-   Separate empirical uncertainty from heuristic fallback margins.
-   Calibrate residuals on held-out data.
-   Report coverage, calibration error, and sample counts.
-   Explain infeasible portfolios and preserve rejection reasons.

### Phase 3 --- Make measurements defensible

-   Add warm-up iterations.
-   Separate warm-up from timed inference.
-   Record batch size, resolution, device, precision, and software
    configuration.
-   Standardize peak allocated/reserved memory reporting.
-   Measure orchestration overhead separately.
-   Ensure methods are compared under equivalent conditions.

### Phase 4 --- Validate agent orchestration

-   Research Agent reports portfolio statistics and data limitations.
-   Strategy Agent explains selection and candidate rejection.
-   Execution Agent validates configurations and runtime constraints.
-   Evaluation Agent distinguishes new measurements from stored profile
    metrics.
-   Reporting Agent produces evidence-grounded reports.
-   LLM outputs must not override deterministic BQRPS decisions.

### Phase 5 --- Write the paper

Proposed sections:

1.  Abstract.
2.  Introduction.
3.  Related Work.
4.  Problem Formulation.
5.  AODMA Architecture.
6.  BQRPS Methodology.
7.  Experimental Setup.
8.  Results and Discussion.
9.  Ablation and Sensitivity Analysis.
10. Limitations.
11. Conclusion.

## 12. Limitations to keep explicit

1.  Novelty has not yet been established.
2.  BQRPS ties Best-FID in the reported main ablation.
3.  Quality uncertainty is not supported by independent-seed
    measurements yet.
4.  Held-out calibration and coverage are not demonstrated.
5.  Pilot latency repetitions are limited.
6.  Benchmark and end-to-end timings are not fully comparable.
7.  Agent evaluation so far is a smoke test, not full FID/KID
    evaluation.
8.  The smoke-test report's FID is a stored portfolio metric, not a
    current-run measurement.
9.  Hardware and RAM constrain experiment scale.

These are open research tasks. Documenting them accurately is necessary
for credible results.

## 13. Immediate next action

Begin the novelty review and create `research/novelty_matrix.md`. Verify
the initial papers from their primary sources and compare them using
consistent criteria.

**Do not change the BQRPS algorithm yet.** First establish the precise
research gap; then revise the method and experiments to test it.

------------------------------------------------------------------------

*This document summarizes implementation and results recorded in the
project history up to 17 November 2026. The directory tree is a
consolidated working overview, not a fresh filesystem audit. Verify
local files and reproduce numerical results before using them as final
publication evidence.*
