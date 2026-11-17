# AODMA
# AODMA — Agentic Optimization of Diffusion Models

### A Budget-Aware Multi-Agent Framework for Risk-Calibrated Diffusion Inference

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/Agents-LangGraph-8A2BE2)](https://www.langchain.com/langgraph)
[![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)]()

AODMA is a research-oriented, multi-agent framework for investigating resource-efficient diffusion model inference.

The project combines a budget-aware sampler selection algorithm, diffusion benchmarking infrastructure, and autonomous AI agents to plan, execute, evaluate, and document experiments.

The primary research direction is **BQRPS (Budgeted Quality-Risk Portfolio Selection)**, a proposed algorithm for selecting diffusion sampler configurations under quality, latency, memory, and inference-budget constraints.

> Research status: The algorithm and its novelty are under investigation. All claimed improvements will be established through controlled experiments and comparisons with existing methods.

---

## 1. Research Motivation

Diffusion probabilistic models produce high-quality generative outputs but can require numerous neural network evaluations during inference.

Existing research investigates several approaches to improving sampling efficiency:

- Optimized timestep schedules.
- Higher-order numerical solvers.
- Adaptive solver selection.
- Error-controlled sampling.
- Inference-time compute allocation.
- Quality-aware inference serving.

However, selecting an appropriate sampler configuration under practical resource constraints remains an important systems problem to investigate.

A configuration that provides excellent image quality may exceed the available latency or memory budget. Another configuration may be faster but produce less reliable results.

AODMA investigates whether a statistically calibrated, uncertainty-aware sampler portfolio can make more reliable inference decisions.

## 2. Research Objectives

1. Implement and benchmark established diffusion sampling algorithms.
2. Develop BQRPS for resource-constrained sampler configuration selection.
3. Investigate uncertainty-aware quality and latency estimation.
4. Evaluate sampler configurations across multiple datasets and NFE budgets.
5. Develop a multi-agent system for autonomous experiment orchestration.
6. Build a production-oriented application for experiment management and visualization.

## 3. Proposed Algorithm: BQRPS

**Budgeted Quality-Risk Portfolio Selection**

BQRPS investigates selecting a sampler configuration from a pre-profiled portfolio based on quality estimates, uncertainty, latency, memory consumption, and available inference budget.

A configuration is represented as:

\[
c=(s,\tau,n,p)
\]

Where:

- \(s\): Sampling algorithm.
- \(\tau\): Timestep schedule.
- \(n\): Number of neural network evaluations.
- \(p\): Numerical precision.

### Quality-risk estimation

For quality metrics where lower is better:

\[
R_Q(c)=\hat Q_c+\beta_Q\sigma_{Q,c}
\]

Latency-risk estimation:

\[
R_T(c)=\hat T_c+\beta_T\sigma_{T,c}
\]

The proposed selection objective is:

\[
c^*=\arg\min_{c\in\mathcal C}R_Q(c)
\]

Subject to:

\[
R_T(c)\leq T_{\max}
\]

\[
M_c\leq M_{\max}
\]

\[
NFE(c)\leq B_{\max}
\]

Here, uncertainty penalties are intended to discourage configurations with unreliable predicted performance.

The final algorithm will include calibration, statistical validation, feasibility handling, and comparisons against simpler configuration-selection strategies.

### Proposed algorithm workflow

1. Construct a portfolio of sampler configurations.
2. Profile configurations on the target hardware.
3. Collect quality, latency, memory, and NFE measurements.
4. Estimate performance uncertainty.
5. Filter configurations that violate resource constraints.
6. Select the feasible configuration with the lowest quality-risk estimate.
7. Monitor observed performance and update calibration statistics.

## 4. Multi-Agent Architecture

AODMA uses a LangGraph-based orchestration system.

| Agent | Responsibility |
|---|---|
| Orchestrator Agent | Coordinates the workflow and maintains experiment state |
| Research Agent | Retrieves research literature and identifies candidate baselines |
| Strategy Agent | Proposes experiment configurations and sampling strategies |
| Execution Agent | Runs experiments on the local GPU |
| Evaluation Agent | Evaluates metrics and validates experimental results |
| Reporting Agent | Generates experiment summaries and research reports |

### Agent workflow

Researcher → Orchestrator → Research / Strategy → Execution → Evaluation → Reporting

The LLM agents operate at the experiment level. The actual diffusion inference and BQRPS selection logic remain deterministic Python/PyTorch components.

No LLM calls are made inside the denoising loop.

## 5. Research Baselines

The initial benchmark portfolio includes:

- DDPM
- DDIM
- DPM-Solver++
- UniPC
- EDM Heun
- Optimized timestep schedules
- Existing adaptive solver methods

Important related work:

1. Ho et al. — Denoising Diffusion Probabilistic Models (2020).
2. Xue et al. — Accelerating Diffusion Sampling with Optimized Time Steps (CVPR 2024).
3. Jo and Choi — Formalizing the Sampling Design Space of Diffusion-Based Generative Models (2026).
4. Kemperman and Ambrogioni — Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling (2026).
5. Cao et al. — Leveraging Inference-Time Compute for Diffusion Models via Global Scheduling of Denoising Trajectories (2026).
6. QoS-Diff — Adaptive Auto-tuning Framework for Low-latency Diffusion Model Inference.

These works will be examined to establish the precise novelty gap and avoid duplicating existing contributions.

## 6. Experimental Methodology

### Datasets

| Dataset | Purpose |
|---|---|
| MNIST | Mathematical and implementation validation |
| Fashion-MNIST | Secondary generalization benchmark |
| CIFAR-10 | Main natural-image benchmark |

### Inference budgets

Initial NFE budgets:

`5, 10, 20, 30, 50`

### Evaluation metrics

| Category | Metrics |
|---|---|
| Image quality | FID, KID |
| Diversity | Precision and Recall |
| Inference efficiency | NFE, wall-clock latency |
| Hardware utilization | Peak GPU memory |
| Selection reliability | Constraint violation rate |
| Calibration | Prediction error and uncertainty coverage |
| Agent performance | Execution success, human intervention, API cost |

Final experiments will use consistent checkpoints, preprocessing, sample counts, evaluation implementations, and random seeds.

### Ablation studies

- BQRPS without quality uncertainty.
- BQRPS without latency uncertainty.
- BQRPS without memory constraints.
- BQRPS without online recalibration.
- BQRPS without cross-dataset calibration.
- Complete BQRPS.

Comparisons will include fixed baselines, best-average selection, latency-first selection, Pareto selection, and the complete proposed method.

## 7. Hardware and Compute Environment

Initial development environment:

| Component | Configuration |
|---|---|
| GPU | NVIDIA GTX 1650 |
| RAM | 8 GB |
| Execution | Local-first |
| Agent intelligence | API-based LLMs |
| Training strategy | Small diffusion models |
| Experiment execution | Sequential |

The project will prioritize memory-efficient training, small U-Net architectures, batch-size optimization, and reproducible experiments.

Large pretrained diffusion models and simultaneous GPU workloads are outside the initial scope.

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Diffusion models | Hugging Face Diffusers |
| Agent orchestration | LangGraph |
| Agent intelligence | API-based LLM |
| Backend | FastAPI |
| Frontend | React + Vite |
| Experiment tracking | MLflow |
| Database | SQLite |
| Configuration | YAML |
| Testing | Pytest |
| Visualization | Matplotlib / Plotly |

## 9. Project Structure

```text
AODMA/
├── agents/
│   ├── orchestrator.py
│   ├── research_agent.py
│   ├── strategy_agent.py
│   ├── execution_agent.py
│   ├── evaluation_agent.py
│   └── reporting_agent.py
│
├── diffusion/
│   ├── models/
│   ├── forward_process.py
│   ├── noise_schedule.py
│   └── training.py
│
├── samplers/
│   ├── ddpm.py
│   ├── ddim.py
│   ├── dpm_solver.py
│   ├── unipc.py
│   └── adaptive_sampler.py
│
├── bqrps/
│   ├── portfolio.py
│   ├── profiler.py
│   ├── calibration.py
│   ├── uncertainty.py
│   └── selector.py
│
├── experiments/
│   ├── configs/
│   ├── baseline/
│   └── ablations/
│
├── evaluation/
│   ├── fid.py
│   ├── kid.py
│   ├── latency.py
│   └── memory.py
│
├── backend/
├── frontend/
├── tracking/
├── tests/
├── checkpoints/
├── results/
├── notebooks/
├── paper/
└── README.md
```

## 10. Implementation Roadmap

| Phase | Objective |
|---|---|
| 1 | Environment setup and DDPM implementation |
| 2 | Baseline sampler implementations |
| 3 | BQRPS algorithm development |
| 4 | Calibration and statistical validation |
| 5 | Multi-agent orchestration |
| 6 | FastAPI and React application |
| 7 | Cross-dataset evaluation |
| 8 | Research analysis and manuscript preparation |

## 11. Expected Research Contributions

The project aims to investigate three contributions:

1. A statistically risk-aware sampler portfolio selection framework.
2. A systematic resource-constrained evaluation of diffusion sampling configurations.
3. A multi-agent experimentation platform for reproducible generative AI research.

The validity and novelty of these contributions will be determined through literature comparison and experimental evidence.

## 12. Reproducibility

Every experiment should record:

- Experiment ID.
- Dataset and checkpoint.
- Sampler and timestep schedule.
- NFE budget.
- Random seed.
- GPU and software environment.
- Image quality metrics.
- Latency and memory measurements.
- Configuration and agent decisions.

The project will prioritize reproducible configurations, transparent evaluation, and version-controlled experimental results.

## 13. Research Status

**Current stage:** Initial implementation and research validation.

Planned first milestone:

- Verify local CUDA and PyTorch.
- Train a small MNIST DDPM.
- Generate initial samples.
- Save a reproducible checkpoint.
- Establish the baseline experiment runner.

---

**AODMA — Researching intelligent, resource-aware diffusion inference.**
