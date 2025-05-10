# SIGIL - Systemic Insight Generator for Irregularity & Leakage

**"She stared at chaos and noticed it was wearing a hat."**

## Philosophy: Interrogating Randomness

Conventional entropy analysis often stops at statistical uniformity. If data *looks* random according to standard tests (like NIST/FIPS), it's deemed sufficient. SIGIL operates on a different, more fundamental premise: **uniformity enforced upon complexity can reveal hidden structures.**

This philosophy stems from direct observation: meticulously inspecting large bit visualizations revealed subtle but persistent geometric artifacts (radial patterns, faint grid-like structures) that appeared *more* clearly in whitened/balanced data – data already verified as statistically near-perfect (.999982 Shannon bits/bit) – compared to the visually noisier raw data. This suggested processing didn't erase all structure but perhaps shifted its visual manifestation.

Think of it like trying to perfectly flatten a complex sand mound. The very act of enforcing uniformity might make the underlying construction method or remaining imperfections *more* apparent than they were in the original chaotic state. Or consider seeing shapes in clouds; while the brain seeks patterns (paradolia), recognizing a shape doesn't negate the underlying physical structure creating that appearance. Similarly, the geometric artifacts observed in processed entropy, while perhaps subtle, represent real structural correlations revealed or induced by processing, not mere illusions.

The demand for statistically perfect randomness, especially through processes like cryptographic whitening or balancing, doesn't necessarily erase all structural information; it may transform it into subtle visual meta-patterns or "ghosts." SIGIL rejects the notion that statistical uniformity equates to structural patternlessness.

SIGIL is designed not just to *validate* randomness according to classical tests, but to *interrogate* it deeply, visually, and structurally. It seeks the hidden "dialects" entropy might be screaming even when statistically quiet, the "ghost ripples" that standard tests miss. It operates on the principle that the drive for cryptographic perfection likely *will* induce or unveil these subtle meta-patterns, which could represent exploitable information leakage.

SIGIL is therefore a framework for **Entropy Forensics** and **Visual Phase Cryptanalysis**, built to hunt for the imperfections, induced patterns, and leakages that hide beneath the veil of apparent uniformity.

## Core Goals

*   **Detect Meta-Patterns:** Identify subtle structures potentially induced by whitening, compression, or other processing, even in statistically uniform data.
*   **Multi-Modal Analysis:** Combine visual (bitmap rendering, *direct inspection*, zooming, comparison), statistical (frequency, runs, chi-square), and transform-based (FFT, Wavelet) methods.
*   **Benchmark & Compare:** Analyze outputs from various entropy sources (ERIS:raw, ERIS:full, other QRNGs, PRNGs) to highlight differences in both statistical and structural characteristics.
*   **Modular & Extensible:** Provide a pluggable architecture for adding new analysis techniques.
*   **Expose Leakage:** Quantify potential structural anomalies, uniformity deviations, and "whiteness leakage" revealed through diverse analytical lenses.

## Architecture

The framework is organized as follows:

*   `analyzers/`: Contains modules for specific analysis techniques (FFT, Wavelet, Stats, Visual).
*   `loaders/`: Handles loading entropy data from various formats (bitstreams, images, numeric sequences).
*   `report/`: Generates summaries, scores, and visualizations from the analysis modules.
*   `utils/`: Shared helper functions and tools (entropy scoring, transform helpers).
*   `examples/`: Demonstrates usage and benchmarking against different entropy sources.
*   `decomposition/`: *(Note: The user provided `decomposition/visual.py`. This might be intended to be under `analyzers/`. Confirm structure if needed.)*

## Core Features (v0.1 - Planned)

*   **FFT Analysis:** Log-magnitude spectrum visualization (`analyzers/fft.py`).
*   **Wavelet Decomposition:** Multi-level wavelet analysis (`analyzers/wavelet.py`).
*   **Visual Rendering:** Bit visualization generation (`decomposition/visual.py` or `analyzers/visual.py`).
*   **Statistical Tests:** Basic frequency, runs, chi-square tests (`analyzers/stats.py`).
*   **Entropy Input Loading:** Loading data from files/streams (`loaders/entropy_input.py`).
*   **Basic Reporting:** Summary generation (`report/summary.py`).

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd SIGIL
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore examples:**
    Review `examples/benchmark_run.py` to see how to load data and run analyses.

## Contributing

SIGIL welcomes contributions that enhance its ability to interrogate randomness and uncover hidden structures. This includes new analysis techniques, improved visualizations, and robust benchmarking methodologies. The goal is to build a toolkit for those who believe perfection might be a lie, and that patternless chaos has its own subtle logic.
