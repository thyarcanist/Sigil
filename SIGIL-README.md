# SIGIL - Systemic Insight Generator for Irregularity & Leakage

**"She stared at chaos and noticed it was wearing a hat."**

## Philosophy: Interrogating Randomness

Conventional entropy analysis often stops at statistical uniformity. If it *looks* random according to standard tests (like FIPS), it's deemed sufficient. SIGIL operates on a different premise: **perfect uniformity might be an illusion**, potentially masking subtle, induced meta-patterns or "compression ghosts" – a form of entropy gentrification.

Inspired by the visual detection of unexpected structural anomalies (radial symmetries, morphological phase shifts) in high-quality, whitened entropy streams benchmarked near theoretical perfection (.999982), SIGIL is designed not just to *validate* randomness, but to *interrogate* it. It seeks the hidden "dialects" entropy might be screaming, the "ghost ripples" that standard tests miss.

SIGIL is a framework for **Entropy Forensics** and **Visual Phase Cryptanalysis**, built to hunt for the imperfections and leakages that might hide beneath the veil of apparent uniformity.

## Core Goals

*   **Detect Meta-Patterns:** Identify subtle structures potentially induced by whitening, compression, or other processing.
*   **Multi-Modal Analysis:** Combine visual (bitmap rendering, zoom analysis), statistical (frequency, runs, chi-square), and transform-based (FFT, Wavelet) methods.
*   **Benchmark & Compare:** Analyze outputs from various entropy sources (ERIS:raw, ERIS:full, other QRNGs, PRNGs) to highlight differences.
*   **Modular & Extensible:** Provide a pluggable architecture for adding new analysis techniques.
*   **Expose Leakage:** Quantify potential structural anomalies, uniformity deviations, and "whiteness leakage".

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
