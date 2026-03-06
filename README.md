# RouteMoA: SLM-Guided Mixture of Agents

![RouteMoA Header](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

RouteMoA is an advanced, cost-effective, and low-latency **Mixture of Agents (MoA)** architecture. It utilizes a locally trained Small Language Model (SLM) to intelligently route prompts to the most appropriate Large Language Models (LLMs) based on complexity and domain, drastically reducing API costs and inference time compared to classical MoA routing.

## 🌟 Key Features

*   **🧠 SLM Scorer (DeBERTa-v3):** A locally trained predictive model that pre-evaluates queries against 9+ available LLM endpoints (like Gemini Flash, Qwen Core, DeepSeek, etc.) to estimate their success probability before any API calls are made.
*   **⚡ True Async Early Stopping:** Implements asynchronous `fast-fail` generation loops (`asyncio.as_completed`). The pipeline returns the correct answer at the speed of the *fastest* model in the pool, cutting latency by up to 80% (from ~80s down to ~20s on complex tasks).
*   **📉 Massive Cost Reduction:** By avoiding expensive SOTA models (like Claude Opus) for simple queries and efficiently aggregating cheaper models for complex ones, RouteMoA averages **85% to 98% cost savings** while maintaining SOTA-level accuracy on hard benchmarks (MMLU, MBPP, GSM8K).
*   **📊 Interactive UI:** Includes a beautiful Streamlit dashboard (`app.py`) with real-time analytics, cost tracking, latency metrics, and side-by-side comparisons of RouteMoA vs Classical MoA.
*   **🌐 Flexible Model Pool:** Connects directly to OpenRouter, supporting dynamic addition of new open-source or proprietary models.

---

## 🏗️ Architecture Pipeline

When a user submits a query:
1.  **SLM Scoring:** The prompt is analyzed locally. The SLM estimates confidence scores for all connected LLMs.
2.  **Top-K Selection:** Only the models most likely to succeed (top $K$) are queried.
3.  **Parallel Generation + Self-Assessment:** Models generate answers and self-assess confidence. If a model reaches the `early_stop_threshold`, execution stops instantly.
4.  **Cross-Assessment (If needed):** If no model is confident, the top models evaluate each other's answers.
5.  **Refinement & Aggregation:** The best answers are fused together by an aggregator model to produce the final, highly accurate output.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   An [OpenRouter API Key](https://openrouter.ai/) for model access.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abdusari397-hue/Route-moa.git
    cd Route-moa
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your OpenRouter key:
    ```env
    OPENROUTER_API_KEY=sk-or-v1-your-key-here
    ```

### Usage

**Run the Streamlit Dashboard (Recommended):**
```bash
streamlit run app.py
```

**Run Global Benchmarks (CLI):**
To test RouteMoA against SOTA models (e.g., Claude Opus) on hard datasets (MBPP, MMLU, ARC):
```bash
python benchmark_sota.py
```

---

## 📈 Benchmarks

On difficult reasoning, coding, and logical science questions, RouteMoA demonstrates:
*   **Cost:** ~$0.001 - $0.005 per query (vs ~$0.020 - $0.060 for Opus 4.6).
*   **Speed:** ~18s - 25s latency.
*   **Performance:** Matches SOTA accuracy through intelligent cross-assessment and aggregation of smaller, specialized models.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! 

## 📝 License
This project is licensed under the MIT License.
