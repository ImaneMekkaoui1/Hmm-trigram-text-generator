# Lab 03: Chaotic Text Generator — Test Bench for Sequence Models

Interactive web test bench comparing a **first-order Hidden Markov Model (HMM)**, a **Role-Split HMM**, and a **Trigram baseline** for part-of-speech text generation and sequence labelling (Viterbi decoding).

Developed as part of the NLP / SDIA curriculum.

---

## 🌟 Overview & Key Concepts

1. **First-Order POS-HMM**: Hidden states are parts-of-speech (DET, NOUN, VERB, ADJ, PREP, PUNCT). Observations are words. Evaluated by counting transitions $P(t_i | t_{i-1})$ and emissions $P(w_i | t_i)$.
2. **Structural Memory via Role-Splitting**: Plain HMM mixes subject and object nouns, collapsing syntax. Role-splitting partitions nouns, adjectives, and determiners by clause role (`_S` for subject, `_O` for object, `_P` for prepositional phrase), raising valid syntactic patterns from **16.2% to 100.0%**.
3. **Trigram Baseline**: Lexical n-gram model with back-off. High subject-verb plausibility (**76.0%**) but lower syntactic consistency (**57.0%**) and vulnerability to training memorisation (**8.8%** copied verbatim at $T=1.0$).
4. **Viterbi Tagger**: Log-space dynamic programming with add-$k$ smoothing ($k=0.01$). Resolves grammatical ambiguity (e.g., *watch* as NOUN vs VERB) and gracefully handles Out-of-Vocabulary (OOV) tokens.
5. **Real-Text Benchmark (UD English EWT)**: Tested on 12,544 training sentences and 2,077 test sentences with 17 UPOS tags. Demonstrates the limitations of first-order Markov models on natural language (88.1% accuracy vs 86.2% baseline; 28.2% on OOV tokens).

---

## 📁 Repository Structure

| Path | Purpose |
|---|---|
| `app.py` | Full-featured Streamlit web application & test bench |
| `src/pipeline.py` | Core model implementations: HMM, role-splitting, Viterbi, Trigram, metrics |
| `src/evaluate.py` | Synthetic corpus evaluation script (4 metrics across 4 temperatures) |
| `src/evaluate_real.py` | Real-world benchmark on Universal Dependencies English EWT |
| `data/corpus.csv` | Synthetic tagged training corpus (1,200 sentences in 60 documents) |
| `data/make_data.py` | Generator script for the synthetic corpus |
| `results/` | Precomputed evaluation logs, JSON metrics, failure cases, and plots |
| `screenshot_generation.png` | Application screenshot: Text Generation & Batch Metrics |
| `screenshot_tagger.png` | Application screenshot: Interactive Viterbi Tagger with OOV handling |
| `requirements.txt` | Python dependencies for local execution and Streamlit Cloud |

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the evaluations
```bash
python -m src.evaluate       # Generates results/metrics.json and results/hmm_results.png
python -m src.evaluate_real  # Downloads UD English EWT and generates results/ewt_results.json
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Public Deployment (Streamlit Community Cloud)

1. Create a public repository on GitHub (e.g. `chaotic-text-generator`).
2. Push this directory:
   ```bash
   git add .
   git commit -m "Initial release: Chaotic text generator test bench"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, select your repository, set the main file path to `app.py`, and click **Deploy**.

---

## 📊 Summary of Benchmark Results

### Synthetic Corpus ($T = 1.0, N = 500$)
| Model | Valid Tag Pattern | Plausible Subject-Verb | Novel Sentences | Distinct-2 |
|---|:---:|:---:|:---:|:---:|
| **Plain HMM** | 16.2% | 14.8% | 99.8% | 0.198 |
| **Role-Split HMM** | **100.0%** | 60.6% | 99.4% | 0.148 |
| **Trigram** | 57.0% | **76.0%** | 91.2% | 0.105 |

### Universal Dependencies English EWT (Real Web Text, 17 UPOS Tags)
| Model | Overall Accuracy | Known Words | Unknown Words (OOV) | Full Sentences Correct |
|---|:---:|:---:|:---:|:---:|
| **Most-Frequent Tag (Baseline)** | 86.2% | 90.8% | 28.6% | 31.9% |
| **HMM + Viterbi** | **88.1%** | **92.9%** | 28.2% | **40.4%** |

---

## ⚠️ Synthetic Corpus Notice
*The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English.*
