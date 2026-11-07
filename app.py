"""Lab 03 - Chaotic Text Generator (Streamlit UI Test Bench).
Run: streamlit run app.py
"""
import io
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import (
    HMM,
    NGramGenerator,
    generation_metrics,
    load_tagged,
    sentence_features,
    split_by_document,
)

st.set_page_config(
    page_title="Chaotic Text Generator - HMM vs Trigram Test Bench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ Model Caching
@st.cache_resource
def load_all():
    corpus_path = os.path.join(BASE_DIR, "data", "corpus.csv")
    docs = load_tagged(corpus_path)
    train, test = split_by_document(docs)
    hmm_plain = HMM(train, split_states=False)
    hmm_split = HMM(train, split_states=True)
    ngram = NGramGenerator([w for w, _ in train])
    train_patterns = {" ".join(t) for _, t in train}
    train_pairs = {sentence_features(w, t)[1] for w, t in train}
    train_sentences_text = {" ".join(w) for w, _ in train}
    return {
        "train": train,
        "test": test,
        "hmm_plain": hmm_plain,
        "hmm_split": hmm_split,
        "ngram": ngram,
        "train_patterns": train_patterns,
        "train_pairs": train_pairs,
        "train_sentences_text": train_sentences_text,
    }


data = load_all()
hmm_plain = data["hmm_plain"]
hmm_split = data["hmm_split"]
ngram = data["ngram"]
train = data["train"]
train_patterns = data["train_patterns"]

# ------------------------------------------------------------------ Header & Notice
st.title("Chaotic Text Generator — HMM & Trigram Test Bench")
st.markdown(
    "A sequence modeling and generation test bench comparing a **first-order Hidden Markov Model (HMM)**, "
    "a **role-split HMM (S / O / P)**, and a **trigram language model** with temperature sampling and Viterbi decoding."
)

# Mandatory Notice (Requirement 6)
st.warning(
    "**Corpus Notice:** The corpus is synthetic and the grammar is tiny. "
    "Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English."
)

# ------------------------------------------------------------------ Sidebar Control Panel (Requirement 1)
st.sidebar.header("Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["Plain HMM", "Role-Split HMM", "Trigram (3-gram)"],
    index=1,
    help="Plain HMM has coarse POS states. Role-split HMM tags states by syntactic role (Subject, Object, Prepositional). Trigram models word sequences directly.",
)

temperature = st.sidebar.slider(
    "Temperature (T)",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="T < 1.0 sharpens distribution (conservative, repetitive); T > 1.0 flattens distribution (chaotic, diverse).",
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=100_000,
    value=42,
    step=1,
    help="Fixed seed ensures exact reproducibility of generation.",
)

num_sentences = st.sidebar.slider(
    "Number of Sentences",
    min_value=10,
    max_value=500,
    value=50,
    step=10,
    help="Sample size to evaluate metrics robustly.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Corpus Statistics:**  \n"
    f"- Training sentences: **{len(train)}**  \n"
    f"- Vocabulary size: **{len(hmm_plain.vocab)}** words  \n"
    f"- POS tags: **{len(hmm_plain.tags)}** ({', '.join(hmm_plain.tags)})  \n"
    f"- Valid tag patterns: **{len(train_patterns)}**"
)

# ------------------------------------------------------------------ Main Interface Tabs
tab_gen, tab_tagger, tab_compare, tab_grammar = st.tabs(
    ["Text Generation & Batch Metrics", "Interactive Viterbi Tagger", "Model Comparison & Plots", "Grammar & Transition Heatmap"]
)

# ------------------------------------------------------------------ TAB 1: Generation & Metrics
with tab_gen:
    st.subheader(f"Generate Text with {model_choice} (T = {temperature:.1f}, Seed = {seed})")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        generate_clicked = st.button("Generate Batch", type="primary", use_container_width=True)

    # Use session state to persist generated results
    if generate_clicked or "batch_results" not in st.session_state:
        rng = random.Random(int(seed))
        samples = []  # list of (words, tags)
        hidden_states_list = []

        with st.spinner("Generating sentences and computing metrics..."):
            if model_choice == "Plain HMM":
                for _ in range(num_sentences):
                    words, states = hmm_plain.sample_with_states(rng, temperature)
                    samples.append((words, states))
                    hidden_states_list.append(" -> ".join(states))
            elif model_choice == "Role-Split HMM":
                for _ in range(num_sentences):
                    words, states = hmm_split.sample_with_states(rng, temperature)
                    coarse_tags = [s.split("_")[0] for s in states]
                    samples.append((words, coarse_tags))
                    hidden_states_list.append(" -> ".join(states))
            else:  # Trigram
                for _ in range(num_sentences):
                    words = ngram.sample_sentence(rng, temperature)
                    tags = hmm_split.viterbi(words)
                    samples.append((words, tags))
                    hidden_states_list.append(" -> ".join(tags))

            batch_metrics = generation_metrics(samples, train, hmm_split)

            rows = []
            for i, ((words, tags), h_states) in enumerate(zip(samples, hidden_states_list), 1):
                sent_str = " ".join(words)
                tag_pattern = " ".join(tags)
                is_valid = tag_pattern in train_patterns
                rows.append({
                    "ID": i,
                    "Sentence": sent_str,
                    "Hidden Tag Sequence": h_states,
                    "Valid Pattern": "Valid" if is_valid else "Invalid",
                    "_valid_bool": is_valid,
                })

            df_results = pd.DataFrame(rows)
            st.session_state["batch_results"] = df_results
            st.session_state["batch_metrics"] = batch_metrics
            st.session_state["current_model"] = model_choice
            st.session_state["current_T"] = temperature

    # Display Metrics Cards (Requirement 3)
    if "batch_metrics" in st.session_state:
        m = st.session_state["batch_metrics"]
        st.markdown("### Batch Quality Metrics (N = {})".format(len(st.session_state["batch_results"])))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Valid Tag Pattern",
            value=f"{m['valid_tag_pattern']:.1%}",
            help="Fraction of sentences whose POS tag sequence matches one of the 16 grammar patterns.",
        )
        c2.metric(
            label="Plausible Subject-Verb",
            value=f"{m['plausible_subject_verb']:.1%}",
            help="Fraction of sentences where (Subject Noun, Verb) co-occurred in the training corpus.",
        )
        c3.metric(
            label="Novel Sentences",
            value=f"{m['novel_sentences']:.1%}",
            help="Fraction of generated sentences not present verbatim in training data (1 - memorisation).",
        )
        c4.metric(
            label="Distinct-2 (Diversity)",
            value=f"{m['distinct_2']:.3f}",
            help="Unique word bigrams divided by total word bigrams. Higher means greater lexical diversity.",
        )

        st.markdown("---")

        # Display Generated Table (Requirement 2)
        df_display = st.session_state["batch_results"]
        st.markdown(f"### Generated Sentences ({len(df_display)} rows)")

        show_filter = st.radio(
            "Filter rows:",
            options=["All Sentences", "Valid Only", "Invalid Only"],
            horizontal=True,
        )
        if show_filter == "Valid Only":
            filtered_df = df_display[df_display["_valid_bool"] == True]
        elif show_filter == "Invalid Only":
            filtered_df = df_display[df_display["_valid_bool"] == False]
        else:
            filtered_df = df_display

        st.dataframe(
            filtered_df[["ID", "Sentence", "Hidden Tag Sequence", "Valid Pattern"]],
            use_container_width=True,
            height=350,
        )

        # Download CSV Button (Requirement 5)
        csv_buffer = io.StringIO()
        df_display[["ID", "Sentence", "Hidden Tag Sequence", "Valid Pattern"]].to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Generated Sentences as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"generated_{st.session_state['current_model'].replace(' ', '_').lower()}_T{st.session_state['current_T']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ------------------------------------------------------------------ TAB 2: Viterbi Tagger (Requirement 4)
with tab_tagger:
    st.subheader("Interactive Sequence Labelling with Viterbi Algorithm")
    st.markdown(
        "Type any English sentence below. The Viterbi dynamic programming decoder computes the most probable "
        "part-of-speech tag sequence in log-space with add-k smoothing (k = 0.01)."
    )

    tagger_model_choice = st.radio(
        "Tagger Model Architecture:",
        options=["Plain HMM", "Role-Split HMM (Refined States)"],
        horizontal=True,
        help="Both taggers map output to coarse POS tags, but Role-Split tracks syntactic roles internally.",
    )
    active_tagger = hmm_split if "Role-Split" in tagger_model_choice else hmm_plain

    default_examples = [
        "the old children watch the birds at the kitchen .",
        "the young children bring the watch behind the school .",
        "the cats sleep in the quiet garden .",
        "the hungry alien reads a book in the spaceship .",
    ]
    selected_example = st.selectbox("Or choose a pre-filled example sentence:", ["(Custom input)"] + default_examples)

    if selected_example != "(Custom input)":
        user_input = st.text_input("Sentence to tag:", value=selected_example)
    else:
        user_input = st.text_input("Sentence to tag:", value="the girls watch the dogs in the park .")

    if user_input.strip():
        # Clean & tokenize sentence with punctuation separation
        tokens = re.findall(r"[\w']+|[.,!?;]", user_input.strip().lower())
        
        # Check for unknown / out-of-vocabulary words
        unknown_words = [w for w in tokens if w not in active_tagger.vocab]

        if unknown_words:
            st.warning(
                f"**Out-of-Vocabulary (OOV) Warning:** The word(s) `{', '.join(unknown_words)}` "
                f"were not seen in the training vocabulary. Viterbi tags them using add-k smoothed emission probabilities: P(w | t) = k / (count(t) + k * |V|)."
            )
        else:
            st.success("All words are in the training vocabulary.")

        # Compute Viterbi decoding without crashing
        try:
            predicted_tags = active_tagger.viterbi(tokens)

            st.markdown("#### Predicted Sequence Alignment:")
            
            # Visual presentation of word -> tag alignment
            cols = st.columns(min(len(tokens), 12))
            for i, (tok, tag) in enumerate(zip(tokens, predicted_tags)):
                col = cols[i % len(cols)]
                with col:
                    tag_color = {
                        "NOUN": "#e3f2fd",
                        "VERB": "#fce4ec",
                        "ADJ": "#f3e5f5",
                        "DET": "#e8f5e9",
                        "PREP": "#fff3e0",
                        "PUNCT": "#ede7f6",
                    }.get(tag, "#f5f5f5")
                    st.markdown(
                        f"""
                        <div style="background-color: {tag_color}; border: 1px solid #ccc; border-radius: 8px; padding: 8px; text-align: center; margin-bottom: 6px;">
                            <div style="font-size: 15px; font-weight: bold; color: #111;">{tok}</div>
                            <div style="font-size: 12px; color: #555; margin-top: 4px; font-weight: 600;">{tag}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Check if tag pattern is valid
            pattern_str = " ".join(predicted_tags)
            is_valid = pattern_str in train_patterns
            if is_valid:
                st.info(f"**Syntactic Pattern:** `{pattern_str}` matches a known training grammar pattern.")
            else:
                st.warning(f"**Syntactic Pattern:** `{pattern_str}` is unseen in the training grammar.")

            # Ambiguity highlight for "watch"
            if "watch" in tokens:
                idx = tokens.index("watch")
                w_tag = predicted_tags[idx]
                st.markdown(
                    f"**Ambiguity Resolution for 'watch':** In this context, *'watch'* is decoded as **{w_tag}**."
                )

        except Exception as e:
            st.error(f"Error during Viterbi decoding: {e}")

# ------------------------------------------------------------------ TAB 3: Model Comparison & Plots
with tab_compare:
    st.subheader("Comparative Benchmark: Plain HMM vs Role-Split HMM vs Trigram")
    st.markdown(
        "Compare all three models evaluated under identical conditions (same random seed, same sample size, multiple temperatures)."
    )

    if st.button("Run Full Comparison Benchmark (T = 0.5, 1.0, 1.5, 2.5)"):
        with st.spinner("Evaluating all models across 4 temperatures (300 sentences each)..."):
            comp_rows = []
            temps = [0.5, 1.0, 1.5, 2.5]
            plot_data = {"Plain HMM": [], "Role-Split HMM": [], "Trigram": []}

            for T in temps:
                r1 = random.Random(42)
                hs = [hmm_plain.sample_sentence(r1, T) for _ in range(300)]
                r2 = random.Random(42)
                h2 = [hmm_split.sample_sentence(r2, T) for _ in range(300)]
                r3 = random.Random(42)
                ns = [(ngram.sample_sentence(r3, T), None) for _ in range(300)]

                m_plain = generation_metrics(hs, train, hmm_plain)
                m_split = generation_metrics(h2, train, hmm_split)
                m_ng = generation_metrics(ns, train, hmm_split)

                plot_data["Plain HMM"].append(m_plain["valid_tag_pattern"])
                plot_data["Role-Split HMM"].append(m_split["valid_tag_pattern"])
                plot_data["Trigram"].append(m_ng["valid_tag_pattern"])

                for m_name, m_dict in [("Plain HMM", m_plain), ("Role-Split HMM", m_split), ("Trigram", m_ng)]:
                    comp_rows.append({
                        "Model": m_name,
                        "Temp (T)": T,
                        "Valid Pattern": f"{m_dict['valid_tag_pattern']:.1%}",
                        "Plausible Subj-Verb": f"{m_dict['plausible_subject_verb']:.1%}",
                        "Novelty": f"{m_dict['novel_sentences']:.1%}",
                        "Distinct-2": f"{m_dict['distinct_2']:.3f}",
                    })

            st.session_state["comp_df"] = pd.DataFrame(comp_rows)
            st.session_state["plot_data"] = plot_data

    if "comp_df" in st.session_state:
        st.dataframe(st.session_state["comp_df"], use_container_width=True)

        st.markdown("#### Valid Tag Pattern vs Temperature")
        plot_df = pd.DataFrame(st.session_state["plot_data"], index=[0.5, 1.0, 1.5, 2.5])
        st.line_chart(plot_df)

    else:
        st.markdown("#### Reference Benchmark Table (T = 1.0, N = 500):")
        ref_data = {
            "Model": ["Plain HMM", "Role-Split HMM", "Trigram (3-gram)"],
            "Valid Tag Pattern": ["16.2%", "100.0%", "57.0%"],
            "Plausible Subject-Verb": ["14.8%", "60.6%", "76.0%"],
            "Novel Sentences": ["99.8%", "99.4%", "91.2%"],
            "Distinct-2": ["0.198", "0.148", "0.105"],
        }
        st.table(pd.DataFrame(ref_data))

        st.markdown("#### Pre-computed Evaluation Summary:")
        img_chart_path = os.path.join(BASE_DIR, "results", "hmm_results.png")
        if os.path.exists(img_chart_path) and st.checkbox("Show saved results chart (results/hmm_results.png)"):
            st.image(img_chart_path, caption="Evaluation results: Transition Matrix & Valid Patterns")

# ------------------------------------------------------------------ TAB 4: Grammar & Transition Matrix Heatmap
with tab_grammar:
    st.subheader("Grammar Rules & Transition Matrix Heatmap")
    st.markdown(
        "The synthetic corpus is generated from 16 canonical tag patterns. "
        "A first-order HMM with plain states mixes pre-verbal subject nouns and post-verbal object nouns, causing broken grammar. "
        "Role-splitting solves this by splitting `NOUN`, `ADJ`, `DET` into `_S` (Subject), `_O` (Object), `_P` (Prepositional)."
    )

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("#### All 16 Valid Tag Patterns in Training Set:")
        st.code("\n".join(sorted(train_patterns)), language="text")

    with col_g2:
        st.markdown("#### Transition Probability Heatmap P(next | current):")
        tags, mat = hmm_plain.transition_matrix()
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(np.array(mat), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=9)
        for i in range(len(tags)):
            for j in range(len(tags)):
                val = mat[i][j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if val > 0.5 else "black")
        ax.set_title("First-Order HMM Transitions")
        ax.set_xlabel("Next Tag")
        ax.set_ylabel("Current Tag")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
