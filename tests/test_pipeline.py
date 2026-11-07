"""Unit test suite for Lab 03 - Chaotic Text Generator.
Tests cover HMM, Role-Splitting, Viterbi decoding, Trigram generator, and Evaluation metrics.
Run: python -m unittest discover -s tests
"""
import os
import random
import sys
import unittest

# Ensure src is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import (
    HMM,
    NGramGenerator,
    coarse,
    generation_metrics,
    load_tagged,
    oov_rate,
    refine,
    sentence_features,
    split_by_document,
)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus_path = os.path.join(BASE_DIR, "data", "corpus.csv")
        cls.docs = load_tagged(corpus_path)
        cls.train, cls.test = split_by_document(cls.docs)
        cls.hmm_plain = HMM(cls.train, split_states=False)
        cls.hmm_split = HMM(cls.train, split_states=True)
        cls.ngram = NGramGenerator([w for w, _ in cls.train])
        cls.train_patterns = {" ".join(t) for _, t in cls.train}

    def test_corpus_splitting(self):
        """Verify train/test split by document (960 train, 240 test, no leakage)."""
        self.assertEqual(len(self.train), 960)
        self.assertEqual(len(self.test), 240)
        self.assertEqual(len(self.train) + len(self.test), 1200)

    def test_state_refinement_and_coarse(self):
        """Verify refine() assigns S, O, P roles and coarse() reverses to basic POS tags."""
        tags = ["DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN", "PUNCT"]
        refined = refine(tags)
        expected = ["DET_S", "NOUN_S", "VERB", "DET_O", "NOUN_O", "PREP", "DET_P", "NOUN_P", "PUNCT"]
        self.assertEqual(refined, expected)
        reverted = coarse(refined)
        self.assertEqual(reverted, tags)

    def test_viterbi_synthetic_accuracy(self):
        """Verify that Viterbi achieves 100% accuracy on synthetic test set and resolves 'watch'."""
        correct = total = watch_correct = watch_total = 0
        for words, gold in self.test:
            pred = self.hmm_plain.viterbi(words)
            self.assertEqual(len(pred), len(gold))
            for w, g, p in zip(words, gold, pred):
                total += 1
                if g == p:
                    correct += 1
                if w == "watch":
                    watch_total += 1
                    if g == p:
                        watch_correct += 1
        self.assertEqual(correct, total)
        self.assertEqual(watch_correct, watch_total)
        self.assertEqual(watch_total, 37)

    def test_viterbi_oov_graceful_handling(self):
        """Verify Viterbi handles out-of-vocabulary words without crashing."""
        oov_sentence = ["the", "cybernetic", "alien", "reads", "a", "holocron", "in", "the", "starship", "."]
        pred_tags = self.hmm_split.viterbi(oov_sentence)
        self.assertEqual(len(pred_tags), len(oov_sentence))
        self.assertIn("PUNCT", pred_tags)

    def test_role_split_generation_valid_patterns(self):
        """Verify that Role-Split HMM achieves 100% valid tag patterns."""
        rng = random.Random(42)
        samples = [self.hmm_split.sample_sentence(rng, temperature=1.0) for _ in range(100)]
        for words, tags in samples:
            pattern = " ".join(tags)
            self.assertIn(pattern, self.train_patterns)

    def test_trigram_generation(self):
        """Verify that Trigram generates coherent sentences ending with PUNCT."""
        rng = random.Random(42)
        words = self.ngram.sample_sentence(rng, temperature=1.0)
        self.assertTrue(len(words) > 0)
        self.assertEqual(words[-1], ".")

    def test_metrics_computation(self):
        """Verify generation_metrics returns all 4 required metrics within valid ranges."""
        rng = random.Random(42)
        samples = [self.hmm_split.sample_sentence(rng, temperature=1.0) for _ in range(50)]
        metrics = generation_metrics(samples, self.train, self.hmm_split)
        self.assertIn("valid_tag_pattern", metrics)
        self.assertIn("plausible_subject_verb", metrics)
        self.assertIn("novel_sentences", metrics)
        self.assertIn("distinct_2", metrics)
        self.assertAlmostEqual(metrics["valid_tag_pattern"], 1.0)
        self.assertTrue(0.0 <= metrics["plausible_subject_verb"] <= 1.0)
        self.assertTrue(0.0 <= metrics["novel_sentences"] <= 1.0)
        self.assertTrue(0.0 < metrics["distinct_2"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
