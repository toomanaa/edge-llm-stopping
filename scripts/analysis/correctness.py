#!/usr/bin/env python3
"""Shared correctness grading, used by both the offline labeler
(label_qa_v2_boundaries.py) and the live controller (live_controller.py)
so there is exactly one definition of "correct" -- not two copies that
can silently drift apart.

Replaces exact-substring-only matching for explain-type ("all" mode)
questions with a lightweight, dependency-free word-form match. Found
necessary during live-controller debugging: several responses correctly
explained a required concept but used a different word form than the
dataset's exact required string -- e.g. "align" instead of "alignment",
"generate" instead of "generator" -- and were being marked wrong purely
because of that surface mismatch, not because the explanation was
actually missing.

Factual ("any" mode) answers are NOT touched by this change and still
use exact substring matching: those are proper nouns and specific facts
(city names, book titles) where stemming would be inappropriate and
risks false positives (e.g. a stemmed "Pari" could spuriously match
unrelated words starting with "pari").
"""
import re


import re

_KNOWN_SUFFIXES = {
    "", "s", "es", "ed", "ing", "e", "er", "ers", "or", "ors",
    "ate", "ates", "ation", "ations", "ment", "ments",
    "tion", "tions", "sion", "sions", "ity", "ities",
    "ive", "ives", "al", "ous", "ly", "y",
}


def _shares_word_root(word, concept, min_common_prefix=5):
    """True if `word` and `concept` share a long common prefix AND
    whatever is left over on each side, after that shared prefix, is a
    recognizable English derivational suffix (or nothing) -- e.g.
    align/alignment share the prefix "align" with leftovers ""/"ment";
    generate/generator share "generat" with leftovers "e"/"or".

    This is deliberately stricter than plain prefix-overlap, which
    produces real false positives on ordinary English: "wind" is a
    (character) prefix of "window", and "star" of "start", but they are
    unrelated words. Requiring the leftover characters to be known
    suffixes (not just short) rejects those while still accepting real
    word-form variants. `min_common_prefix` guards against short
    accidental overlaps in unrelated short words; the tradeoff is that
    some genuine short-root pairs (e.g. cold/colder, common prefix only
    4 characters) fall just under this floor and are not matched --
    a known, accepted limitation, safer than the false-positive risk of
    a lower floor."""
    common = 0
    for a, b in zip(word, concept):
        if a == b:
            common += 1
        else:
            break
    if common < min_common_prefix:
        return False
    return word[common:] in _KNOWN_SUFFIXES and concept[common:] in _KNOWN_SUFFIXES


def _word_boundary_match(text_lower, phrase_lower):
    """Exact match, but respecting word boundaries -- unlike plain
    Python `in`, this does not treat "wind" as present just because it
    is spelled inside "window", or "paris" as present just because it
    is spelled inside "pariswich". A short, deliberately narrow
    allowance for a trailing "s" or "es" is included so that legitimate
    plurals of a correct factual answer still match (e.g. required
    answer "seismograph" against generated text "Seismographs measure
    earthquakes.") -- a real regression caught in testing: a hard
    boundary requirement alone rejects "seismographs" because there is
    no boundary between "seismograph" and its own trailing "s". This
    optional suffix is intentionally limited to plain pluralization,
    not the broader word-form matching used for explain concepts, since
    factual answers are names and specific facts where broader fuzzy
    matching would risk false positives."""
    pattern = r"\b" + re.escape(phrase_lower) + r"(?:s|es)?\b"
    return re.search(pattern, text_lower) is not None


def contains_concept(text_lower, concept_lower):
    """Word-form-aware check for whether `concept_lower` (a single
    required concept, already lowercased) appears in `text_lower`
    (already lowercased). Tries a word-boundary-respecting exact match
    first (handles multi-word concepts correctly too); falls back to
    per-word root matching for single-word concepts only."""
    if _word_boundary_match(text_lower, concept_lower):
        return True
    if " " in concept_lower:
        # multi-word concept and no exact match -- do not attempt fuzzy
        # matching across word boundaries, too easy to produce false
        # positives
        return False
    words = re.findall(r"[a-z]+", text_lower)
    return any(_shares_word_root(w, concept_lower) for w in words)


def is_correct(text, answers, match_mode):
    """match_mode 'any': at least one accepted answer string present,
    word-boundary-respecting exact match only (factual questions --
    proper nouns; word-form matching would be inappropriate here and
    risks false positives on names).
    match_mode 'all': every required concept present, using
    contains_concept's word-form-aware matching (explain questions)."""
    t = text.lower()
    if match_mode == "all":
        return all(contains_concept(t, a.lower()) for a in answers)
    return any(_word_boundary_match(t, a.lower()) for a in answers)
