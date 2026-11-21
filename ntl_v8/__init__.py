"""Standalone V8-arm probe for Name That LLM (DAiL bnty_0716).

Arms measured here:
  V7-control : TF-IDF(word) + TF-IDF(char) + standardized style block, logistic regression.
  V8-TRACE   : V7-control + TRACE rank-transition features (within-prompt-group ranks
               of the style views and their step-to-step transitions).
"""

__all__ = ["data", "features", "model", "oof"]
