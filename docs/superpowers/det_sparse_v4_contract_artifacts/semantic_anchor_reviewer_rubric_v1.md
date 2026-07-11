# Semantic anchor reviewer rubric v1

Status: offline smoke rubric only.

Reviewers inspect the prompt-visible corpus before inference and redacted model
decisions after a complete sealed raw manifest exists. They do not inspect
development topics, qrels, retrieval results, reranker scores, or model logs
outside the sealed local run artifacts.

For the smoke fixture, reviewers check only that the synthetic case is not
copied from consumed topic text, the gold range is globally shortest among
acceptable referent spans, and runner-visible files do not expose gold labels.
