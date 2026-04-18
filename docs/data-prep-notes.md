# Data Preparation Notes

This note summarizes a public-safe data preparation workflow for C and C++
training corpora.

Major stages:
- collect permissive-source code and documentation
- normalize encodings and line endings
- deduplicate near-identical files and generated artifacts
- extract structure-aware metadata from build and syntax signals
- apply masking and curriculum rules for training subsets

Why the pipeline exists:
- raw corpora over-reward repeated boilerplate
- code-only samples miss useful explanatory context
- documentation-only samples miss executable structure
- versioned dataset snapshots make regressions easier to explain

Public-safe outcome:
- reproducible dataset versions
- explicit schema changes
- small examples that can be cited without exposing internal storage layout
