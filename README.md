# MegaCpp Public Sample Pack

Sanitized public excerpts, article sources, and reference snippets for MegaCpp. The repository id stays `site_samples`, but the contents are the public pack linked from `megacpp.com` articles.

This repository holds:

- `articles/`: public-safe blog markdown consumed by `megacpp.com`
- `docs/`: supporting public notes and article-safe background material
- `examples/`: compact public-safe runnable samples
- `excerpts/`: article-linked code and note snippets extracted from the MegaCpp research and production codebases

`megacpp.com` consumes `articles/*.md` via `npm run sync:blog`, which generates committed JSON artifacts under `megacpp.com/src/generated/blog`.

## Directory guide

- `articles/`
  Public article sources.
- `docs/`
  Public notes that can be linked directly from articles.
- `examples/`
  Minimal standalone samples for concepts such as STP, FIRE, data masking, XLA flags, and OOM triage.
- `excerpts/code/`
  Sanitized code excerpts grouped by source repo and topic.
- `excerpts/docs/`
  Sanitized doc excerpts grouped by source repo and topic.
- `excerpts/images/`
  Public diagrams, charts, and screenshots referenced by articles.

## Safety rules

- no local machine paths
- no internal hostnames or IP addresses
- no secrets or private infra identifiers
- no unpublished private environment labels
- keep public wording generic and citation-friendly

## Excerpt naming

- Article-linked excerpt paths live under `excerpts/<kind>/<source-repo>/<topic>/`.
- File names use `article-slug__topic__v1.ext`.
- Doc excerpts should say they were edited for public clarity.
- Code excerpts should keep only the minimum context needed for the article claim.

## Current packs

- `articles/*.md`
- `docs/*.md`
- `examples/**/*.py`
- `excerpts/code/**`
- `excerpts/docs/**`
