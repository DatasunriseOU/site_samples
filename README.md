# site_samples

Sanitized public excerpts for public article references.

This repository holds small public-safe markdown notes and code fragments that
back public article references.

Scope:
- training and systems notes
- data preparation notes
- TPU/XLA and distributed debugging notes
- minimal code samples with private details removed

Safety rules:
- no local machine paths
- no internal hostnames or IP addresses
- no secrets or private infra identifiers
- no internal-only project or product naming

Review note:
- new docs/examples should stay generic and citation-friendly
- avoid company, cluster, region, and environment-specific labels

Current packs:
- `docs/plasticity-toolkit-notes.md`
- `docs/stp-notes.md`
- `docs/hybrid-layout-notes.md`
- `docs/tpu-bringup-notes.md`
- `docs/data-prep-notes.md`
- `docs/distributed-debugging-notes.md`
- `examples/fire/fire_sample.py`
- `examples/fire/redo_sample.py`
- `examples/stp/stp_sample.py`
- `examples/hybrid/hybrid_pattern_sample.py`
- `examples/xla/xla_flag_profile.py`
- `examples/data/masking_pipeline_sample.py`
- `examples/distributed/oom_triage_sample.py`
