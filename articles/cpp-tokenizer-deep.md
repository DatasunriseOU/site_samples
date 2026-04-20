---
title: "Inside the MegaCpp C++ tokenizer: fixed vocab, BPE, and per-specialist sub-vocabs"
description: "A deep look at the tokenizer we ship: half hand-curated vocabulary, half learned BPE, what changed between v2 and v3, where the collisions live, and how per-specialist sub-vocabs fall out of the shared 64K layout."
date: "2026-04-18"
tags: ["tokenizer", "bpe", "C++", "vocab"]
---

The tokenizer story is usually told at the summary level: we grew from 32K to 48K and then to a 64K-class v3 layout, seeded some morphemes, and locked the fixed-token bands. That summary is still almost useless for engineering. What actually mattered was the frequency analysis on the real corpus, the collisions between fixed and learned slots, and the per-specialist sub-vocab story — which is not a separate artifact but a discipline of BPE seeding and runtime ID masking.

## Why MegaCpp cares about this

A bad tokenizer is the cheapest way to degrade a code model. It wastes context, shatters high-frequency patterns into multi-token sequences, confuses the attention map, and silently inflates loss on the identifiers the model should be best at. For specialists training at 4K, 16K, and 64K, every percentage point of expansion ratio costs real compute at 64K and real answer quality at 4K. The other reason: our model family is a set of specialists sharing one 64K vocabulary. Their practical working sets differ enough that "per-specialist sub-vocab" is a useful abstraction even when no separate artifact exists on disk.

## What we built in MegaCpp

The tokenizer is a hybrid. Half of the vocabulary is hand-curated fixed tokens; half is learned BPE. The tokenizer implementation wraps the HuggingFace `tokenizers` backend with BERT-style whitespace handling and a custom decoder that knows the difference between a standalone added token and a BPE suffix fragment.

### The fixed half

The fixed half covers things BPE cannot be trusted to learn well:

1. Special tokens (IDs 0-63 in v3). `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, FIM markers, `<CODE_START>`/`<CODE_END>`, thinking tokens (`<THINK_START>`/`<THINK_END>` and sub-variants for error, fix, trace, verify, plan), tool-call tokens (`<QUERY_TOOL>`, `<TOOL_RESULT>`), compile/script markers, diff and comment delimiters, file separators, and a reserved tail. Control surface, not learnable.
2. C++ keywords, operators, preprocessor directives, attributes. Extended through C++23/26 (`constinit`, `co_await`, `co_yield`, `co_return`, `contract_assert`, `_Atomic`, and the attribute set).
3. Number patterns. A dedicated band for hex prefixes (`0x`, `0X`), common byte values (`0x00`, `0xFF`, `0x80`), 32-bit magic constants (`0xDEADBEEF`, `0xCAFEBABE`), float suffixes and common float literals, scientific notation, binary literal prefixes, and C++23 integer suffixes including `z` and `uz` for `size_t`. Before this band existed, `0xDEADBEEF` was five or six BPE tokens; now it is one.
4. Punctuation and indent tokens. Explicit tokens for `"  "`, `"    "`, `"        "`, and `"\t"`.
5. STL and stdlib identifiers at high frequency.
6. Domain bands: GPU/accelerator tokens (CUDA runtime, cuBLAS, cuDNN, Thrust/CUB, CUTLASS, NCCL, atomics, graph API), ROCm/HIP mirrors, TPU/XLA op names (MHLO dialect, Pallas/Mosaic surface), SQL keywords, query/DB tokens, C++23/26 library surface, and testing/build-framework tokens (GTest, Catch2, Boost.Test).

Each added token is registered through HuggingFace's added-tokens mechanism, which is what the production tokenizer path uses to distinguish "this token is a full word" from "this token is a BPE fragment." That distinction drives the decoder's space-reconstruction heuristics, because a C++ identifier like `end_point` may arrive as `end` + `_` + `po` + `int` through the pre-tokenizer and has to decode without a stray space between `po` and `int`.

### The learned half

The learned half is BPE, but it is BPE seeded aggressively on corpus-measured morphemes. The frequency analysis ran a Rust analyzer (`tools/vocab_analyzer`, rayon parallel, 48 threads) across 1,435,084 C++ source files — 22.3 GB, 333 open-source projects — in 51.4 seconds. The headline number that shaped BPE seeding was morpheme dominance: 88.3M morpheme hits across 128 proposed morphemes, against 6.94M total hits for the 697 proposed fixed domain tokens. A 12.7x ratio, and it pointed directly at what to invest in.

Morpheme classes seeded into BPE: common components (value/index/offset/node/ptr/buffer/count/context, 59.9M hits over 52 items), C++ stems (init/read/write/create/start/format/lock/parse/find/alloc/insert, 22.9M over 30), prefixes (proto, sub, non, multi, meta, mono; 4.2M over 24), and suffixes (1.2M over 22). Total ~99% coverage of the proposed morpheme set. Aggressive early merges (`init`, `read`, `write`, `buffer`, `value`, `index`, `node`, `ptr`) mean `initialize` tokenizes as `init` + `ialize`, not fragment soup. After seeding, BPE ran a standard merge schedule until the remaining budget filled.

One specific seeding decision paid off disproportionately: `std::` is 6.8M namespace-qualified references out of 28.2M total namespace references — an order of magnitude ahead of the next entries (`llvm::` 552K, `boost::` 517K, `detail::` 495K, `cutlass::` 443K, `absl::` 322K). Making sure `std::` merged early meant `std::vector<std::string>` becomes three tokens instead of five or six, and attention visualizations over STL-heavy code got visibly cleaner after v3 shipped.

### v2 to v3: what actually changed in token-frequency terms

The internal v2 and v3 sizing proposals looked clean on paper — 1,700 fixed slots for GPU/SQL/query/DB/C++23-26/testing domains, organized into neat bands from 5300 to 6999. We ran the frequency analysis before committing those slots, which is where things got uncomfortable.

The first uncomfortable finding was the **generic word problem**. Of 473 domain tokens found in the smaller 150K-doc analysis, 209 were generic C++ identifiers appearing in 2% to 30% of documents. Words like `query`, `Status`, `map`, `enum`, `chunk`, `expected`, `stride`, `transfer`, `receiver` — technically valid "domain" tokens under the original taxonomy, but wastes of fixed slots because BPE learns them perfectly well as merges. We cut them. The initial budget went from 1,700 down to ~216 on the 150K analysis, an 87% reduction.

The second uncomfortable finding, from the full 1.06M-document corpus run, confirmed the direction but softened the magnitude. More documents surfaced more edge-case hits for borderline tokens (ODBC, ROCm, some C++23 types), so the final domain budget settled at ~415 tokens. The categories we cut entirely on the full-corpus pass:

- MongoDB `$`-prefixed operators (39 tokens, 148 total hits across 1.06M docs). C++ corpora simply do not contain MongoDB query DSLs.
- Redis commands (26 tokens, 430 hits). Only `SREM` had any real usage, and even that was incidental.
- CMake (22 tokens, 1,685 hits). Almost entirely stray references in comments, because `CMakeLists.txt` is filtered at the file-type stage.
- C++ ORM tokens (13 tokens, 401 hits). Negligible across the ORM band.
- Protobuf keywords as a category (18 tokens, 1.39M hits — but every single token was a generic English word that BPE learns for free).

Within surviving categories we pruned to compound and special-pattern tokens only. gRPC kept `ClientContext`, `ServerBuilder`, `CompletionQueue`, dropped `Status`, `Channel`, `Server`. C++23 types kept `source_location`, `flat_map`, `mdspan`, dropped `expected` (108K generic hits). C++23 ranges kept `cartesian_product` and `zip_transform`, dropped `stride` and `chunk`. The surviving rule of thumb: fixed slots only for tokens with distinctive naming patterns — double-underscore, SCREAMING_CASE, or prefix-style API names like `cublasSgemm`, `ncclAllReduce`, `__device__`.

The 282 slots freed by those cuts went directly to BPE merges. In v3 the learned vocabulary rose from 58,336 to ~58,618, small in absolute terms but every freed slot is a merge that absorbs something the model actually sees.

### Vocabulary collisions

Collisions between the hand-curated half and the learned half show up in two places. First, suffix-vs-standalone ambiguity: a token like `s`, `is`, `or`, `if`, `in` could be a BPE suffix (`char` + `s` = `chars`) or a standalone identifier. The decoder's `_is_bpe_suffix` logic is context-dependent: if the previous token is a fixed added token like `char`, a single-char `s` attaches as a suffix; if the previous token is a BPE fragment, a common short word stays standalone. We maintain an allow-list of common short words so the decoder does not collapse them into the previous identifier. Second, the underscore-identifier case: the pre-tokenizer splits on `_`, so `end_point` arrives as `end` + `_` + `po` + `int`, and the decoder tracks an `in_underscore_id` state so subsequent fragments continue joining. Neither case is research, both are load-bearing — a tokenizer whose decode round-trip is off by a space every few hundred tokens silently corrupts every SFT example and verifier check.

### Per-specialist sub-vocabs

We ship one shared 64K artifact, not per-specialist tokenizers. The strongest checked-in evidence is the fixed-token manifest for the v3 C++ tokenizer, which declares `_total_vocab: 65536` and describes IDs `7200-65535` as the learned BPE band. Each specialist has a characteristic distribution over that shared vocabulary, which is useful to think about as a sub-vocab. A systems-C specialist spikes on `__attribute__`, `__builtin_*`, `likely`/`unlikely`, byte-value hex literals, and the preprocessor band, with almost no hits on CUTLASS/CUDA. A template-heavy generic C++ specialist saturates STL and Boost, touches the number-pattern band lightly, and uses attributes more than CUDA. A GPU specialist spikes on `__global__`, `__device__`, `cudaMalloc`, `threadIdx`, `cublasSgemm`, and atomics; a TPU/Pallas specialist lights up `mhlo.*`, `pallas.program_id`, `BlockSpec`, and `GridSpec`.

Platform vocabulary is separate. A dedicated platform-metadata layer defines a 113-entry label-to-ID space consumed by an `nn.EmbeddingBag(mode='sum')` path, not by the text tokenizer. Six categories (OS, RTOS, GPU, architecture, compiler, C++ standard), up to 20 IDs per document. A prefix emitter can render the same info as a `// platform: ...` comment that does go through the tokenizer, but the ID-embedding path is primary.

The practical effect is that "per-specialist sub-vocab" is emergent from training mix and runtime ID masking, not from artifact duplication. We considered cold-freezing unused domain bands at inference (zeroing softmax over the CUDA band for a systems-C specialist, for example) but have not shipped it. The BPE band is shared, added-token IDs are stable across specialists, and the merge schedule is fixed, which keeps weight sharing and ensemble-time routing simpler than per-specialist tokenizers.

## How it lands in MegaCpp

In production the tokenizer is a build-time artifact, not a runtime library. The runtime tokenizer JSON (~2.2 MiB) is produced by the upstream build and copied into the tokenizer directory the training launchers point at. The data-prep pipeline consumes it by path, so bumping the tokenizer means bumping the upstream tokenizer revision and re-running the downstream prep stages. The final validation stage asserts `max(token_id) < vocab_size`, so mismatched pairs fail fast. The formatted decode path optionally shells out to `clang-format`, with a raw-decode fallback; training does not use this path.

We lift the tokenizer into MegaCpp as-is. The only difference: the production launcher uses a HuggingFace tokenizer type with an explicit vocab-size flag and model directory, and never re-trains at launch.

## Ablations and what we kept

The ablations worth keeping on file:

- v1 at 32K was clearly under-vocabularied for code. BPE-only, no morpheme seeding, no number-pattern band, no thinking tokens. The fix was not "more BPE merges"; pure BPE on a bigger budget does not recover single-token `__device__` or `cudaMalloc`. We committed to hybrid early.
- The v2 proposal at 48K was a useful fallback but never shipped; the measured embedding cost of 65K was fine at our model sizes, so we went straight to v3.
- The proposed 1,700 domain slots were almost entirely wrong. We kept 415 and freed the rest to BPE. The freed slots are not glamorous but they are real merges on high-frequency morphemes.
- We considered a Unicode normalization pass (accents and math symbols). The corpus is 99.98% ASCII; the pass was not worth the cost of a non-invertible transform in an otherwise round-tripping pipeline.
- We considered per-specialist vocabularies. We rejected them: shared weights and stable IDs matter more for ensemble routing than a small per-specialist efficiency win. Runtime ID masking is the lighter-weight alternative when we need it.

## Production checklist

- The tokenizer artifact is tied to a specific upstream revision; that revision is recorded with every checkpoint the dataset feeds.
- Added-token IDs are stable across versions for the lifetime of a specialist family; thinking, tool-call, and compile tokens have pinned IDs so SFT-formatted data survives tokenizer rebuilds.
- Vocabulary size is asserted at every stage: data prep (`verify_dataset_megacpp.py`), training launcher flags, inference loader. Mismatches fail fast, not silently.
- Decode round-trip is pinned by unit tests for the suffix-vs-standalone and underscore-identifier cases. Changes to `_is_bpe_suffix` require regression coverage.
- Per-specialist runtime ID masking is a feature flag, off by default; it is a tool for ensemble routing, not a requirement.
- Platform-info IDs are a separate artifact from the token vocabulary and are consumed through an embedding-bag path, not through the text tokenizer.

## Vocab snapshot

| Layer | Slot count (approx) | Source | Notes |
|-------|---------------------|--------|-------|
| Fixed, hand-curated | thousands | keywords, punctuation, operators, morphemes | stable across versions |
| Learned BPE | tens of thousands | corpus frequencies | rebuilt v2 -> v3 |
| Reserved and special | small | `<doc>`, `<mask>`, tool tokens | never reassigned |
| Per-specialist working set | subset of total | BPE seeding + runtime ID masking | no separate artifact on disk |

```python
# runtime ID masking for a specialist: disallow IDs outside the working set
import numpy as np
mask = np.full(vocab_size, -np.inf, dtype=np.float32)
mask[specialist_ids] = 0.0
logits = logits + mask  # applied before softmax
```

## References

- [Reference corpus and tokenizer pinning notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md)
- [Semantic indexing and compile-command notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/semantic-indexing-notes.md)
- [Hybrid layout notes for platform and block metadata](https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md)
- [Compile-commands sample artifact](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands_fixture.json)
- [MegaCpp source repository](https://github.com/DatasunriseOU/cppmega)
- [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies](https://openreview.net/forum?id=j4e4SkA5Xq)
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
