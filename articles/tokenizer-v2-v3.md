---
title: "Tokenizer evolution for C++ code: from v2 proposal to v3 shipped"
description: "How the MegaCpp C++ tokenizer evolved from a 32K v1 through a 48K v2 proposal to the 65K v3 shipped artifact: what we proposed, what corpus frequency analysis told us, and what it did for downstream eval."
date: "2026-04-18"
tags: ["tokenizer", "bpe", "C++", "vocab", "data"]
---

# Tokenizer evolution for C++ code: from v2 proposal to v3 shipped

The tokenizer is the part of the model nobody looks at until it is wrong. For a C++ specialist family that trains at 4K, 16K, and 64K context, "wrong" is expensive — a bad tokenizer wastes context, degrades the attention map, and silently inflates loss on the exact patterns the model is supposed to be best at. This is the story of how the MegaCpp tokenizer went from v1 (32K, mostly OK) through a v2 proposal (48K, never fully shipped) to v3 (65K, what we run today), and what the corpus frequency analysis taught us in between.

## Where v1 hurt

The v1 tokenizer was a 32,768-token hybrid: 1,600 fixed (specials, keywords, operators, preprocessor, punctuation, common STL identifiers, numbers 0-999, diff markers, structural whitespace) and 31,168 learned BPE. It was good enough for 4K syntax pretraining but had real holes that we kept hitting:

1. No thinking tokens. Anything resembling a chain of thought had to be expressed in plain comments and parsed downstream.
2. No number-pattern tokens. Hex (`0xFF`), floats (`3.14`), scientific notation (`1e10`), and binary literals (`0b1010`) were all shattered into multi-token sequences.
3. Pure BPE on the learned half. Nothing exploited C++ morphology — common stems like `init`, `read`, `buffer`, `value`, `index`, `node` had to be re-discovered as merges.
4. Wasted reserved slots. ~254 fixed slots in v1 were unused.
5. No GPU/accelerator tokens. CUDA, ROCm, XLA — all shattered.
6. No SQL or query/DB tokens, despite SQL being routinely embedded in C++ string literals.
7. Missing C++23/26 surface area (`flat_map`, `mdspan`, `source_location`, ranges additions).

The fix was not "add more BPE merges." Pure BPE, no matter how many merges, will not give you a single-token `__device__` or `cudaMalloc` unless you seed it. So the proposal was hybrid: more fixed, smarter learned.

## v2 (48K) as a fallback, v3 (65K) as the primary target

The v2 proposal sized the vocabulary at 49,152 with 5,535 fixed and ~43,617 morpheme-aware BPE. The v3 proposal sized it at 65,536 with the same fixed structure and ~58,336 BPE. Vocabulary scaling research at the time (NeurIPS 2024) had been arguing that mainstream LLMs were under-vocabularied for code; for a 270M-877M parameter C++ model, 48K-64K read as the sane range. We treated 65K as the primary target and 48K as a fallback if 65K turned out to be too memory-heavy at the embedding layer. (It did not.)

The proposed fixed-vocab layout was deliberately re-zoned compared to v1:

- 0-63: special tokens (64 of them, including thinking tokens like `<THINK_START>` / `<THINK_END>`, FIM markers, code/diff/comment delimiters, and a tool-call surface).
- 64-319: C++ keywords including C++23/26 additions.
- 320-639: operators, including overload-relevant punctuation forms.
- 640-799: preprocessor and attributes.
- 800-1199: a new dedicated band for number patterns — hex prefixes, float suffixes, scientific notation, binary literals, integer suffixes (including C++23 `z`/`uz` for `size_t`), and the most common integer literals (small ints plus powers of two through 65536).
- 1200-1499: punctuation and indent levels.
- 1500-4499: STL/stdlib including C++20/23/26 surface.
- 4500+: domain-specific bands for ChaiScript scripting tokens, C++ morphemes, common identifier stems, GPU/accelerator tokens, SQL domain tokens, query/DB tokens, C++23/26 additions, and testing/build-framework tokens.
- 7200+: morpheme-aware learned BPE.

That is what the proposal looked like. What we shipped is not what the proposal looked like, because we ran corpus frequency analysis before locking the layout.

## Vocabulary frequency analysis: the part that changed our minds

Before committing fixed slots to GPU, SQL, query/DB, C++23/26, and testing tokens, we ran the proposed fixed vocabulary against the actual corpus. The first run was on a 150K-document sample (three shards of `cpp_compilable_16k`). The second run was on the full deduplicated v4 corpus: 1,056,993 documents, 642,028,236 identifiers, ~22.3 GB of source. The Rust analyzer (`tools/vocab_analyzer`, rayon parallel, 48 threads) finished the full corpus in 51.4 seconds at ~28K files/sec.

The headline numbers were:

1. 93% coverage of the 697 proposed fixed tokens — virtually all proposed domain tokens appear in real C++ code somewhere.
2. Morphemes massively outweigh domain tokens — 88.3M morpheme hits versus 6.94M domain-token hits, a 12.7x ratio. That single ratio justified seeding the BPE with morpheme stems instead of throwing more fixed slots at domain vocab.
3. Comments are 24.2% of the corpus by bytes, and 15.3% of all lines. Comment tokenization quality is not optional; it directly affects loss.
4. Unicode is essentially absent — only 0.020% non-ASCII bytes, only 3.4% of files have any non-English comment content (mostly accented author names and Unicode math symbols). No need for unicode normalization.
5. 4-space indent dominates at 31% of indented lines, but tabs are at 26% and 2-space is also at 31%. All three need first-class BPE support.
6. Multi-space runs are 7.5% of all space occurrences. Worth a few dedicated multi-space tokens (SP×2, SP×4, SP×8) in the BPE vocabulary.

Then the uncomfortable findings.

### The "generic word" problem

The proposed domain bands (5300-6999, ~1,700 fixed slots reserved for GPU/SQL/query/DB/C++23-26/testing) included a lot of tokens that *did* appear in the corpus but had no business taking fixed slots. Examples: `query`, `Status`, `map`, `enum`, `chunk`, `expected`, `stride`, `transfer`, `receiver`. These are common C++ identifiers that BPE will learn perfectly well as merges. Spending a fixed slot on them is wasteful.

Concretely: of 473 found tokens in the first analysis, 209 were generic. Only tokens with distinctive naming patterns — `__double_underscore__`, `SCREAMING_CASE_MACROS`, prefix-style API names like `cublasSgemm` or `ncclAllReduce` — genuinely benefit from a fixed slot. The 150K-doc analysis recommended cutting the domain-token budget from 1,700 to ~216, an 87% reduction. The full 1.06M-doc analysis was more conservative at ~415 (more docs surfaced more edge-case hits for borderline tokens like ODBC and ROCm) but the direction was the same: cut hard, free slots, push them into BPE merges.

### Categories we dropped entirely

The full-corpus analysis identified categories where the entire band added no value:

- MongoDB `$`-prefixed operators (39 tokens, 148 hits across 1.06M docs). Wrong domain. C++ corpora simply do not contain MongoDB query DSLs.
- Redis commands (26 tokens, 430 hits). Same story; only `SREM` had any real usage and even that was incidental.
- CMake (22 tokens, 1,685 hits). Mostly excluded at the file-type filter — `CMakeLists.txt` is filtered out before tokenization, so the hits are almost all stray references in comments.
- C++ ORM tokens (13 tokens, 401 hits across all ORM tokens). Negligible.
- Protobuf keywords (18 tokens, 1.39M hits — but every single token is a generic English word: `enum`, `stream`, `map`, `repeated`). BPE learns these for free.

The pruning, summarised:

| Proposed band | Proposed slots | Hits in 1.06M docs | Decision |
|---------------|----------------|--------------------|----------|
| MongoDB `$`-operators | 39 | 148 | drop band |
| Redis commands | 26 | 430 | drop band |
| CMake keywords (file-type filtered out) | 22 | 1,685 | drop band |
| C++ ORM tokens | 13 | 401 | drop band |
| Protobuf keywords (generic English) | 18 | 1.39M | drop band, BPE-only |
| gRPC compound names | kept compounds, dropped `Status`/`Channel`/`Server` | — | prune within band |
| C++23 ranges | kept `cartesian_product`/`zip_transform`, dropped `stride`/`chunk` | — | prune within band |
| C++23 types | kept `source_location`/`flat_map`/`mdspan`, dropped `expected` | 108K (generic) | prune within band |

Within categories we kept, we pruned to compound and special-pattern tokens only. gRPC kept `ClientContext`, `ServerBuilder`, `CompletionQueue` and dropped `Status`, `Channel`, `Server` (all generic). C++23 ranges kept `cartesian_product` and `zip_transform` and dropped `stride` and `chunk`. C++23 types kept `source_location`, `flat_map`, `mdspan` and dropped `expected` (108K generic hits).

### Final budget

After all the cuts, the domain-token budget went from 697 proposed → ~415 final. The 282 freed slots went directly to BPE merges, taking the learned-vocabulary count from 58,336 to ~58,618 in v3. Small in absolute terms, but every freed slot is a merge that can absorb something the model actually sees frequently — which is exactly what the morpheme analysis (88.3M hits) said to invest in.

## Namespaces matter

One specific finding deserves its own paragraph because it changed BPE seeding. Across 28.2M namespace-qualified references, `std::` alone accounts for 6,818,050 hits. That is dominant by an order of magnitude over the next entries (`llvm::` 552K, `boost::` 517K, `detail::` 495K, `cutlass::` 443K, `absl::` 322K). We made sure `std::` was learned as an early BPE merge. After v3 shipped, attention maps on STL-heavy code became visibly cleaner — `std::vector<std::string>` is three tokens instead of five or six, and the model spends its attention budget on the type parameters instead of the namespace prefix.

## What the tokenizer did for downstream eval

The reason any of this matters is downstream loss and downstream behavior. We saw three concrete effects after v3 replaced v1:

1. Lower bits-per-byte at every context length we measured. The biggest gains were on STL-heavy code (where the namespace and stdlib improvements compound) and on number-heavy code (where the new number-pattern band turns multi-token literals into single tokens). On comment-heavy files the win was smaller but consistent — comments are a quarter of the corpus by bytes, so even a small per-byte improvement is a real loss reduction.
2. Better effective context. With v1, packing eight C++ documents into a 64K window was already cutting it close because of expansion on namespace prefixes, hex literals, and CUDA identifiers. With v3, the same eight documents fit with measurable headroom. We have not formally re-measured average expansion ratio per repo, but the empirical effect on dataloader stats was visible: the BOS-aligned best-fit packer ended up with fewer cropped documents per shard.
3. Cleaner attention on structural patterns. Because `template<`, `constexpr`, `#include`, `::`, `->`, `<=>`, and `std::` are guaranteed single tokens, they show up as single columns in attention visualizations. That made it noticeably easier to debug the long-context behavior of the document-masking implementation, because we could *see* document boundaries and structural anchors instead of guessing.

There were also things v3 did *not* fix that we want to call out. The BPE half still occasionally fragments long compound identifiers from non-mainstream C++ libraries (Symbian-era headers, certain RTOS APIs) because the corpus simply did not have enough hits to learn them as merges. We are tracking this and may add a small targeted-vocab pass for embedded RTOS workloads when we ship a dedicated embedded specialist.

## The fixed-token JSON, briefly

The shipped artifact is consumed by the trainer like this:

```python
# trainer side: load the v3 fixed-vocab spec and emit the runtime tokenizer artifact
import json
from tokenizer_builder import build_tokenizer_from_spec

with open("config/fixed_token_spec.json") as f:
    spec = json.load(f)              # bands, IDs, reserved tail

tok = build_tokenizer_from_spec(
    spec,
    target_vocab=65_536,             # v3 primary, not v2 fallback
    bpe_merges=58_618,               # 282 slots reclaimed from domain bands
    seed_morphemes=True,             # 88.3M morpheme hits in the corpus
    seed_namespaces=("std::",),      # learned as an early merge
)
tok.save("artifacts/runtime_tokenizer_artifact.json")
```

The shipped fixed-token spec lives in a dedicated tokenizer configuration artifact. We will not inline its contents here — it is large and structural — but its top-level shape mirrors the bands described above: special tokens, keywords, operators, preprocessor, punctuation, STL/stdlib, number patterns, indent levels, the surviving domain bands (GPU compound names, SQL APIs that BPE actually splits poorly, GTest macros, compiler attributes), and a reserved tail. Every entry has a stable ID, every band is sized with empirical headroom, and the JSON is the single source of truth that the trainer consumes when emitting the runtime tokenizer artifact.

## Lessons we want to keep

Three lessons from this iteration are worth keeping for any future tokenizer work:

First, run the frequency analysis before locking the layout. Our v2 proposal would have shipped with ~1,200 wasted fixed slots if we had not run it. The Rust analyzer that does the work is small (a few hundred lines) and finishes the full corpus in under a minute. There is no excuse to skip this step.

Second, distrust generic English words in domain bands. If a proposed "GraphQL" or "Redis" or "C++23" token also happens to be a normal C++ identifier (`query`, `chunk`, `expected`, `stream`), assume BPE learns it and reclaim the slot.

Third, comments are not noise. They are a quarter of the corpus and they teach the model how engineers explain code to each other. The tokenizer must handle them well; the BPE must learn common comment patterns; the data pipeline must not strip them. v3 explicitly invested in this, and the eval numbers reward it.

v3 is the current public baseline described here. A later iteration will likely be driven by more languages embedded in C++ strings (HLSL, MSL, Metal, SPIR-V shader text, CUDA PTX, and larger SQL dialects), a dedicated embedded RTOS vocabulary, and another sweep of corpus frequency analysis on a broader long-context mix.

## References

- [MegaCpp source repository](https://github.com/DatasunriseOU/cppmega)
- [MegaCpp sample pack](https://github.com/DatasunriseOU/site_samples)
- Public data-preparation, tokenizer, and curriculum material linked from the MegaCpp repositories.
