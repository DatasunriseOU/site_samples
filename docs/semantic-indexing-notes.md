# Semantic Indexing Notes

This note summarizes the public version of MegaCpp's build-aware semantic
indexing story.

## What the build database gives you

When a repository provides a valid `compile_commands.json`, Clang tooling can
see the same include paths, defines, and language modes that the real build
uses. That makes cross-file symbol resolution much more trustworthy than a
syntax-only pass.

## What it does not guarantee

- generated headers may still be missing
- the build database may cover only part of the repository
- stale command lines can produce semantically coherent but wrong results
- macro drift can change which declarations even exist

## MegaCpp public takeaway

- keep a broad syntax-first lane for coverage
- keep a compile-aware lane for high-trust cross-file structure
- record which outputs came from which lane
- never present partial semantic coverage as full semantic truth

## Suggested metrics

- percentage of files matched to compile commands
- percentage of matched files that indexed successfully
- percentage of repositories with partial coverage only
- percentage of outputs that fell back to syntax-only structure
