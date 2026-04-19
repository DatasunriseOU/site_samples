# Data And Masking Examples

This directory contains public-safe examples from the MegaCpp POC data pipeline.

These examples show how raw C and C++ sources become enriched training rows:
- compile command recovery
- enriched JSONL normalization
- token-level structure and chunk metadata
- structure embedding inputs and chunk-relation metadata
- document-level platform labels for broadcast embeddings
- masking and infill transforms
- packed-row contracts for training

These examples plug into the model before training starts. They prepare the
metadata that later drives document masking, structure-aware losses, edit-style
augmentation, and long-context packing.

Key files in this directory:
- `compile_commands_context_example.py`: extracts compiler context from `compile_commands.json`
- `enriched_jsonl_record_to_parquet.py`: normalizes enriched records for parquet output
- `token_chunk_layout_sample.py`: turns chunk boundaries and graph edges into token-aligned layout
- `structure_embedding_components_sample.py`: shows which enriched columns feed the structure embedding path
- `structure_graph_relations_sample.py`: normalizes chunk relations for TreeFFN and relation-bias style features
- `platform_embedding_sample.py`: summarizes per-document platform label inputs
- `loader_enriched_columns_sample.py`: shows the loader fallback path for optional enriched JSON columns
- `masking_pipeline_excerpt.py`: preserves metadata through FIM-style masking
- `packed_rows_schema_sample.py`: captures the packed-row column contract
- `packed_row_builder_example.py`: demonstrates row packing at fixed sequence length
- `ifim_sample.py` and `sri_sample.py`: edit-style augmentation surfaces

The code is intentionally trimmed to stay readable, but the control flow and
data contracts come from real MegaCpp POC sources.
