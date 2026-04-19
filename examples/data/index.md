# Data And Masking Index

Use this folder when you want to understand how MegaCpp training data is built.

What these examples do:
- recover build context for C and C++ files
- normalize enriched source records
- map character-level structure into token-aligned metadata
- keep metadata valid through masking and infill transforms
- pack multiple documents into a single fixed-length training row

Where this plugs into the model:
- before tokenizer-aware materialization
- before packed-row dataloading
- before document masking and structure-aware training losses

If you only read three files, start with:
- `enriched_jsonl_record_to_parquet.py`
- `masking_pipeline_excerpt.py`
- `packed_rows_schema_sample.py`
