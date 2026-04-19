# Long Context Examples

This directory contains public-safe long-context helpers from the MegaCpp POC.

The goal of these examples is to show how long sequences stay usable after
masking or structural transforms. At long context, metadata mistakes get more
expensive: chunk offsets, structure labels, and valid-token boundaries must
still line up with the actual training tokens.

Key files in this directory:
- `fim_long_context_metadata_sample.py`: remaps token metadata through FIM
- `chunk_boundary_remap_sample.py`: remaps chunk offsets and drops boundary-crossing chunks
- `doc_mask_segment_ids_sample.py`: keeps document IDs and segment IDs aligned in packed long sequences

These helpers sit between augmentation and training. They preserve the contract
that attention masks, structure features, and chunk graphs all describe the same
sequence seen by the model.
