# Long Context Index

This folder focuses on sequence transforms that still need correct metadata at
64K, 128K, and beyond.

What these examples do:
- remap token-level metadata after FIM permutations
- remap chunk boundary offsets after the token order changes
- avoid half-valid graph metadata by dropping chunks that cross a split
- preserve document IDs and segment IDs for document masking and state-reset features

Where this plugs in:
- after masking or infill augmentation
- before structure-aware losses, packed-row masking, and long-context training
