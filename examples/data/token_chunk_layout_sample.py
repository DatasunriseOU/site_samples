"""Token chunk layout excerpt from the enriched pipeline.

This example shows how character-level chunk metadata becomes token-level chunk
layout. The problem it solves is alignment: structure chunks, call edges, and
type edges need token offsets that match the training sequence seen by the
model.
"""

from __future__ import annotations


TOKEN_CHUNK_STARTS_COLUMN = "token_chunk_starts"
TOKEN_CHUNK_ENDS_COLUMN = "token_chunk_ends"
TOKEN_CHUNK_KINDS_COLUMN = "token_chunk_kinds"
TOKEN_CHUNK_DEP_LEVELS_COLUMN = "token_chunk_dep_levels"
TOKEN_CALL_EDGES_COLUMN = "token_call_edges"
TOKEN_TYPE_EDGES_COLUMN = "token_type_edges"


def _kind_to_int(kind: object) -> int:
    if isinstance(kind, int):
        return kind
    lookup = {
        "root": 0,
        "namespace": 1,
        "class": 2,
        "struct": 3,
        "function": 4,
        "method": 5,
        "field": 6,
    }
    return lookup.get(str(kind), 0)


def _normalize_graph_edge_pairs(raw_edges: object) -> list[tuple[int, int]]:
    if not isinstance(raw_edges, list):
        return []
    pairs: list[tuple[int, int]] = []
    for edge in raw_edges:
        if isinstance(edge, dict):
            src = edge.get("src", edge.get("source"))
            dst = edge.get("dst", edge.get("target"))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        try:
            pairs.append((int(src), int(dst)))
        except (TypeError, ValueError):
            continue
    return pairs


def _remap_token_edges(
    edge_pairs: list[tuple[int, int]],
    index_map: dict[int, int],
) -> list[dict[str, int]]:
    remapped: list[dict[str, int]] = []
    for src, dst in edge_pairs:
        if src not in index_map or dst not in index_map:
            continue
        remapped.append({"src": index_map[src], "dst": index_map[dst]})
    return remapped


def build_token_chunk_layout(
    doc: dict,
    token_chunks: list[dict],
    token_count: int,
) -> dict[str, list[int] | list[dict[str, int]]]:
    """Convert token-offset chunks into the packed metadata layout."""
    if not token_chunks:
        return {
            TOKEN_CHUNK_STARTS_COLUMN: [],
            TOKEN_CHUNK_ENDS_COLUMN: [],
            TOKEN_CHUNK_KINDS_COLUMN: [],
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: [],
            TOKEN_CALL_EDGES_COLUMN: [],
            TOKEN_TYPE_EDGES_COLUMN: [],
        }

    chunk_entries = list(enumerate(token_chunks))
    chunk_entries.sort(key=lambda item: int(item[1].get("token_offset", 0)))
    index_map = {orig_idx: new_idx for new_idx, (orig_idx, _) in enumerate(chunk_entries)}

    starts = [int(chunk.get("token_offset", 0)) for _, chunk in chunk_entries]
    ends = [starts[idx + 1] if idx + 1 < len(starts) else int(token_count) for idx in range(len(starts))]
    kinds = [_kind_to_int(chunk.get("kind", 0)) for _, chunk in chunk_entries]
    dep_levels = [int(chunk.get("dep_level", 0)) for _, chunk in chunk_entries]

    call_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("call_edges", [])),
        index_map,
    )
    type_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("type_edges", [])),
        index_map,
    )

    return {
        TOKEN_CHUNK_STARTS_COLUMN: starts,
        TOKEN_CHUNK_ENDS_COLUMN: ends,
        TOKEN_CHUNK_KINDS_COLUMN: kinds,
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: dep_levels,
        TOKEN_CALL_EDGES_COLUMN: call_edges,
        TOKEN_TYPE_EDGES_COLUMN: type_edges,
    }
