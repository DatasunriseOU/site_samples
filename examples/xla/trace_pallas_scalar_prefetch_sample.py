"""trace-pallas scalar-prefetch sample.

What it is: a MegaCpp POC-based excerpt of the autograd wrapper around a TPU Pallas
attention kernel with softcapping and shrunk sparse grids.
Why it exists: the bridge needed one place that traces the Pallas body, records
the custom-call payload, and feeds the compact sparse metadata in the exact
argument order the TPU runtime expects.
What problem it solves: it keeps grid-width, `data_next`, and block-mask wiring
stable across forward and backward instead of letting each call site rebuild the
contract ad hoc.
"""

from __future__ import annotations

import torch


class SoftcappedFlashAttention(torch.autograd.Function):
    """Flash Attention with softcapping and grid shrinking via trace-pallas."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softcap,
        sm_scale,
        q_segment_ids,
        kv_segment_ids,
        data_next_torch,
        block_mask_torch,
        partial_mask_torch,
        data_next_t_torch,
        block_mask_t_torch,
        grid_width,
        grid_width_t,
        mask_obj,
    ):
        trace_pallas = _require_trace_pallas()
        torch_xla = _require_torch_xla()

        segment_ids = None
        q_segment_ids_fa = None
        kv_segment_ids_fa = None
        if q_segment_ids is not None:
            to_jax_shape_dtype_struct = _require_to_jax_shape_dtype_struct()
            segment_ids_type = _require_segment_ids_type()
            assert kv_segment_ids is not None
            segment_ids = segment_ids_type(
                to_jax_shape_dtype_struct(q_segment_ids),
                to_jax_shape_dtype_struct(kv_segment_ids),
            )
            q_segment_ids_fa = q_segment_ids.unsqueeze(-1)
            kv_segment_ids_fa = kv_segment_ids.unsqueeze(1)

        payload, _ = trace_pallas(
            _flash_attention_impl_softcap,
            q,
            k,
            v,
            None,
            segment_ids,
            data_next_torch,
            block_mask_torch,
            partial_mask_torch,
            grid_width,
            True,
            sm_scale,
            softcap,
            mask_obj.mask_fn if mask_obj is not None else None,
            static_argnums=range(8, 18),
            use_cache=True,
        )

        shapes = [list(q.shape)]
        dtypes = [q.dtype]
        args = [data_next_torch, block_mask_torch, q, k, v]
        if q_segment_ids_fa is not None:
            args += [q_segment_ids_fa, kv_segment_ids_fa]
        args.append(partial_mask_torch)

        with torch.no_grad():
            result = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

        ctx.save_for_backward(
            q,
            k,
            v,
            q_segment_ids,
            kv_segment_ids,
            data_next_torch,
            block_mask_torch,
            partial_mask_torch,
            data_next_t_torch,
            block_mask_t_torch,
        )
        ctx.softcap = softcap
        ctx.sm_scale = sm_scale
        ctx.mask_obj = mask_obj
        ctx.grid_width = grid_width
        ctx.grid_width_t = grid_width_t
        return result[0]


def _require_trace_pallas():
    raise NotImplementedError("Publication sample keeps only the call contract.")


def _require_torch_xla():
    raise NotImplementedError("Publication sample keeps only the call contract.")


def _require_to_jax_shape_dtype_struct():
    raise NotImplementedError("Publication sample keeps only the call contract.")


def _require_segment_ids_type():
    raise NotImplementedError("Publication sample keeps only the call contract.")


def _flash_attention_impl_softcap(*args, **kwargs):
    raise NotImplementedError("Publication sample keeps only the call contract.")
