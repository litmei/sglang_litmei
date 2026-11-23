"""Opt-in intermediate tensor dump for cross-platform parity debugging.

Usage:
    SGLANG_DEBUG_MID_DUMP=/tmp/qwen3.mid \
    SGLANG_DEBUG_MID_DUMP_TOKENS=<prefill_token_count> \
    python -m sglang.launch_server ...

Each rank writes ``<prefix>.rank<rank>.txt``.  Only the first forward whose
input token count matches ``SGLANG_DEBUG_MID_DUMP_TOKENS`` (or the very first
forward when it is unset) is recorded.  One line per tensor::

    call|name|shape|dtype|sum=..|mean=..|std=..|absmax=..|first=<N values>

Set ``SGLANG_DEBUG_MID_DUMP_CALLS`` to record more than one matching forward.
The dump is a no-op when ``SGLANG_DEBUG_MID_DUMP`` is unset.
"""

import os

import torch

_PREFIX = os.environ.get("SGLANG_DEBUG_MID_DUMP", "")
_ENABLED = bool(_PREFIX)
_MAX_CALLS = int(os.environ.get("SGLANG_DEBUG_MID_DUMP_CALLS", "1"))
_TOKENS_RAW = os.environ.get("SGLANG_DEBUG_MID_DUMP_TOKENS")
_TOKENS = int(_TOKENS_RAW) if _TOKENS_RAW else None
_FIRST = int(os.environ.get("SGLANG_DEBUG_MID_DUMP_FIRST", "8"))

_active = False
_calls = 0
_path = None
_skip_logged = False


def _path_for_rank():
    global _path
    if _path is None:
        rank = 0
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            pass
        _path = f"{_PREFIX}.rank{rank}.txt"
        parent = os.path.dirname(_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(_path, "w") as f:
            f.write("# mid-dump\n")
        print(f"[mid-dump] writing {_path}", flush=True)
    return _path


def _format(obj):
    if not isinstance(obj, torch.Tensor):
        return f"{type(obj).__name__}|{obj}"[:400]
    flat = obj.detach().reshape(-1)
    if flat.numel() == 0:
        return f"{tuple(obj.shape)}|{obj.dtype}|empty"
    f = flat.float()
    head = ",".join(f"{v:.6g}" for v in f[:_FIRST].tolist())
    std = f.std().item() if f.numel() > 1 else 0.0
    return (
        f"{tuple(obj.shape)}|{obj.dtype}|sum={f.sum().item():.6g}|"
        f"mean={f.mean().item():.6g}|std={std:.6g}|"
        f"absmax={f.abs().max().item():.6g}|first={head}"
    )


def _dump(name, obj):
    if not _active:
        return
    if isinstance(obj, (tuple, list)):
        for i, item in enumerate(obj):
            _dump(f"{name}[{i}]", item)
        return
    with open(_path_for_rank(), "a") as fh:
        fh.write(f"{_calls}|{name}|{_format(obj)}\n")


def _model_pre_hook(module, args, kwargs):
    global _active, _calls, _skip_logged
    if not _ENABLED or _calls >= _MAX_CALLS:
        _active = False
        return
    input_ids = args[0] if args else kwargs.get("input_ids")
    n = int(input_ids.numel()) if isinstance(input_ids, torch.Tensor) else -1
    if _TOKENS is not None and n != _TOKENS:
        if not _skip_logged:
            _skip_logged = True
            print(
                f"[mid-dump] first forward tokens={n}, configured "
                f"SGLANG_DEBUG_MID_DUMP_TOKENS={_TOKENS}; not dumping",
                flush=True,
            )
        _active = False
        return
    _active = True
    _calls += 1
    print(f"[mid-dump] recording call={_calls} tokens={n}", flush=True)
    _dump("input_ids", input_ids)


def _model_post_hook(module, args, output):
    global _active
    if _active:
        logits = getattr(output, "next_token_logits", output)
        _dump("logits", logits)
    _active = False


def _hook(name):
    def fn(module, args, output):
        _dump(name, output)

    return fn


def maybe_install_debug_hooks(model):
    """Register dump hooks on a Qwen3-style model. No-op unless enabled."""
    if not _ENABLED:
        return

    print(
        f"[mid-dump] hooks installed prefix={_PREFIX} tokens={_TOKENS} "
        f"calls={_MAX_CALLS}",
        flush=True,
    )
    model.register_forward_pre_hook(_model_pre_hook, with_kwargs=True)
    model.register_forward_hook(_model_post_hook)

    inner = getattr(model, "model", None)
    if inner is None:
        return

    embed = getattr(inner, "embed_tokens", None)
    if isinstance(embed, torch.nn.Module):
        embed.register_forward_pre_hook(
            lambda m, args: _dump("embed.input_ids", args[0] if args else None)
        )
        embed.register_forward_hook(_hook("embed.out"))

    for i, layer in enumerate(getattr(inner, "layers", []) or []):
        if not isinstance(layer, torch.nn.Module):
            continue
        layer.register_forward_hook(_hook(f"L{i}.out"))

        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, torch.nn.Module):
            for attr in ("qkv_proj", "q_norm", "k_norm", "rotary_emb", "attn", "o_proj"):
                mod = getattr(attn, attr, None)
                if isinstance(mod, torch.nn.Module):
                    mod.register_forward_hook(_hook(f"L{i}.{attr}"))

        for attr in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, attr, None)
            if isinstance(mod, torch.nn.Module):
                mod.register_forward_hook(_hook(f"L{i}.{attr}"))

        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, torch.nn.Module):
            for attr in ("gate_up_proj", "down_proj"):
                mod = getattr(mlp, attr, None)
                if isinstance(mod, torch.nn.Module):
                    mod.register_forward_hook(_hook(f"L{i}.{attr}"))

    norm = getattr(inner, "norm", None)
    if isinstance(norm, torch.nn.Module):
        norm.register_forward_hook(_hook("final_norm"))
