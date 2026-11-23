#!/usr/bin/env python3
"""Compare two SGLANG_DEBUG_MID_DUMP files (NPU vs Kunpeng).

Usage:
    python compare_mid_dump.py /tmp/npu.mid.rank0.txt /tmp/cpu.mid.rank0.txt [threshold]

Prints the tensors with the largest relative difference and the first
divergent tensor in file order.
"""

import re
import sys

LINE_RE = re.compile(r"^(?P<call>\d+)\|(?P<name>.+?)\|(?P<rest>.*)$")
NUM_RE = re.compile(r"(sum|mean|std|absmax)=([-+0-9.eE]+)")
FIRST_RE = re.compile(r"first=(.*)$")


def parse(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            key = (m.group("call"), m.group("name"))
            rest = m.group("rest")
            stats = {k: float(v) for k, v in NUM_RE.findall(rest)}
            fm = FIRST_RE.search(rest)
            first = (
                [float(x) for x in fm.group(1).split(",") if x]
                if fm and fm.group(1)
                else []
            )
            shape = rest.split("|", 2)[0]
            out[key] = (shape, stats, first)
    return out


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def main(a_path, b_path, threshold=1e-2):
    a, b = parse(a_path), parse(b_path)
    keys = sorted(set(a) & set(b), key=lambda x: (x[0], x[1]))
    if not keys:
        print("no common keys; check SGLANG_DEBUG_MID_DUMP_TOKENS and inputs")
        return

    scored = []
    for key in keys:
        sa, st_a, f_a = a[key]
        sb, st_b, f_b = b[key]
        diffs = {k: rel(st_a.get(k, 0), st_b.get(k, 0)) for k in ("sum", "mean", "std", "absmax")}
        fd = (
            max((rel(x, y) for x, y in zip(f_a, f_b)), default=0.0)
            if f_a and f_b
            else 0.0
        )
        scored.append((max([fd, *diffs.values()]), key, sa != sb, diffs, fd))

    scored.sort(reverse=True)
    print(f"compared {len(keys)} tensors")
    for score, (call, name), shape_mismatch, d, fd in scored[:20]:
        print(
            f"call={call} {name} score={score:.3g} shape_mismatch={shape_mismatch} "
            f"first_diff={fd:.3g} sum={d['sum']:.3g} mean={d['mean']:.3g} "
            f"std={d['std']:.3g} absmax={d['absmax']:.3g}"
        )

    print("\nfirst divergent tensor in file order:")
    for key in keys:
        sa, st_a, f_a = a[key]
        sb, st_b, f_b = b[key]
        d = max(rel(st_a.get(k, 0), st_b.get(k, 0)) for k in ("sum", "mean", "std", "absmax"))
        fd = (
            max((rel(x, y) for x, y in zip(f_a, f_b)), default=0.0)
            if f_a and f_b
            else 0.0
        )
        if sa != sb or max(d, fd) > threshold:
            print(f"  call={key[0]} {key[1]} shape={sa} vs {sb} score={max(d, fd):.3g}")
            break


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2)
