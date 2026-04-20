#!/usr/bin/env python3
"""
Research-only helper for the deeper GB10 driver patch lane.

This script works on a copied `libcuda.so` and is included for reproducibility
of the user-space driver experiments discussed in the GB10 articles. It is not
public proof of shipping `tcgen05` support on GB10 and should be used only in
an isolated environment via `LD_LIBRARY_PATH`.
"""

import argparse, os, re, shutil, struct, subprocess, sys
from pathlib import Path

# Compute capabilities that appear in the driver. Besides the published
# CCs, the table also contains a handful of "odd" rows (placeholders for
# unused arches), so the filter must be permissive: any maj in [3..13]
# and any min in [0..9] — 130 allowed pairs total.
def is_plausible_cc(maj, mn):
    return 3 <= maj <= 13 and 0 <= mn <= 9

# The table is recognized by the simultaneous presence of several target
# records (anchors). Anchors are the known CCs of real production arches:
REQUIRED_ANCHORS = [
    (9, 0),    # sm_90  (Hopper)
    (10, 0),   # sm_100 (Blackwell DC)
    (12, 0),   # sm_120 (GeForce Blackwell)
    (12, 1),   # sm_121 (Spark Blackwell)
]

# Patch targets. Each is a set of CC pairs to rewrite to (10, 0) with
# flags=(10, 0).
TARGETS = {
    "gb10":  [(12, 1)],              # sm_121 / sm_121a
    "5090":  [(12, 0)],              # sm_120 / sm_120a (and variants)
    "both":  [(12, 1), (12, 0)],     # GB10 + 5090 in one pass
}

# Replacement values
NEW_CC       = (10, 0)      # sm_100
NEW_FLAGS    = (10, 0)      # compatibility
ENTRY_SIZE   = 20           # struct cc_entry — 5 × u32


def find_libcuda():
    """Try to locate libcuda.so.<version> on the system."""
    candidates = []
    for p in [
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/lib/aarch64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ]:
        if not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            if re.match(r"libcuda\.so\.\d", name):
                candidates.append(os.path.join(p, name))
    real = [c for c in candidates if not os.path.islink(c)]
    if real:
        return real[0]
    if candidates:
        return candidates[0]
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True)
        for line in out.splitlines():
            m = re.search(r"libcuda\.so\.\d[\.\d]*\s+.*=>\s+(\S+)", line)
            if m and not os.path.islink(m.group(1)):
                return m.group(1)
    except Exception:
        pass
    return None


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def _looks_like_entry(data, off):
    if off + ENTRY_SIZE > len(data):
        return False
    a, maj, mn, f1, f2 = struct.unpack_from("<5I", data, off)
    return (0 <= a <= 0xFF and is_plausible_cc(maj, mn)
            and 0 <= f1 <= 0xFF and 0 <= f2 <= 0xFF)


def _walk_run(data, start_off):
    off = start_off
    ccs = set()
    while off + ENTRY_SIZE <= len(data) and _looks_like_entry(data, off):
        _, maj, mn, _, _ = struct.unpack_from("<5I", data, off)
        ccs.add((maj, mn))
        off += ENTRY_SIZE
    return (off - start_off) // ENTRY_SIZE, off, ccs


def find_cc_table(data):
    candidates = []
    seen_starts = set()

    for maj, mn in REQUIRED_ANCHORS:
        patt = struct.pack("<I", maj) + struct.pack("<I", mn)
        p = 0
        while True:
            idx = data.find(patt, p)
            if idx < 0:
                break
            p = idx + 1
            cand_entry = idx - 4
            if cand_entry < 0 or not _looks_like_entry(data, cand_entry):
                continue
            start = cand_entry
            while start - ENTRY_SIZE >= 0 and _looks_like_entry(data, start - ENTRY_SIZE):
                start -= ENTRY_SIZE
            if start in seen_starts:
                continue
            seen_starts.add(start)
            count, end, ccs = _walk_run(data, start)
            if count >= 10:
                candidates.append((count, start, end, ccs))

    good = [c for c in candidates if all(a in c[3] for a in REQUIRED_ANCHORS)]
    if good:
        good.sort(key=lambda c: -c[0])
        return good[0][:3]
    if candidates:
        candidates.sort(key=lambda c: -c[0])
        return candidates[0][:3]
    return (0, 0, 0)


def patch_entries(data, start, end, target_ccs):
    patched = []
    buf = bytearray(data)
    new_maj, new_min = NEW_CC
    new_f1, new_f2 = NEW_FLAGS
    for off in range(start, end, ENTRY_SIZE):
        a, maj, mn, f1, f2 = struct.unpack_from("<5I", buf, off)
        if (maj, mn) in target_ccs:
            struct.pack_into("<5I", buf, off, a, new_maj, new_min, new_f1, new_f2)
            patched.append((off, a, (maj, mn), (f1, f2)))
    return bytes(buf), patched


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--lib", help="path to libcuda.so.<version> (autodetect by default)")
    ap.add_argument("--target", choices=list(TARGETS), default="gb10",
                    help="which arch to rewrite into sm_100 (gb10 / 5090 / both)")
    ap.add_argument("--out", default="./patched_libcuda",
                    help="directory for the patched copy")
    ap.add_argument("--backup", default="./backups",
                    help="directory for the original backup")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan only, do not write anything")
    ap.add_argument("--force", action="store_true",
                    help="do not prompt for confirmation")
    args = ap.parse_args()

    lib_path = args.lib or find_libcuda()
    if not lib_path:
        print("ERROR: libcuda.so.<version> not found. Pass --lib manually.", file=sys.stderr)
        sys.exit(2)
    lib_path = os.path.realpath(lib_path)

    print("== patch_libcuda.py ==")
    print(f"libcuda:   {lib_path}")
    print(f"target:    {args.target}  (CCs to rewrite: {TARGETS[args.target]})")
    print(f"replace with: cc={NEW_CC[0]}.{NEW_CC[1]}, flags={NEW_FLAGS}")
    print(f"backup to: {args.backup}")
    print(f"output to: {args.out}")
    print()

    data = read_file(lib_path)
    count, start, end = find_cc_table(data)
    if count < 40:
        print(
            f"ERROR: CC table not found (longest contiguous run was {count} entries).",
            file=sys.stderr,
        )
        sys.exit(3)

    print(f"found CC table: {count} entries @ 0x{start:x}")
    print("relevant entries:")
    for off in range(start, end, ENTRY_SIZE):
        a, maj, mn, f1, f2 = struct.unpack_from("<5I", data, off)
        if (maj, mn) in TARGETS[args.target] or (maj, mn) == (10, 0):
            marker = "  <-- TARGET" if (maj, mn) in TARGETS[args.target] else ""
            print(f"    0x{off:08x}  arch_id=0x{a:02x} ({a:3d})  cc={maj}.{mn}  flags=({f1}, {f2}){marker}")

    new_data, patched = patch_entries(data, start, end, TARGETS[args.target])
    if not patched:
        print("NOTHING TO PATCH (no target CC entries found).", file=sys.stderr)
        sys.exit(4)

    print("\nplan:")
    for off, a, old_cc, old_flags in patched:
        print(
            f"  patch @0x{off:08x}: arch_id=0x{a:02x} cc {old_cc[0]}.{old_cc[1]} -> {NEW_CC[0]}.{NEW_CC[1]}, flags {old_flags} -> {NEW_FLAGS}"
        )

    if args.dry_run:
        print("\ndry-run requested; nothing written")
        return

    if not args.force:
        resp = input("\nWrite patched copy? [y/N] ").strip().lower()
        if resp not in {"y", "yes"}:
            print("aborted")
            return

    out_dir = Path(args.out)
    backup_dir = Path(args.backup)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    lib_name = Path(lib_path).name
    backup_path = backup_dir / f"{lib_name}.orig"
    out_path = out_dir / lib_name

    if not backup_path.exists():
        shutil.copy2(lib_path, backup_path)
        print(f"backup written: {backup_path}")
    else:
        print(f"backup exists:   {backup_path}")

    with open(out_path, "wb") as f:
        f.write(new_data)
    os.chmod(out_path, 0o755)

    symlink_path = out_dir / "libcuda.so.1"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(lib_name)

    print(f"patched copy:    {out_path}")
    print(f"symlink:         {symlink_path} -> {lib_name}")
    print()
    print("Use via:")
    print(f"  LD_LIBRARY_PATH={out_dir}:$LD_LIBRARY_PATH <your CUDA program>")


if __name__ == "__main__":
    main()
