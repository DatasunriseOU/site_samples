#!/usr/bin/env python3
"""
Patch a CUDA .cubin ELF so the driver accepts it on a different SM target.

This is the narrowest patcher in the GB10 lane: it rewrites only the ELF
`e_flags` arch field and preserves the upper bits.
"""

import sys, struct, os

ARCH_LOW16 = {
    "sm_90":   0x5A00,
    "sm_90a":  0x5A02,
    "sm_100":  0x6400,
    "sm_100a": 0x6402,
    "sm_103":  0x6700,
    "sm_103a": 0x6702,
    "sm_120":  0x7800,
    "sm_120a": 0x7802,
    "sm_121":  0x7900,
    "sm_121a": 0x7902,
}

E_FLAGS_OFFSET = 0x30

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    src, dst, src_arch, dst_arch = sys.argv[1:]

    if dst_arch not in ARCH_LOW16:
        print(f"Unknown target arch {dst_arch}. Known: {sorted(ARCH_LOW16)}")
        sys.exit(2)
    new_low = ARCH_LOW16[dst_arch]

    with open(src, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF":
        print(f"{src}: not an ELF"); sys.exit(3)

    old_flags = struct.unpack_from("<I", data, E_FLAGS_OFFSET)[0]
    old_low   = old_flags & 0xFFFF
    upper     = old_flags & 0xFFFF0000
    new_flags = upper | new_low

    if src_arch in ARCH_LOW16 and ARCH_LOW16[src_arch] != old_low:
        print(f"WARN: source arch {src_arch} expected low 0x{ARCH_LOW16[src_arch]:04x}, "
              f"file has 0x{old_low:04x}")

    struct.pack_into("<I", data, E_FLAGS_OFFSET, new_flags)

    with open(dst, "wb") as f:
        f.write(data)
    os.chmod(dst, 0o644)

    print(f"[+] {src} -> {dst}")
    print(f"    e_flags: 0x{old_flags:08x}  ->  0x{new_flags:08x}  ({src_arch} -> {dst_arch})")
    print(f"    (upper 0x{upper:08x} preserved; arch low 0x{old_low:04x} -> 0x{new_low:04x})")

if __name__ == "__main__":
    main()
