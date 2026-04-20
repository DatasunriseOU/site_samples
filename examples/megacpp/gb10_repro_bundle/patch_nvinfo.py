#!/usr/bin/env python3
"""
Strip selected tcgen05-specific capability records from `.nv.info.<kernel>`.

This keeps the gate walk explicit: the driver gets past more metadata, but the
public-safe lane still stops at the signed capability boundary.
"""

import sys, struct, os

STRIP_TAG_ATTR = [
    (0x04, 0x28),
    (0x04, 0x29),
    (0x04, 0x1e),
    (0x04, 0x4f),
    (0x01, 0x41),
    (0x01, 0x51),
    (0x02, 0x4c),
]

EHDR_FMT = "<16sHHIQQQIHHHHHH"
SHDR_FMT = "<IIQQQQIIQQ"

def main():
    if len(sys.argv) < 4:
        print("usage: patch_nvinfo.py <src> <dst> <kernel>...")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    kernels = sys.argv[3:]

    with open(src, "rb") as f:
        data = bytearray(f.read())

    e = struct.unpack_from(EHDR_FMT, data, 0)
    e_shoff, e_shentsize, e_shnum, e_shstrndx = e[6], e[11], e[12], e[13]

    shstr = struct.unpack_from(SHDR_FMT, data, e_shoff + e_shstrndx*e_shentsize)
    shstr_off = shstr[4]

    for i in range(e_shnum):
        sh = struct.unpack_from(SHDR_FMT, data, e_shoff + i*e_shentsize)
        name_off = shstr_off + sh[0]
        end = data.index(b'\x00', name_off)
        name = data[name_off:end].decode()
        if not name.startswith(".nv.info."): continue
        if not any(name == f".nv.info.{k}" for k in kernels): continue
        sec_off, sec_size = sh[4], sh[5]
        print(f"[{name}] size={sec_size}")
        pos = 0
        stripped = 0
        while pos + 4 <= sec_size:
            tag  = data[sec_off + pos + 0]
            attr = data[sec_off + pos + 1]
            if tag == 0:
                pos += 4
                continue
            if tag == 4:
                size = int.from_bytes(data[sec_off+pos+2: sec_off+pos+4], "little")
                total = 4 + size
            else:
                total = 4
            if (tag, attr) in STRIP_TAG_ATTR:
                print(f"  strip rec @+0x{pos:x}: tag=0x{tag:02x} attr=0x{attr:02x} "
                      f"len=0x{total:x}")
                for k in range(total):
                    data[sec_off + pos + k] = 0x00
                stripped += 1
            pos += total
        print(f"  stripped {stripped} record(s)")

    with open(dst, "wb") as f:
        f.write(data)
    os.chmod(dst, 0o644)
    print(f"[+] wrote {dst}")

if __name__ == "__main__":
    main()
