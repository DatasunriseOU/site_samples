#!/usr/bin/env python3
"""
Convert selected WEAK UND symbols in a CUDA .cubin to defined symbols.

This is used in the GB10 gate walk for reserved-SMEM symbols that the sm_121
driver does not resolve the same way as the sm_100 path.
"""

import sys, struct, os

SHN_ABS   = 0xFFF1
STB_GLOBAL = 1

EHDR_FMT = "<16sHHIQQQIHHHHHH"
SHDR_FMT = "<IIQQQQIIQQ"
SYM_FMT  = "<IBBHQQ"

def read_cstring(buf, off):
    end = buf.index(b'\x00', off)
    return buf[off:end].decode()

def main():
    args = sys.argv[1:]
    shndx_target = SHN_ABS
    if "--shndx" in args:
        i = args.index("--shndx")
        shndx_target = int(args[i+1], 0)
        del args[i:i+2]
    if len(args) < 3:
        print(__doc__); sys.exit(1)
    src, dst = args[0], args[1]
    names = set(args[2:])

    with open(src, "rb") as f:
        data = bytearray(f.read())
    assert data[:4] == b"\x7fELF" and data[4] == 2, "need ELF64"

    (_, _, _, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum, _) = \
        struct.unpack_from(EHDR_FMT, data, 0)

    symtab_off=symtab_size=symtab_link=0
    for i in range(e_shnum):
        sh = struct.unpack_from(SHDR_FMT, data, e_shoff + i*e_shentsize)
        if sh[1] == 2:
            symtab_off, symtab_size, symtab_link = sh[4], sh[5], sh[6]
            break
    assert symtab_off, "no .symtab"

    sh_strtab = struct.unpack_from(SHDR_FMT, data, e_shoff + symtab_link*e_shentsize)
    strtab_off = sh_strtab[4]

    sym_size = struct.calcsize(SYM_FMT)
    count = symtab_size // sym_size
    patched = []
    for i in range(count):
        off = symtab_off + i*sym_size
        (st_name, st_info, st_other, st_shndx, st_value, st_size_) = \
            struct.unpack_from(SYM_FMT, data, off)
        if st_name == 0:
            continue
        name = read_cstring(data, strtab_off + st_name)
        if name in names and st_shndx == 0:
            new_info = (STB_GLOBAL << 4) | (st_info & 0x0F)
            struct.pack_into(SYM_FMT, data, off,
                             st_name, new_info, st_other, shndx_target,
                             st_value, st_size_)
            patched.append((name, st_value, st_size_, shndx_target))

    with open(dst, "wb") as f:
        f.write(data)
    os.chmod(dst, 0o644)

    print(f"[+] {src} -> {dst}")
    for n, v, s, sh in patched:
        sh_name = {0xFFF1:"ABS", 0xFFF2:"COMMON"}.get(sh, f"section#{sh}")
        print(f"    patched {n!r}: UND/WEAK -> {sh_name}/GLOBAL  (value=0x{v:x}, size={s})")
    if not patched:
        print("    (no matching UND symbols found)")

if __name__ == "__main__":
    main()
