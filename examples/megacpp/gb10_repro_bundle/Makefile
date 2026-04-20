# GB10 reproducibility bundle:
# - build a baseline sm_100a cubin and rewrite only its ELF arch field;
# - walk the staged tcgen05 gate path with symbol and .nv.info patches;
# - keep the deeper libcuda patch lane separate under driver_patch_lane/.

CUDA      ?= /usr/local/cuda
NVCC      ?= $(CUDA)/bin/nvcc
CXX       ?= g++

SRC_ARCH  ?= sm_100a
DST_ARCH  ?= sm_121a

BASE_SRC      := kernel_baseline_100a.cubin
BASE_DST      := kernel_baseline_patched.cubin
ALLOC_SRC     := alloc_100a.cubin
ALLOC_DST     := alloc_patched.cubin
ALLOC_INFO    := alloc_patched_info.cubin
FULL_SRC      := kernel_sm100a.cubin
FULL_DST      := kernel_sm100a_patched.cubin
LOADER        := loader
QUERY_ATTRS   := query_attrs

NVCCFLAGS := -arch=$(SRC_ARCH) --cubin -std=c++17 -lineinfo \
             -Xptxas -gno-tmem-access-check
CXXFLAGS  := -O2 -std=c++17 -I$(CUDA)/include
LDFLAGS   := -L$(CUDA)/lib64 -lcuda

.PHONY: all clean inspect build-baseline build-alloc build-full \
        run-baseline probe-alloc-gates run-full-alloc run-full-mma run-full-tma

all: $(BASE_DST) $(ALLOC_DST) $(FULL_DST) $(LOADER) $(QUERY_ATTRS)

$(BASE_SRC): kernel_baseline.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BASE_DST): $(BASE_SRC) patch_elf.py
	./patch_elf.py $(BASE_SRC) $(BASE_DST) $(SRC_ARCH) $(DST_ARCH)

$(ALLOC_SRC): kernel_alloc_only.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(ALLOC_DST): $(ALLOC_SRC) patch_elf.py
	./patch_elf.py $(ALLOC_SRC) $(ALLOC_DST) $(SRC_ARCH) $(DST_ARCH)

$(FULL_SRC): kernel_sm100a.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(FULL_DST): $(FULL_SRC) patch_elf.py
	./patch_elf.py $(FULL_SRC) $(FULL_DST) $(SRC_ARCH) $(DST_ARCH)

$(LOADER): loader.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(QUERY_ATTRS): query_attrs.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

build-baseline: $(BASE_DST) $(LOADER) $(QUERY_ATTRS)

build-alloc: $(ALLOC_DST) $(LOADER)

build-full: $(FULL_DST) $(LOADER)

inspect: all
	@echo "--- baseline source cubin ($(SRC_ARCH)) ---"
	@readelf -h $(BASE_SRC) | grep -E "Flags|Machine"
	@echo "--- baseline patched cubin (-> $(DST_ARCH)) ---"
	@readelf -h $(BASE_DST) | grep -E "Flags|Machine"
	@echo "--- alloc patched cubin (-> $(DST_ARCH)) ---"
	@readelf -h $(ALLOC_DST) | grep -E "Flags|Machine"

run-baseline: build-baseline
	./$(LOADER) $(BASE_DST) k_baseline 32

probe-alloc-gates: build-alloc
	./$(LOADER) $(ALLOC_DST) k_tcgen05_alloc 128 || true
	./patch_symbols.py $(ALLOC_DST) $(ALLOC_DST) .nv.reservedSmem.offset0 .nv.reservedSmem.cap
	./$(LOADER) $(ALLOC_DST) k_tcgen05_alloc 128 || true
	./patch_nvinfo.py $(ALLOC_DST) $(ALLOC_INFO) k_tcgen05_alloc
	./$(LOADER) $(ALLOC_INFO) k_tcgen05_alloc 128 || true

run-full-alloc: build-full
	./$(LOADER) $(FULL_DST) k_tcgen05_alloc 128

run-full-mma: build-full
	./$(LOADER) $(FULL_DST) k_tcgen05_mma 128

run-full-tma: build-full
	./$(LOADER) $(FULL_DST) k_tma_multicast 128 1 2 16384

clean:
	rm -f $(BASE_SRC) $(BASE_DST) \
	      $(ALLOC_SRC) $(ALLOC_DST) $(ALLOC_INFO) \
	      $(FULL_SRC) $(FULL_DST) \
	      $(LOADER) $(QUERY_ATTRS)
