.PHONY: env build build-checked test clean

# Project-local venv, managed by uv. Created once; `make test` depends on
# it so a fresh clone just needs `make test` after the submodule is
# checked out. `uv venv` errors if .venv already exists (no implicit
# reuse), so guard creation -- otherwise every `make test` after the
# first fails outright instead of just reinstalling requirements.
#
# torch comes from PyTorch's own index, not PyPI: PyPI's default Linux
# wheel targets CUDA 13, which needs NVIDIA driver >= 580 -- on an older
# driver CUDA silently reports unavailable. cu126 runs on any driver
# >= 525 (CUDA 12.x minor-version compatibility) and newer, and covers
# every GPU up to Hopper. Override for other hardware:
#   make env TORCH_INDEX=https://download.pytorch.org/whl/cu130  # Blackwell
#   make env TORCH_INDEX=https://download.pytorch.org/whl/cpu    # no GPU
# Installed before requirements.txt so its plain `torch` is already
# satisfied and not swapped for the PyPI build.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu126

env:
	[ -d .venv ] || uv venv .venv --python 3.11
	uv pip install --python .venv/bin/python torch --index-url $(TORCH_INDEX)
	uv pip install --python .venv/bin/python -r requirements.txt

build:
	./fortran/build.sh
	./fortran/sacsma_build.sh

build-checked:
	BOUNDS_CHECK=1 ./fortran/build.sh
	BOUNDS_CHECK=1 ./fortran/sacsma_build.sh

test: env build
	.venv/bin/python -m pytest tests/ -v

clean:
	rm -f fortran/*.so fortran/*.o *.mod
	rm -rf fortran/_sacsma_patched
