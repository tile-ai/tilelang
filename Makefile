PROJECT_NAME   = tilelang
SHELL          = /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c
PROJECT_PATH   = $(PROJECT_NAME)
SOURCE_FOLDERS = $(PROJECT_PATH) benchmark docs examples maint src testing
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.py" -o -iname "*.pyi") version_provider.py
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.[ch]pp" -o -iname "*.cc" -o -iname "*.c" -o -iname "*.h")
CUDA_FILES     = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.cu" -o -iname "*.cuh")
COMMIT_HASH    = $(shell git rev-parse HEAD)
COMMIT_HASH_SHORT = $(shell git rev-parse --short=7 HEAD)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTEST         ?= $(PYTHON) -X dev -m pytest -Walways
PYTESTOPTS     ?=
CMAKE_CONFIGURE_OPTS ?=

.PHONY: default
default: install

.PHONY: install
install:
	$(PYTHON) -m pip install -v .

.PHONY: install-editable install-e
install-editable install-e:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade --requirement requirements-dev.txt
	$(PYTHON) -m pip install -v --no-build-isolation --editable .

.PHONY: uninstall
uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(1))

.PHONY: pre-commit-install
pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

.PHONY: python-format-install
python-format-install:
	$(call check_pip_install,yapf)

.PHONY: ruff-install
ruff-install:
	$(call check_pip_install,ruff)

.PHONY: docs-install
docs-install:
	$(PYTHON) -m pip install --requirement docs/requirements.txt

.PHONY: pytest-install
pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-xdist)

.PHONY: test-install
test-install: pytest-install
	$(PYTHON) -m pip install --requirement requirements-test.txt

.PHONY: cmake-install
cmake-install:
	command -v cmake || $(call check_pip_install,cmake)

.PHONY: clang-format-install
clang-format-install:
	$(call check_pip_install,clang-format)

.PHONY: clang-tidy-install
clang-tidy-install:
	$(call check_pip_install,clang-tidy)

# Tests

.PHONY: pytest test
pytest test: pytest-install
	$(PYTEST) --version
	cd testing && $(PYTHON) -X dev -Walways -Werror -c 'import $(PROJECT_NAME)' && \
	$(PYTEST) --verbose --color=yes --durations=10 --showlocals \
		$(PYTESTOPTS) .

# Python Linters

.PHONY: pre-commit
pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit --version
	$(PYTHON) -m pre_commit run --all-files

.PHONY: python-format pyfmt yapf
python-format pyfmt yapf: python-format-install
	$(PYTHON) -m yapf --version
	@$(PYTHON) -m yapf --quiet docs/conf.py &>/dev/null || true
	$(PYTHON) -m yapf --in-place --parallel $(PYTHON_FILES)

.PHONY: ruff
ruff: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check .

.PHONY: ruff-fix
ruff-fix: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check --fix --exit-non-zero-on-fix .

# C++ Linters

.PHONY: cmake-configure
cmake-configure: cmake-install
	cmake --version
	cmake -S . -B cmake-build \
		--fresh $(CMAKE_CONFIGURE_OPTS) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

.PHONY: cmake cmake-build
cmake cmake-build: cmake-configure
	cmake --build cmake-build --parallel

.PHONY: clang-format
clang-format: clang-format-install
	clang-format --version
	clang-format --style=file --Werror -i $(CXX_FILES)

.PHONY: clang-tidy
clang-tidy: clang-tidy-install cmake-configure
	clang-tidy --version
	if [[ -x "$(shell command -v run-clang-tidy)" ]]; then \
		run-clang-tidy -clang-tidy-binary="$(shell command -v clang-tidy)" \
			-fix -p="cmake-build" $(CXX_FILES); \
	else \
		clang-tidy --fix -p="cmake-build" $(CXX_FILES); \
	fi

# Documentation

.PHONY: docs
docs: docs-install
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-autobuild)
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs docs/_build

.PHONY: clean-docs
clean-docs:
	$(MAKE) -C docs clean || true

# Utility Functions

.PHONY: format
format: pre-commit python-format ruff clang-format

.PHONY: lint
lint: format clang-tidy

.PHONY: clean-python
clean-python:
	find . -type f -name '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +

.PHONY: clean-build
clean-build:
	rm -rf build/ dist/ cmake-build/ cmake-build-*/
	find $(PROJECT_PATH) -type f -name '*.so' -delete
	rm -rf *.egg-info .eggs

.PHONY: clean
clean: clean-python clean-build clean-docs
