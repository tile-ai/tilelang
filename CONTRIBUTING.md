# Contributing

That would be awesome if you want to contribute something to TileLang!

## Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Report Bugs](#report-bugs)
- [Ask Questions](#ask-questions)
- [Submit Pull Requests](#submit-pull-requests)
- [Setup Development Environment](#setup-development-environment)
- [Install Develop Version](#install-develop-version)
- [Lint Check](#lint-check)
- [Test Locally](#test-locally)
- [Build Wheels](#build-wheels)
- [Documentation](#documentation)

## Report Bugs

If you run into any weird behavior while using TileLang, feel free to open a new issue in this repository! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open must include:

- Code snippet that reproduces the bug with a minimal setup.
- A clear explanation of what the issue is.

## Ask Questions

Please ask questions in issues.

## Submit Pull Requests

All pull requests are super welcomed and greatly appreciated! Issues in need of a solution are marked with a [`♥ help`](https://github.com/ianstormtaylor/TileLang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22) label if you're looking for somewhere to start.

If you're new to contributing to TileLang, you can follow the following guidelines before submitting a pull request.

> [!NOTE]
> Please include tests and docs with every pull request if applicable!

## Setup Development Environment

Before contributing to TileLang, please follow the instructions below to setup.

1. Fork TileLang ([fork](https://github.com/tile-ai/tilelang/fork)) on GitHub and clone the repository.

    ```bash
    git clone --recurse-submodules git@github.com:<your username>/tilelang.git  # use the SSH protocol
    cd tilelang

    git remote add upstream git@github.com:tile-ai/tilelang.git
    ```

2. Setup a development environment:

    ```bash
    uv venv --seed .venv  # use `python3 -m venv .venv` if you don't have `uv`

    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel "build[uv]"
    uv pip install --requirements requirements-dev.txt
    ```

3. Setup the [`pre-commit`](https://pre-commit.com) hooks:

    ```bash
    pre-commit install --install-hooks
    ```

Then you are ready to rock. Thanks for contributing to TileLang!

## Install Develop Version

To install TileLang in an "editable" mode, run:

```bash
python3 -m pip install --no-build-isolation --verbose --editable .
```

in the main directory. This installation is removable by:

```bash
python3 -m pip uninstall tilelang
```

We also recommend installing TileLang in a more manual way for better control over the build process, by compiling the C++ extensions first and set the `PYTHONPATH`. See [Working from Source via `PYTHONPATH`](https://tilelang.com/get_started/Installation.html#working-from-source-via-pythonpath) for detailed instructions.

## Lint Check

To check the linting, run:

```bash
pre-commit run --all-files
```

## Test Locally

To run the tests, start by building the project as described in the [Setup Development Environment](#setup-development-environment) section.

Then you can rerun the tests with:

```bash
python3 -m pytest testing
```

## Build Wheels

_TBA_

## Documentation

_TBA_
