# Contributing

That would be awesome if you want to contribute something to TileLang!

- [Contributing](CONTRIBUTING.md#contributing)
  - [Reporting Bugs](CONTRIBUTING.md#reporting-bugs)
  - [Asking Questions](CONTRIBUTING.md#asking-questions)
  - [Submitting Pull Requests](CONTRIBUTING.md#submitting-pull-requests)
  - [Repository Setup](CONTRIBUTING.md#repository-setup)
  - [Running Tests](CONTRIBUTING.md#running-tests)

## Reporting Bugs

If you run into any weird behavior while using TileLang, feel free to open a new issue in this repository! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open must include:

- Code snippet that reproduces the bug with a minimal setup.
- A clear explanation of what the issue is.


## Asking Questions

Please ask questions in issues.

## Submitting Pull Requests

All pull requests are super welcomed and greatly appreciated! Issues in need of a solution are marked with a [`♥ help`](https://github.com/ianstormtaylor/TileLang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22) label if you're looking for somewhere to start.

Please run `./format.sh` before submitting a pull request to make sure that your code is formatted correctly.

Please include tests and docs with every pull request!

## Repository Setup

To run the build, you need to have the TileLang repository cloned to your computer. After that, you need to `cd` into the directory where you cloned it, and install the dependencies with `python`:

```bash
python setup.py install
```


## Running Tests

To run the tests, start by building the project as described in the [Repository Setup](CONTRIBUTING.md#repository-setup) section.

Then you can rerun the tests with:

```text
python -m pytest testing
```

