These tests are written for pytest and focus on the functions defined in testing/python/language/test_customize.py.

Key points:
- Framework: pytest (no additional dependencies added).
- Heavy external modules (tvm, tilelang) are stubbed via sys.modules to keep tests fast and deterministic.
- We validate public interfaces, happy paths, edge cases, and error handling using lightweight fakes/recorders.
- If your environment already provides tvm/tilelang, the stubs will still be used because tests import the module under test after installing stubs.