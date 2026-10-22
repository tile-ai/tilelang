# ST Case Specification

Use a JSON specification when a task includes many cases or contract uncertainty makes a reviewable checklist valuable. Small, straightforward changes with unambiguous meaning may be written directly in pytest.

Deliveries built from scratch and final acceptance must also include a coverage assessment across ten dimensions. Each dimension must record one of `covered`, `partial`, `missing`, `unknown`, or `not_applicable`, along with the case IDs and rationale supporting that conclusion. Only `covered` and reviewed `not_applicable` statuses pass the complete-delivery gate.

```json
{
  "coverage": {
    "functionality": {
      "status": "covered",
      "case_ids": ["tail-plus-one"],
      "rationale": "This case compares every output value required by the documentation."
    },
    "gradient": {
      "status": "not_applicable",
      "case_ids": [],
      "rationale": "The public operator has no backward-computation contract."
    },
    "execution_evidence": {
      "status": "missing",
      "case_ids": [],
      "rationale": "The kernel has not been implemented yet."
    }
  }
}
```

The complete key set is `functionality`, `precision`, `boundary`, `gradient`, `state_mutation`, `invalid_rejection`, `layout_interface`, `backend_branch`, `randomness`, and `execution_evidence`.

## Data Structure

```json
{
  "schema_version": 1,
  "operator": "example_op",
  "contract_sources": [
    {
      "id": "api",
      "kind": "public_docstring",
      "path": "operators/example/kernel.py",
      "symbol": "example_op",
      "claim": "Describes the input and output shapes."
    }
  ],
  "requirements": [
    {
      "id": "R-OUTPUT",
      "statement": "Return one output value for every input element.",
      "source_ids": ["api"],
      "applicability": "confirmed"
    }
  ],
  "cases": [
    {
      "id": "tail-plus-one",
      "kind": "boundary",
      "level": 1,
      "requirement_ids": ["R-OUTPUT"],
      "inputs": {"num_elements": 129},
      "oracles": [
        {"kind": "pytorch_reference", "source_id": "api"}
      ],
      "assertions": ["shape", "dtype", "values"],
      "path_evidence": {
        "kind": "tail",
        "axis": "num_elements",
        "logical_size": 129,
        "block_size": 128,
        "source_id": "api"
      },
      "status": "designed",
      "result": "not_run"
    }
  ]
}
```

Allowed source kinds include `user_requirement`, `interface_doc`, `design_doc`, `public_docstring`, `api_validation`, `mathematical_definition`, `reference`, `cuda_implementation`, `ascend_implementation`, `existing_test`, and `implementation`.

Primary oracle kinds are `pytorch_reference`, `cpu_reference`, `exact_expected`, `mathematical`, `metamorphic`, and `exception_contract`. `cuda_differential` and `legacy_differential` are auxiliary oracles and cannot serve as the sole semantic authority.

Assertion names describe what is checked; they are not executable code. Common values include `shape`, `dtype`, `device`, `values`, `gradients`, `aux_outputs`, `state_changed`, `state_unchanged`, `protected_storage`, `exception_type`, and `exception_message`.

Lifecycle statuses are `designed`, `implemented`, `collected`, and `executed`. Run results are `not_run`, `passed`, `failed`, `skipped`, `xfailed`, and `error`. Only a case with status `executed` may have a result other than `not_run`; cases with status `collected` or `executed` must include `test_nodeid`.

Running `scripts/check_st_spec.py` detects structural errors, missing source references, correctness conclusions based only on shape checks, CUDA-only oracles, incorrect tail markers, weak negative tests, and inconsistencies between lifecycle status and result. Passing this checker means only that the specification is valid; it does not prove that the operator is correct or that pytest has run.

The delivery gate requires all ten dimensions to be assessed:

```bash
python scripts/check_st_spec.py path/to/spec.json \
  --repo-root . --require-complete-coverage
```

With this option enabled, only `covered` or reviewed `not_applicable` statuses are accepted. `execution_evidence: covered` must reference a case whose lifecycle status is `executed` and whose result is `passed`.
