"""Extract finite memory and flag events from native IR without rewriting it."""

from itertools import product

from tvm_tirx import ir, sym
from testing.ascend._ir import allocated_tensors, attribute_fields, nodes, scheduled_fields, substitute


def extract(module):
    allocations = allocated_tensors(module, include_scheduled=True)
    a, b = ([value for value in allocations if value.name == name] for name in ("a", "b"))
    assert len(a) == len(b) == 1
    a, b = a[0], b[0]
    assert str(a.ty.scope) == str(b.ty.scope) == "shared.dyn"
    assert str(a.ty.dtype) == str(b.ty.dtype)
    storages = {a: "a", b: "b"}
    analyzer = sym.Analyzer()
    events = []

    def concrete(value, environment):
        result = analyzer.simplify(substitute(value, environment))
        assert isinstance(result, ir.IntImm), ("nonconstant index/guard", result)
        return int(result)

    def storage(value):
        return storages.get(value, "external:" + value.name)

    def scalar_reads(value, environment):
        value = analyzer.simplify(substitute(value, environment))
        for call in nodes(value, ir.Call):
            assert not isinstance(call.op, ir.Op) or call.op.name != "tirx.if_then_else", ("unresolved reader predicate", call)
        return [
            {"storage": storage(load.source), "index": [concrete(index, environment) for index in load.indices]}
            for load in nodes(value, ir.TensorLoad)
        ]

    def region_access(value, environment):
        assert isinstance(value, ir.TensorRegion)
        ranges = [
            range(concrete(region.begin, environment), concrete(region.begin, environment) + concrete(region.extent, environment))
            for region in value.region
        ]
        indices = list(product(*ranges))
        assert len(indices) <= 64
        return [{"storage": storage(value.source), "index": list(index)} for index in indices]

    def visit(statement, environment, loops=(), core=None, stage=0):
        context = {"loops": list(loops), "core": core, "stage": stage}
        if isinstance(statement, ir.SeqStmt):
            for child in statement.seq:
                visit(child, environment, loops, core, stage)
        elif isinstance(statement, ir.RegionStmt):
            if statement.op.name == "tirx.launch_thread":
                assert len(statement.body_params) == 1 and concrete(statement.args[1], environment) == 1
                visit(statement.body, {**environment, statement.body_params[0]: 0}, loops, core, stage)
                return
            assert statement.op.name == "tl.attr", statement.op.name
            fields = attribute_fields(statement)
            if fields["key"] == "tl.ascend_task":
                core = int(fields["node"]["core_mask"])
            elif fields["key"] == "tl.schedule_unit":
                stage = int(fields["node"]["stage"])
            else:
                assert fields["key"] in ("tl.sblock", "tl.loop.annotations", "tl.kernel"), fields["key"]
            visit(statement.body, environment, loops, core, stage)
        elif isinstance(statement, ir.For):
            begin, end, step = [concrete(value, environment) for value in (statement.begin, statement.end, statement.step)]
            assert step > 0 and 0 < len(range(begin, end, step)) <= 16
            for iteration in range(begin, end, step):
                visit(statement.body, {**environment, statement.var: iteration}, (*loops, [statement.var.name, iteration]), core, stage)
        elif isinstance(statement, ir.Bind):
            value = statement.value.args[0] if scheduled_fields(statement) is not None else statement.value
            assert isinstance(value, ir.Call) and value.op.name == "tirx.alloc_tensor", statement
        elif isinstance(statement, ir.TensorStore):
            events.append(
                {
                    **context,
                    "kind": "scalar",
                    "pipe": "S",
                    "reads": scalar_reads(statement.value, environment),
                    "writes": [
                        {"storage": storage(statement.dest), "index": [concrete(index, environment) for index in statement.indices]}
                    ],
                }
            )
        elif isinstance(statement, ir.Evaluate):
            call = statement.value
            assert isinstance(call, ir.Call) and isinstance(call.op, ir.Op), statement
            if call.op.name == "tl.annotation":
                assert not nodes(call, ir.TensorLoad)
                return
            if call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                kind = "set" if "set_flag" in call.op.name else "wait"
                channel = str(call.args[0].value)
                assert channel in ("S_MTE3", "MTE3_S"), channel
                events.append(
                    {
                        **context,
                        "kind": kind,
                        "pipe": channel.split("_")[0 if kind == "set" else 1],
                        "channel": channel,
                        "token": concrete(call.args[1], environment),
                    }
                )
            elif call.op.name == "tl.tileop.ascend_copy":
                assert str(call.args[0].source.ty.scope) == "shared.dyn"
                assert str(call.args[1].source.ty.scope) == "global"
                reads, writes = [region_access(region, environment) for region in call.args[:2]]
                assert reads and all(access["storage"] == "a" for access in reads)
                assert writes and all(access["storage"] not in ("a", "b") for access in writes)
                events.append({**context, "kind": "copy", "pipe": "MTE3", "reads": reads, "writes": writes})
            else:
                raise AssertionError(("unmodeled executable call", call.op.name))
        else:
            raise AssertionError(("unmodeled control or statement", type(statement).__name__))

    visit(module["main"].body, {})
    return events
