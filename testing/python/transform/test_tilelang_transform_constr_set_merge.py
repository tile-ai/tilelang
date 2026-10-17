# Regression tests for ConstrSet::Merge bind-before-use normalization and
# Constr::Populate predicate expansion (issue #3220).
#
# Merge concatenates two lexically-collected sets, so an entry of the first
# set can read a bind var the second set defines. Replaying that order
# digested the predicate before its definition existed (EnterConstraint
# evaluates eagerly, exactly like Bind) and the guard fact was lost.
import tilelang.testing
from tilelang import tvm
from tvm import tirx

_prove_raw = tvm.get_global_func("tl.analysis.ConstrSetsProve")


def _prove(first, second, goal):
    def tag(entries):
        return [[tirx.StringImm(e[0]), *e[1:]] for e in entries]

    return _prove_raw(tag(first), tag(second), goal)


def _i32(name):
    return tirx.Var(name, "int32")


def _c(value):
    return tirx.IntImm("int32", value)


def test_merged_predicate_before_bind_recovers_guard():
    # Set A holds the guard as an opaque bound boolean; set B defines it.
    # After Merge the predicate precedes the bind; normalization must hoist
    # the bind so the guard expands and the fact survives.
    i = _i32("i")
    cond = tirx.Var("cond", "bool")
    first = [["pred", cond]]
    second = [["range", i, _c(0), _c(256)], ["bind", cond, i < _c(64)]]
    assert _prove(first, second, i < _c(64))


def test_merged_bind_chain_hoists_transitively():
    # The referenced bind itself reads another later-defined bind; the hoist
    # must pull the whole definition chain, in dependency order.
    i = _i32("i")
    inner = tirx.Var("inner", "bool")
    outer = tirx.Var("outer", "bool")
    first = [["pred", outer]]
    second = [
        ["range", i, _c(0), _c(256)],
        ["bind", inner, i < _c(64)],
        ["bind", outer, tirx.And(inner, i < _c(32))],
    ]
    assert _prove(first, second, i < _c(32))


def test_assume_predicate_expands_too():
    i = _i32("i")
    cond = tirx.Var("cond", "bool")
    first = [["assume", cond]]
    second = [["range", i, _c(0), _c(256)], ["bind", cond, i < _c(64)]]
    assert _prove(first, second, i < _c(64))


def test_program_order_tightening_is_preserved():
    # The #2805 guarantee: a bind evaluated under a preceding predicate
    # captures the tightened bound. Within one set nothing reads a later
    # bind, so normalization must not reorder here.
    tx = _i32("tx")
    v = _i32("v")
    first = [
        ["range", tx, _c(0), _c(256)],
        ["pred", tx < _c(64)],
        ["bind", v, tx],
    ]
    assert _prove(first, [], v < _c(64))


def test_unbound_guard_stays_unprovable():
    # Sanity: without the defining bind the goal must not become provable.
    i = _i32("i")
    cond = tirx.Var("cond", "bool")
    first = [["pred", cond]]
    second = [["range", i, _c(0), _c(256)]]
    assert not _prove(first, second, i < _c(64))


if __name__ == "__main__":
    tilelang.testing.main()
