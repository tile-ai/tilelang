import tilelang.language as T
from tilelang.language.eager.ast import mutate


def test_augassign_ast_no_placeholder_collision_for_value():
    def demo(
        A: T.Tensor[(1,), T.int32],
        B: T.Tensor[(1,), T.int32],
    ):
        value = A[0]
        value -= 1
        B[0] = value

    ir_gen = mutate(demo)
    # Regression test: the AugAssign lowering previously used placeholder names
    # that collided with common user variable names like `value`, producing:
    #   value = __tb.aug_assign('Sub', 1, 1)
    assert "__tb.aug_assign('Sub', value, 1, name='value')" in ir_gen.source
    assert "__tb.aug_assign('Sub', 1, 1)" not in ir_gen.source


def test_augassign_immutable_var_is_lowered_as_rebind():
    @T.prim_func
    def main(
        A: T.Tensor[(1,), T.int32],
        B: T.Tensor[(1,), T.int32],
    ):
        x = A[0]
        x -= 1
        B[0] = x

    # Just building the PrimFunc is sufficient; pre-fix this raised because
    # augmented assignment on immutable `Var` was rejected.
    assert main is not None
