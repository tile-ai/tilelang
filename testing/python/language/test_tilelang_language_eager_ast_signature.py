from tilelang.language.eager.ast import BaseBuilder, mutate


class _TestBuilder(BaseBuilder):
    def override(self, name: str):
        if name == "range":
            return range
        return super().override(name)

    def set_fileline(self, filename: str, lineno: int, name: str):
        pass


def test_mutate_accepts_varargs_parameter():
    def first_arg(*args):
        return args[0]

    ir_gen = mutate(first_arg)

    assert ir_gen.gen(_TestBuilder())("sentinel", "ignored") == "sentinel"


def test_mutate_preserves_kwargs_parameter():
    def get_kwarg(**kwargs):
        return kwargs["key"]

    ir_gen = mutate(get_kwarg)

    assert ir_gen.gen(_TestBuilder())(key="value") == "value"
