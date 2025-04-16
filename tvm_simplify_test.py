
import tilelang
import tvm
from tvm import te

def test_simplify():
    # Define symbolic variables
    i = te.var("i")
    tx = te.var("tx")
    vec = te.var("vec")
    ana = tvm.arith.Analyzer()
    ana.bind(i, tvm.ir.Range.from_min_extent(0, 1024))
    ana.bind(tx, tvm.ir.Range.from_min_extent(0, 128))
    ana.bind(vec, tvm.ir.Range.from_min_extent(0, 8))
    # Create the expression
    # expr = (tx * 8 + vec) // 576
    expr = (i * 1024 + tx * 8 + vec) // 576
    # Simplify the expression
    simplified = ana.simplify(expr, steps=16)
    
    print("Original expression:", expr)
    print("Simplified expression:", simplified)

if __name__ == "__main__":
    test_simplify() 