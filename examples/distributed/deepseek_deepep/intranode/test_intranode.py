import tilelang
import tilelang.testing

import example_intranode


@tilelang.testing.requires_cuda
def test_intranode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["example_intranode.py"])  # optionally add testing params here
    example_intranode.main()


if __name__ == "__main__":
    tilelang.testing.main()
