import tilelang
import tilelang.testing
import tilelang.language as T


def test_tilelang_intimm():
    T.int32(0x7FFFFFFF)
    T.int32(-0x7FFFFFFF - 1)
    T.uint32(0xFFFFFFFF)
    T.int64(0x7FFFFFFFFFFFFFFF)
    T.int64(-0x7FFFFFFFFFFFFFFF - 1)
    T.uint64(0xFFFFFFFFFFFFFFFF)

    a = T.int32()
    a & 0x7FFFFFFF

    a = T.uint32()
    a & 0xFFFFFFFF

    a = T.int64()
    a & 0x7FFFFFFFFFFFFFFF

    a = T.uint64()
    a & T.uint64(0xFFFFFFFFFFFFFFFF)


if __name__ == "__main__":
    tilelang.testing.main()
