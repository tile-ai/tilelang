import tilelang
import tilelang.testing
import tilelang.language as T


def test_tilelang_intimm():
    T.int32(0x7fffffff)
    T.int32(-0x7fffffff - 1)
    T.uint32(0xffffffff)
    T.int64(0x7fffffffffffffff)
    T.int64(-0x7fffffffffffffff - 1)
    T.uint64(0xffffffffffffffff)

    a = T.int32()
    a & 0x7fffffff

    a = T.uint32()
    a & 0xffffffff

    a = T.int64()
    a & 0x7fffffffffffffff

    a = T.uint64()
    a & T.uint64(0xffffffffffffffff)


if __name__ == '__main__':
    tilelang.testing.main()
