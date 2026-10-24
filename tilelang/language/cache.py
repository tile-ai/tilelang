from enum import IntEnum


class EvictPriority(IntEnum):
    Default = -1
    EvictNormal = 0
    EvictFirst = 1
    EvictLast = 2
    EvictUnchanged = 3
    NoAllocate = 4
