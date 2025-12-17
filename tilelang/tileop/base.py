from __future__ import annotations
from enum import IntEnum


class GemmWarpPolicy(IntEnum):
    """
    Enumeration for GEMM Warp Partitioning Policies.
    """

    Square = 0  # Balance warps evenly in a "square" aspect ratio.
    FullRow = 1  # Assign all warps to rows.
    FullCol = 2  # Assign all warps to columns.

    def is_square(self) -> bool:
        """
        Check if the policy is a square partitioning.

        Returns:
            bool: True if the policy is square, False otherwise.
        """
        return self == GemmWarpPolicy.Square

    def is_full_row(self) -> bool:
        """
        Check if the policy is a full row partitioning.

        Returns:
            bool: True if the policy is full row, False otherwise.
        """
        return self == GemmWarpPolicy.FullRow

    def is_full_col(self) -> bool:
        """
        Check if the policy is a full column partitioning.

        Returns:
            bool: True if the policy is full column, False otherwise.
        """
        return self == GemmWarpPolicy.FullCol

    @staticmethod
    def to_prime_factors(num):
        """
        Compute the prime factorization of a given number.

        Args:
            num (int): The number to factorize.

        Returns:
            list: A list of prime factors of the number.
        """
        factors = []
        i = 2
        # Find all prime factors up to the square root of the number.
        while i * i <= num:
            while num % i == 0:  # Check divisibility by `i`.
                factors.append(i)
                num //= i
            i += 1
        # If the remaining number is greater than 1, it's a prime factor.
        if num > 1:
            factors.append(num)
        return factors

    def compute_warp_partition(self, M, N, num_warps):
        """
        Compute the warp partition (m_warp, n_warp) based on the given policy.

        Args:
            M (int): The number of rows in the GEMM workload.
            N (int): The number of columns in the GEMM workload.
            num_warps (int): The total number of warps available.

        Returns:
            tuple: A tuple (m_warp, n_warp) representing the partitioning of warps.

        Raises:
            ValueError: If the policy is invalid or the partitioning fails.
            AssertionError: If M or N is not divisible by the required factor for FullRow or FullCol policies.
        """
        m_warp = 1  # Initial warp count for rows.
        n_warp = 1  # Initial warp count for columns.

        if self.is_full_row():
            # FullRow policy: Allocate all warps to rows.
            m_warp = num_warps
            n_warp = 1

            # If M cannot be evenly divided by m_warp*16, try to split remaining warps to N
            if M % (m_warp * 16) != 0:
                # Calculate how many warps we can use for M
                max_m_warps = M // 16
                m_warp = max_m_warps
                # Use remaining warps for N
                n_warp = num_warps // m_warp
                if n_warp == 0:
                    n_warp = 1

        elif self.is_full_col():
            # FullCol policy: Allocate all warps to columns.
            m_warp = 1
            n_warp = num_warps

            # If N cannot be evenly divided by n_warp*8, try to split remaining warps to M
            if N % (n_warp * 8) != 0:
                # Calculate how many warps we can use for N
                max_n_warps = N // 8
                n_warp = max_n_warps
                # Use remaining warps for M
                m_warp = num_warps // n_warp
                if m_warp == 0:
                    m_warp = 1

        elif self.is_square():
            # First calculate the maximum possible warps for each dimension
            max_m_warps = M // 16  # Each warp needs at least 16 elements in M
            max_n_warps = N // 8  # Each warp needs at least 8 elements in N

            # Calculate the ideal ratio of M/N warps based on the matrix dimensions
            ideal_ratio = 1.0
            if N > 0:
                ideal_ratio = float(M) / N

            # Start with a balanced initial guess
            m_warp = 1
            n_warp = 1

            # Try to find the best balanced partition
            best_m = 1
            best_n = 1
            best_balance = float("inf")

            # Try all possible combinations that satisfy the constraints
            for m in range(1, min(max_m_warps, num_warps) + 1):
                n = num_warps // m
                if n > max_n_warps:
                    continue
                if m * n != num_warps:
                    continue

                # Calculate how balanced this partition is
                m_per_warp = float(M) / (m * 16)
                n_per_warp = float(N) / (n * 8)
                balance = abs(m_per_warp / n_per_warp - ideal_ratio)

                if balance < best_balance:
                    best_balance = balance
                    best_m = m
                    best_n = n

            m_warp = best_m
            n_warp = best_n

        else:
            # Raise an error for unknown policies.
            raise ValueError(f"Unknown GemmWarpPolicy: {self}")

        return m_warp, n_warp

    @classmethod
    def from_warp_partition(cls, m_warp: int, n_warp: int) -> GemmWarpPolicy:
        """
        Determine the warp policy based on the given warp partitioning.

        Args:
            m_warp (int): Number of warps in the row dimension
            n_warp (int): Number of warps in the column dimension

        Returns:
            GemmWarpPolicy: The corresponding warp policy

        Examples:
            >>> GemmWarpPolicy.from_block_row_cols(4, 1)  # All warps in rows
            GemmWarpPolicy.FullRow
            >>> GemmWarpPolicy.from_block_row_cols(1, 4)  # All warps in columns
            GemmWarpPolicy.FullCol
            >>> GemmWarpPolicy.from_block_row_cols(2, 2)  # Balanced distribution
            GemmWarpPolicy.Square
        """
        if n_warp == 1 and m_warp > 1:
            return cls.FullRow
        elif m_warp == 1 and n_warp > 1:
            return cls.FullCol
        else:
            return cls.Square
