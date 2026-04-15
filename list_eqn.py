from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

# ============================================================
# Boolean function:
#   F[i] = f(bits_of_i)
#
# Variable order:
#   0:x, 1:y, 2:z, 3:t, 4:u, 5:w, 6:a, 7:b, 8:c
#
# Bits are encoded as an integer in the natural left-to-right order:
#   (x,y,z,t,u,w,a,b,c) -> binary integer
#
# Example:
#   (0,0,0,0,0,0,0,0,0) -> 0
#   (0,0,0,0,0,0,0,0,1) -> 1
#   (1,1,1,1,1,1,1,1,1) -> 511
#
# A pattern is a 9-tuple. Each entry is either:
#   - None  -> put 0 in this position
#   - k     -> copy original bit number k
# ============================================================

N = 9
ZERO = None


def bits_to_int(bits: Tuple[int, ...]) -> int:
    """Convert a tuple of 9 bits to an integer."""
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def int_to_bits(i: int, n: int = N) -> Tuple[int, ...]:
    """Convert integer to an n-bit tuple."""
    return tuple((i >> (n - 1 - k)) & 1 for k in range(n))


def apply_pattern(bits: Tuple[int, ...], pattern: Tuple[Optional[int], ...]) -> Tuple[int, ...]:
    """Apply a pattern to the original bits."""
    out = []
    for p in pattern:
        out.append(0 if p is None else bits[p])
    return tuple(out)


@dataclass(frozen=True)
class Equation:
    """
    Represents an equation:
        sum(coeffs[i] * F[i] for i in coeffs) + const = 0
    where F[i] are unknown Boolean values in {0,1}.
    """
    coeffs: Tuple[Tuple[int, int], ...]   # sorted (index, coefficient)
    const: int

    def support_size(self) -> int:
        return len(self.coeffs)

    def to_sparse_string(self) -> str:
        if not self.coeffs:
            return ""
        else:
            parts = []
            for idx, coeff in self.coeffs:
                if coeff == 1:
                    parts.append(f"x_{idx}")
                elif coeff == -1:
                    parts.append(f"-x_{idx}")
                else:
                    parts.append(f"{coeff}*x_{idx}")
            lhs = " + ".join(parts).replace("+ -", "- ")

        return f"{lhs} = {-1*int(self.const)}"
    

def canonical_equation(coeffs: Dict[int, int], const: int) -> Equation:
    """
    Remove zero coefficients, sort terms, normalize sign so the first
    nonzero coefficient is positive. This gives a canonical form for deduplication.
    """
    filtered = {i: a for i, a in coeffs.items() if a != 0}

    if not filtered:
        return Equation(tuple(), const)

    items = tuple(sorted(filtered.items()))
    if items[0][1] < 0:
        items = tuple((i, -a) for i, a in items)
        const = -const

    return Equation(items, const)


def substitute_fixed_values(eq: Equation, fixed: Dict[int, int]) -> Equation:
    """
    Substitute fixed values F[i] = value into the equation.
    """
    coeffs = dict(eq.coeffs)
    const = eq.const

    for idx, value in fixed.items():
        if idx in coeffs:
            const += coeffs[idx] * value
            del coeffs[idx]

    return canonical_equation(coeffs, const)


# ============================================================
# Corrected functional identity:
#
# f(x,y,z,t,u,w,a,b,c)
#    = x
#    + f(0,y,z,0,u,w,0,b,c)
#    + f(0,0,y,0,0,u,0,0,b)
#    + f(0,0,0,0,y,z,0,u,w)
#    + f(0,0,0,0,0,y,0,0,u)
#    + f(0,0,0,0,0,0,0,y,z)
#    + f(0,0,0,0,0,0,0,0,y)
#    + f(0,0,0,t,u,w,a,b,c)
#    + f(0,0,0,0,t,u,0,a,b)
#    + f(0,0,0,0,0,t,0,0,a)
#    + f(0,0,0,0,0,0,t,u,w)
#    + f(0,0,0,0,0,0,0,t,u)
#    + f(0,0,0,0,0,0,0,0,t)
#    - f(0,x,y,0,t,u,0,a,b)
#    - f(0,0,x,0,0,t,0,0,a)
#    - f(0,0,0,x,y,z,t,u,w)
#    - f(0,0,0,0,x,y,0,t,u)
#    - f(0,0,0,0,0,x,0,0,t)
#    - f(0,0,0,0,0,0,x,y,z)
#    - f(0,0,0,0,0,0,0,x,y)
#    - f(0,0,0,0,0,0,0,0,x)
#    - f(0,0,0,0,u,w,0,b,c)
#    - f(0,0,0,0,0,u,0,0,b)
#    - f(0,0,0,0,0,0,0,u,w)
#    - f(0,0,0,0,0,0,0,0,u)
#
# After moving everything to the LHS:
#
# f(orig)
# - [all positive f-terms]
# + [all negative f-terms]
# - x
# = 0
# ============================================================

POSITIVE_TERMS: List[Tuple[Optional[int], ...]] = [
    # f(0,y,z,0,u,w,0,b,c)
    (ZERO, 1, 2, ZERO, 4, 5, ZERO, 7, 8),

    # f(0,0,y,0,0,u,0,0,b)
    (ZERO, ZERO, 1, ZERO, ZERO, 4, ZERO, ZERO, 7),

    # f(0,0,0,0,y,z,0,u,w)
    (ZERO, ZERO, ZERO, ZERO, 1, 2, ZERO, 4, 5),

    # f(0,0,0,0,0,y,0,0,u)
    (ZERO, ZERO, ZERO, ZERO, ZERO, 1, ZERO, ZERO, 4),

    # f(0,0,0,0,0,0,0,y,z)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 1, 2),

    # f(0,0,0,0,0,0,0,0,y)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 1),

    # f(0,0,0,t,u,w,a,b,c)
    (ZERO, ZERO, ZERO, 3, 4, 5, 6, 7, 8),

    # f(0,0,0,0,t,u,0,a,b)
    (ZERO, ZERO, ZERO, ZERO, 3, 4, ZERO, 6, 7),

    # f(0,0,0,0,0,t,0,0,a)
    (ZERO, ZERO, ZERO, ZERO, ZERO, 3, ZERO, ZERO, 6),

    # f(0,0,0,0,0,0,t,u,w)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 3, 4, 5),

    # f(0,0,0,0,0,0,0,t,u)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 3, 4),

    # f(0,0,0,0,0,0,0,0,t)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 3),
]

NEGATIVE_TERMS: List[Tuple[Optional[int], ...]] = [
    # f(0,x,y,0,t,u,0,a,b)
    (ZERO, 0, 1, ZERO, 3, 4, ZERO, 6, 7),

    # f(0,0,x,0,0,t,0,0,a)
    (ZERO, ZERO, 0, ZERO, ZERO, 3, ZERO, ZERO, 6),

    # f(0,0,0,x,y,z,t,u,w)
    (ZERO, ZERO, ZERO, 0, 1, 2, 3, 4, 5),

    # f(0,0,0,0,x,y,0,t,u)
    (ZERO, ZERO, ZERO, ZERO, 0, 1, ZERO, 3, 4),

    # f(0,0,0,0,0,x,0,0,t)
    (ZERO, ZERO, ZERO, ZERO, ZERO, 0, ZERO, ZERO, 3),

    # f(0,0,0,0,0,0,x,y,z)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 0, 1, 2),

    # f(0,0,0,0,0,0,0,x,y)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 0, 1),

    # f(0,0,0,0,0,0,0,0,x)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 0),

    # f(0,0,0,0,u,w,0,b,c)
    (ZERO, ZERO, ZERO, ZERO, 4, 5, ZERO, 7, 8),

    # f(0,0,0,0,0,u,0,0,b)
    (ZERO, ZERO, ZERO, ZERO, ZERO, 4, ZERO, ZERO, 7),

    # f(0,0,0,0,0,0,0,u,w)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 4, 5),

    # f(0,0,0,0,0,0,0,0,u)
    (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 4),
]


def build_equation_for_assignment(bits: Tuple[int, ...]) -> Equation:
    """
    Build one linear equation for a fixed Boolean assignment of
    (x,y,z,t,u,w,a,b,c).
    """
    coeffs: Dict[int, int] = defaultdict(int)

    # + f(orig)
    orig_idx = bits_to_int(bits)
    coeffs[orig_idx] += 1

    # - positive terms
    for pat in POSITIVE_TERMS:
        idx = bits_to_int(apply_pattern(bits, pat))
        coeffs[idx] -= 1

    # + negative terms
    for pat in NEGATIVE_TERMS:
        idx = bits_to_int(apply_pattern(bits, pat))
        coeffs[idx] += 1

    # - x
    const = -bits[0]

    return canonical_equation(coeffs, const)


def boundary_equations() -> List[Equation]:
    """
    Add:
      F[0] = 0
      F[511] = 1
    in the standard form sum(coeffs*F)+const=0.
    """
    eq0 = canonical_equation({0: 1}, 0)
    eq511 = canonical_equation({511: 1}, -1)
    return [eq0, eq511]


def deduplicate_equations(equations: Iterable[Equation]) -> Dict[Equation, int]:
    counts: Dict[Equation, int] = defaultdict(int)
    for eq in equations:
        counts[eq] += 1
    return dict(counts)


def generate_all_equations(substitute_boundaries: bool = True) -> Dict[Equation, int]:
    """
    Enumerate all 512 assignments, simplify, optionally substitute
    F[0]=0 and F[511]=1, then deduplicate.
    """
    equations: List[Equation] = []

    for bits in product([0, 1], repeat=N):
        eq = build_equation_for_assignment(bits)
        equations.append(eq)

    equations.extend(boundary_equations())

    if substitute_boundaries:
        fixed = {0: 0, 511: 1}
        equations = [substitute_fixed_values(eq, fixed) for eq in equations]

    return deduplicate_equations(equations)


def equations_to_matrix(eq_counts: Dict[Equation, int], size: int = 512) -> Tuple[List[List[int]], List[int], List[int], List[Equation]]:
    """
    Convert distinct equations to A * F = b.
    Returns:
      A              : list of rows
      b              : RHS vector
      multiplicities : how many original equations collapsed to this one
      equations      : corresponding Equation objects
    """
    equations = sorted(
        eq_counts.keys(),
        key=lambda eq: (eq.support_size(), eq.const, eq.coeffs)
    )

    A = []
    b = []
    multiplicities = []

    for eq in equations:
        row = [0] * size
        for idx, coeff in eq.coeffs:
            row[idx] = coeff
        A.append(row)
        b.append(-eq.const)
        multiplicities.append(eq_counts[eq])

    return A, b, multiplicities, equations


def save_sparse_equations(filename: str, eq_counts: Dict[Equation, int]) -> None:
    items = sorted(
        eq_counts.items(),
        key=lambda kv: (kv[0].support_size(), kv[0].const, kv[0].coeffs)
    )

    with open(filename, "w", encoding="utf-8") as f:
        #f.write(f"Distinct equations: {len(eq_counts)}\n\n")
        for k, (eq, count) in enumerate(items, start=1):
            #f.write(f"[{k}] multiplicity={count}\n")
            tmp = eq.to_sparse_string()
            if len(tmp)>0:
                f.write(tmp + "\n")


def save_dense_matrix_csv(filename: str, A: List[List[int]], b: List[int], multiplicities: List[int]) -> None:
    """
    Save matrix in CSV:
      mult, a_0, a_1, ..., a_511, rhs
    """
    with open(filename, "w", encoding="utf-8") as f:
        header = ["mult"] + [f"a_{i}" for i in range(512)] + ["rhs"]
        f.write(",".join(header) + "\n")

        for row, rhs, mult in zip(A, b, multiplicities):
            line = [str(mult)] + [str(v) for v in row] + [str(rhs)]
            f.write(",".join(line) + "\n")


def save_assignment_equations(filename: str) -> None:
    """
    Save all 512 raw equations before deduplication, one per assignment.
    Useful for debugging and checking the construction.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for bits in product([0, 1], repeat=N):
            idx = bits_to_int(bits)
            eq = build_equation_for_assignment(bits)
            f.write(f"assignment={bits}  i={idx}\n")
            f.write(eq.to_sparse_string() + "\n\n")


def try_sympy_rank(A: List[List[int]], b: List[int]) -> None:
    """
    Optional: compute ranks over Q using sympy.
    """
    try:
        import sympy as sp
    except ImportError:
        print("sympy not installed; skipping rank computation.")
        return

    M = sp.Matrix(A)
    rhs = sp.Matrix(b)

    rank_A = M.rank()
    rank_aug = M.row_join(rhs).rank()

    print()
    print("Sympy rank analysis over Q:")
    print(f"  rank(A)     = {rank_A}")
    print(f"  rank([A|b]) = {rank_aug}")
    if rank_A == rank_aug:
        print(f"  affine solution-space dimension over Q: {512 - rank_A}")
    else:
        print("  The linear system over Q is inconsistent.")


def main() -> None:
    eq_counts = generate_all_equations(substitute_boundaries=True)

    print(f"Number of distinct simplified equations: {len(eq_counts)}")
    print()

    items = sorted(
        eq_counts.items(),
        key=lambda kv: (kv[0].support_size(), kv[0].const, kv[0].coeffs)
    )

    print("First 40 distinct equations:")
    print("---------------------------")
    for k, (eq, count) in enumerate(items[:40], start=1):
        print(f"[{k}] multiplicity={count:3d}   {eq.to_sparse_string()}")

    A, b, multiplicities, equations = equations_to_matrix(eq_counts)

    save_sparse_equations("simplified_equations.txt", eq_counts)
    save_dense_matrix_csv("simplified_equations_matrix.csv", A, b, multiplicities)
    save_assignment_equations("all_512_raw_equations.txt")

    print()
    print("Saved files:")
    print("  simplified_equations.txt")
    print("  simplified_equations_matrix.csv")
    print("  all_512_raw_equations.txt")

    try_sympy_rank(A, b)


if __name__ == "__main__":
    main()