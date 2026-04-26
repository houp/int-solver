from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ComplementedVariable:
    name: str


ArgAtom = int | str | ComplementedVariable
DEFAULT_SPEC_PATH = (
    Path(__file__).resolve().parent / "identities" / "number_conserving.func"
)


def bits_to_int(bits: Tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


@dataclass(frozen=True)
class FunctionCall:
    args: Tuple[ArgAtom, ...]

    def variables(self) -> set[str]:
        used: set[str] = set()
        for arg in self.args:
            if isinstance(arg, str):
                used.add(arg)
            elif isinstance(arg, ComplementedVariable):
                used.add(arg.name)
        return used


@dataclass
class LinearExpression:
    const: int = 0
    variable_terms: Dict[str, int] = field(default_factory=dict)
    function_terms: Dict[FunctionCall, int] = field(default_factory=dict)

    def copy(self) -> "LinearExpression":
        return LinearExpression(
            const=self.const,
            variable_terms=dict(self.variable_terms),
            function_terms=dict(self.function_terms),
        )

    def is_constant(self) -> bool:
        return not self.variable_terms and not self.function_terms

    def scale(self, factor: int) -> "LinearExpression":
        scaled = LinearExpression(const=self.const * factor)
        for name, coeff in self.variable_terms.items():
            new_coeff = coeff * factor
            if new_coeff:
                scaled.variable_terms[name] = new_coeff
        for call, coeff in self.function_terms.items():
            new_coeff = coeff * factor
            if new_coeff:
                scaled.function_terms[call] = new_coeff
        return scaled

    def add(self, other: "LinearExpression", factor: int = 1) -> "LinearExpression":
        merged = self.copy()
        merged.const += factor * other.const
        for name, coeff in other.variable_terms.items():
            merged.variable_terms[name] = merged.variable_terms.get(name, 0) + factor * coeff
            if merged.variable_terms[name] == 0:
                del merged.variable_terms[name]
        for call, coeff in other.function_terms.items():
            merged.function_terms[call] = merged.function_terms.get(call, 0) + factor * coeff
            if merged.function_terms[call] == 0:
                del merged.function_terms[call]
        return merged

    def variables(self) -> set[str]:
        used = set(self.variable_terms)
        for call in self.function_terms:
            used.update(call.variables())
        return used


@dataclass(frozen=True)
class Equation:
    coeffs: Tuple[Tuple[int, int], ...]
    const: int

    def support_size(self) -> int:
        return len(self.coeffs)

    def to_sparse_string(self) -> str:
        if not self.coeffs:
            return ""

        parts: List[str] = []
        for idx, coeff in self.coeffs:
            if coeff == 1:
                parts.append(f"x_{idx}")
            elif coeff == -1:
                parts.append(f"-x_{idx}")
            else:
                parts.append(f"{coeff}*x_{idx}")
        lhs = " + ".join(parts).replace("+ -", "- ")
        return f"{lhs} = {-self.const}"


@dataclass(frozen=True)
class Implication:
    lhs: int
    rhs: int

    def to_sparse_string(self) -> str:
        return f"x_{self.lhs} <= x_{self.rhs}"


Constraint = Equation | Implication


@dataclass(frozen=True)
class IdentitySpec:
    statement: str
    free_variables: Tuple[str, ...]
    relation: str
    lhs: LinearExpression
    rhs: LinearExpression


@dataclass(frozen=True)
class FunctionalSystemSpec:
    variable_order: Tuple[str, ...]
    identities: Tuple[IdentitySpec, ...]

    @property
    def truth_table_size(self) -> int:
        return 1 << len(self.variable_order)


@dataclass(frozen=True)
class InstantiatedEquation:
    identity_index: int
    statement: str
    free_variables: Tuple[str, ...]
    assignment: Tuple[int, ...]
    constraint: Constraint


def canonical_equation(coeffs: Dict[int, int], const: int) -> Equation:
    filtered = {idx: coeff for idx, coeff in coeffs.items() if coeff != 0}
    if not filtered:
        return Equation(tuple(), const)

    items = tuple(sorted(filtered.items()))
    if items[0][1] < 0:
        items = tuple((idx, -coeff) for idx, coeff in items)
        const = -const
    return Equation(items, const)


def substitute_fixed_values(eq: Equation, fixed: Dict[int, int]) -> Equation:
    coeffs = dict(eq.coeffs)
    const = eq.const
    for idx, value in fixed.items():
        if idx in coeffs:
            const += coeffs[idx] * value
            del coeffs[idx]
    return canonical_equation(coeffs, const)


def substitute_constraint_fixed_values(
    constraint: Constraint,
    fixed: Dict[int, int],
) -> Constraint | None:
    if isinstance(constraint, Equation):
        return substitute_fixed_values(constraint, fixed)

    lhs_fixed = fixed.get(constraint.lhs)
    rhs_fixed = fixed.get(constraint.rhs)
    if lhs_fixed is not None and rhs_fixed is not None:
        if lhs_fixed <= rhs_fixed:
            return None
        return Equation(tuple(), 1)
    if lhs_fixed is not None:
        return None if lhs_fixed == 0 else canonical_equation({constraint.rhs: 1}, -1)
    if rhs_fixed is not None:
        return None if rhs_fixed == 1 else canonical_equation({constraint.lhs: 1}, 0)
    return constraint


def deduplicate_constraints(constraints: Iterable[Constraint]) -> Dict[Constraint, int]:
    counts: DefaultDict[Constraint, int] = defaultdict(int)
    for constraint in constraints:
        counts[constraint] += 1
    return dict(counts)


def _normalize_inline_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _strip_comments(text: str) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        before_comment = line.split("#", 1)[0]
        lines.append(before_comment.rstrip())
    return "\n".join(lines)


def _parse_variable_order(line: str) -> Tuple[str, ...]:
    tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
    if not tokens:
        raise ValueError("Variable order cannot be empty")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Variable order contains duplicates")
    for token in tokens:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            raise ValueError(f"Invalid variable name in vars section: {token!r}")
    return tuple(tokens)


def _split_statements(text: str) -> List[str]:
    statements: List[str] = []
    chunk: List[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unmatched closing parenthesis in identity file")
        if char == ";" and depth == 0:
            statement = "".join(chunk).strip()
            if statement:
                statements.append(statement)
            chunk = []
            continue
        chunk.append(char)

    tail = "".join(chunk).strip()
    if tail:
        raise ValueError(
            "Identity file ended without ';'. Terminate each identity with a semicolon."
        )
    if depth != 0:
        raise ValueError("Unbalanced parentheses in identity file")
    return statements


def _parse_call_arg(node: ast.AST, variable_order: Sequence[str]) -> ArgAtom:
    if isinstance(node, ast.Name):
        if node.id not in variable_order:
            raise ValueError(f"Unknown variable in f(...) argument: {node.id}")
        return node.id
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, int)
        and node.left.value == 1
        and isinstance(node.right, ast.Name)
    ):
        if node.right.id not in variable_order:
            raise ValueError(f"Unknown variable in f(...) argument: {node.right.id}")
        return ComplementedVariable(node.right.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if node.value not in (0, 1):
            raise ValueError("Only 0 and 1 are allowed as literal f(...) arguments")
        return int(node.value)
    raise ValueError(
        "Arguments to f(...) must be variables, the literals 0/1, or complements like 1 - x"
    )


def _expression_from_ast(node: ast.AST, variable_order: Sequence[str]) -> LinearExpression:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return LinearExpression(const=int(node.value))

    if isinstance(node, ast.Name):
        if node.id not in variable_order:
            raise ValueError(f"Unknown variable: {node.id}")
        return LinearExpression(variable_terms={node.id: 1})

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "f":
            raise ValueError("Only a single function named f(...) is supported")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in f(...)")
        if len(node.args) != len(variable_order):
            raise ValueError(
                f"f(...) must have exactly {len(variable_order)} arguments, "
                f"got {len(node.args)}"
            )
        call = FunctionCall(tuple(_parse_call_arg(arg, variable_order) for arg in node.args))
        return LinearExpression(function_terms={call: 1})

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        expr = _expression_from_ast(node.operand, variable_order)
        return expr if isinstance(node.op, ast.UAdd) else expr.scale(-1)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _expression_from_ast(node.left, variable_order)
        right = _expression_from_ast(node.right, variable_order)
        return left.add(right)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        left = _expression_from_ast(node.left, variable_order)
        right = _expression_from_ast(node.right, variable_order)
        return left.add(right, factor=-1)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _expression_from_ast(node.left, variable_order)
        right = _expression_from_ast(node.right, variable_order)
        if left.is_constant():
            return right.scale(left.const)
        if right.is_constant():
            return left.scale(right.const)
        raise ValueError("Only scalar multiplication is supported")

    raise ValueError(f"Unsupported syntax in identity expression: {ast.dump(node)}")


def _parse_linear_expression(expr_text: str, variable_order: Sequence[str]) -> LinearExpression:
    normalized = _normalize_inline_whitespace(expr_text)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid identity expression: {normalized}") from exc
    return _expression_from_ast(tree.body, variable_order)


def _parse_relation(statement: str) -> Tuple[str, str, str]:
    for operator in ("!=", "<=", ">=", "="):
        if operator in statement:
            left_text, right_text = statement.split(operator, 1)
            return operator, left_text.strip(), right_text.strip()
    raise ValueError(f"Identity is missing a supported relation operator: {statement}")


def _parse_identity(statement: str, variable_order: Sequence[str]) -> IdentitySpec:
    relation, left_text, right_text = _parse_relation(statement)
    lhs = _parse_linear_expression(left_text, variable_order)
    rhs = _parse_linear_expression(right_text, variable_order)

    if relation in ("!=", "<=", ">="):
        for side_name, expr in (("left", lhs), ("right", rhs)):
            if expr.variable_terms:
                raise ValueError(
                    f"{relation} constraints only support Boolean atoms on the {side_name}-hand side"
                )
            if expr.const not in (0, 1) and not expr.function_terms:
                raise ValueError(
                    f"{relation} constraints only support literal 0/1 or one f(...) call on the {side_name}-hand side"
                )
            if expr.function_terms:
                if expr.const != 0 or len(expr.function_terms) != 1:
                    raise ValueError(
                        f"{relation} constraints only support literal 0/1 or one f(...) call on the {side_name}-hand side"
                    )
                (_, coeff), = expr.function_terms.items()
                if coeff != 1:
                    raise ValueError(
                        f"{relation} constraints only support literal 0/1 or one f(...) call on the {side_name}-hand side"
                    )
    else:
        if lhs.const != 0 or lhs.variable_terms or len(lhs.function_terms) != 1:
            raise ValueError(
                "Left-hand side must be exactly one f(...) call with coefficient 1"
            )
        (_, lhs_coeff), = lhs.function_terms.items()
        if lhs_coeff != 1:
            raise ValueError(
                "Left-hand side must be exactly one f(...) call with coefficient 1"
            )

    free_variables = tuple(
        name for name in variable_order if name in lhs.variables() | rhs.variables()
    )
    normalized_statement = (
        f"{_normalize_inline_whitespace(left_text)} {relation} {_normalize_inline_whitespace(right_text)}"
    )
    return IdentitySpec(
        statement=normalized_statement,
        free_variables=free_variables,
        relation=relation,
        lhs=lhs,
        rhs=rhs,
    )


def load_spec(path: Path) -> FunctionalSystemSpec:
    cleaned = _strip_comments(path.read_text(encoding="utf-8"))
    body_lines: List[str] = []
    variable_order: Tuple[str, ...] | None = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            body_lines.append("")
            continue
        match = re.match(r"^(vars|variables)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            if variable_order is not None:
                raise ValueError("Identity file may define vars: only once")
            variable_order = _parse_variable_order(match.group(2))
            continue
        body_lines.append(raw_line)

    if variable_order is None:
        raise ValueError("Identity file must define variable order via 'vars:'")

    statements = _split_statements("\n".join(body_lines))
    if not statements:
        raise ValueError("Identity file does not contain any identities")

    identities = tuple(_parse_identity(statement, variable_order) for statement in statements)
    return FunctionalSystemSpec(variable_order=variable_order, identities=identities)


def _evaluate_linear_expression(expr: LinearExpression, env: Dict[str, int]) -> Tuple[DefaultDict[int, int], int]:
    coeffs: DefaultDict[int, int] = defaultdict(int)
    const = expr.const

    for name, coeff in expr.variable_terms.items():
        const += coeff * env[name]

    for call, coeff in expr.function_terms.items():
        bits = []
        for arg in call.args:
            if isinstance(arg, int):
                bits.append(arg)
            elif isinstance(arg, str):
                bits.append(env[arg])
            else:
                bits.append(1 - env[arg.name])
        coeffs[bits_to_int(tuple(bits))] += coeff

    return coeffs, const


def instantiate_constraint(identity: IdentitySpec, env: Dict[str, int]) -> Constraint | None:
    if identity.relation == "=":
        lhs_coeffs, lhs_const = _evaluate_linear_expression(identity.lhs, env)
        rhs_coeffs, rhs_const = _evaluate_linear_expression(identity.rhs, env)
        coeffs: DefaultDict[int, int] = defaultdict(int)
        for idx, coeff in lhs_coeffs.items():
            coeffs[idx] += coeff
        for idx, coeff in rhs_coeffs.items():
            coeffs[idx] -= coeff
        return canonical_equation(dict(coeffs), lhs_const - rhs_const)

    lhs_coeffs, lhs_const = _evaluate_linear_expression(identity.lhs, env)
    rhs_coeffs, rhs_const = _evaluate_linear_expression(identity.rhs, env)

    if lhs_coeffs:
        (lhs_idx, lhs_coeff), = lhs_coeffs.items()
        if lhs_coeff != 1 or lhs_const != 0:
            raise ValueError(f"{identity.relation} constraints require Boolean atoms on the left-hand side")
        lhs_atom = ("var", lhs_idx)
    else:
        lhs_atom = ("const", lhs_const)

    if rhs_coeffs:
        (rhs_idx, rhs_coeff), = rhs_coeffs.items()
        if rhs_coeff != 1 or rhs_const != 0:
            raise ValueError(f"{identity.relation} constraints require Boolean atoms on the right-hand side")
        rhs_atom = ("var", rhs_idx)
    else:
        rhs_atom = ("const", rhs_const)

    relation = identity.relation
    if relation == ">=":
        lhs_atom, rhs_atom = rhs_atom, lhs_atom
        relation = "<="

    if relation == "!=":
        if lhs_atom[0] == "const" and rhs_atom[0] == "const":
            if lhs_atom[1] == rhs_atom[1]:
                return Equation(tuple(), 1)
            return None
        if lhs_atom[0] == "const":
            return canonical_equation({rhs_atom[1]: 1}, lhs_atom[1] - 1)
        if rhs_atom[0] == "const":
            return canonical_equation({lhs_atom[1]: 1}, rhs_atom[1] - 1)
        return canonical_equation({lhs_atom[1]: 1, rhs_atom[1]: 1}, -1)

    if relation == "<=":
        if lhs_atom[0] == "const" and rhs_atom[0] == "const":
            if lhs_atom[1] <= rhs_atom[1]:
                return None
            return Equation(tuple(), 1)
        if lhs_atom[0] == "const":
            return None if lhs_atom[1] == 0 else canonical_equation({rhs_atom[1]: 1}, -1)
        if rhs_atom[0] == "const":
            return None if rhs_atom[1] == 1 else canonical_equation({lhs_atom[1]: 1}, 0)
        return Implication(lhs=lhs_atom[1], rhs=rhs_atom[1])

    raise ValueError(f"Unsupported relation: {identity.relation}")


def instantiate_identity(
    identity_index: int,
    identity: IdentitySpec,
    variable_order: Sequence[str],
) -> List[InstantiatedEquation]:
    instances: List[InstantiatedEquation] = []
    for assignment in product([0, 1], repeat=len(identity.free_variables)):
        env = dict(zip(identity.free_variables, assignment))
        constraint = instantiate_constraint(identity, env)
        if constraint is None:
            continue
        instances.append(
            InstantiatedEquation(
                identity_index=identity_index,
                statement=identity.statement,
                free_variables=identity.free_variables,
                assignment=assignment,
                constraint=constraint,
            )
        )
    return instances


def instantiate_all_identities(spec: FunctionalSystemSpec) -> List[InstantiatedEquation]:
    all_instances: List[InstantiatedEquation] = []
    for index, identity in enumerate(spec.identities, start=1):
        all_instances.extend(instantiate_identity(index, identity, spec.variable_order))
    return all_instances


def extract_fixed_values(constraints: Iterable[Constraint]) -> Dict[int, int]:
    fixed_values: Dict[int, int] = {}
    for equation in constraints:
        if not isinstance(equation, Equation):
            continue
        if len(equation.coeffs) != 1:
            continue
        idx, coeff = equation.coeffs[0]
        if coeff != 1:
            continue
        value = -equation.const
        if value not in (0, 1):
            continue
        previous = fixed_values.get(idx)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting fixed values for x_{idx}: {previous} vs {value}")
        fixed_values[idx] = value
    return fixed_values


def constraints_to_matrix(
    eq_counts: Dict[Constraint, int],
    size: int,
) -> Tuple[List[List[int]], List[int], List[int], List[str], List[Constraint]]:
    equations = sorted(
        eq_counts.keys(),
        key=lambda equation: (
            0 if isinstance(equation, Equation) else 1,
            equation.support_size() if isinstance(equation, Equation) else 2,
            equation.const if isinstance(equation, Equation) else 0,
            equation.coeffs if isinstance(equation, Equation) else ((equation.lhs, equation.rhs),),
        ),
    )

    rows: List[List[int]] = []
    rhs: List[int] = []
    multiplicities: List[int] = []
    senses: List[str] = []
    for equation in equations:
        row = [0] * size
        if isinstance(equation, Equation):
            for idx, coeff in equation.coeffs:
                row[idx] = coeff
            rhs_value = -equation.const
            sense = "="
        else:
            row[equation.lhs] = 1
            row[equation.rhs] = -1
            rhs_value = 0
            sense = "<="
        rows.append(row)
        rhs.append(rhs_value)
        multiplicities.append(eq_counts[equation])
        senses.append(sense)
    return rows, rhs, multiplicities, senses, equations


def save_sparse_equations(path: Path, eq_counts: Dict[Constraint, int]) -> None:
    items = sorted(
        eq_counts.items(),
        key=lambda item: (
            0 if isinstance(item[0], Equation) else 1,
            item[0].support_size() if isinstance(item[0], Equation) else 2,
            item[0].const if isinstance(item[0], Equation) else 0,
            item[0].coeffs if isinstance(item[0], Equation) else ((item[0].lhs, item[0].rhs),),
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for equation, _ in items:
            line = equation.to_sparse_string()
            if line:
                handle.write(line + "\n")


def save_dense_matrix_csv(
    path: Path,
    matrix: List[List[int]],
    rhs: List[int],
    multiplicities: List[int],
    senses: List[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        width = len(matrix[0]) if matrix else 0
        header = ["mult", "sense", *[f"a_{idx}" for idx in range(width)], "rhs"]
        handle.write(",".join(header) + "\n")
        for row, row_rhs, mult, sense in zip(matrix, rhs, multiplicities, senses):
            handle.write(",".join([str(mult), sense, *[str(value) for value in row], str(row_rhs)]) + "\n")


def save_instantiated_equations(path: Path, instances: Sequence[InstantiatedEquation]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for instance in instances:
            assignment_pairs = ", ".join(
                f"{name}={value}" for name, value in zip(instance.free_variables, instance.assignment)
            )
            handle.write(
                f"identity={instance.identity_index} "
                f"vars=({', '.join(instance.free_variables)}) "
                f"assignment=({assignment_pairs})\n"
            )
            handle.write(instance.statement + "\n")
            handle.write(instance.constraint.to_sparse_string() + "\n\n")


def try_sympy_rank(matrix: List[List[int]], rhs: List[int], size: int) -> None:
    try:
        import sympy as sp
    except ImportError:
        print("sympy not installed; skipping rank computation.")
        return

    mat = sp.Matrix(matrix)
    rhs_mat = sp.Matrix(rhs)
    rank_a = mat.rank()
    rank_aug = mat.row_join(rhs_mat).rank()

    print()
    print("Sympy rank analysis over Q:")
    print(f"  rank(A)     = {rank_a}")
    print(f"  rank([A|b]) = {rank_aug}")
    if rank_a == rank_aug:
        print(f"  affine solution-space dimension over Q: {size - rank_a}")
    else:
        print("  The linear system over Q is inconsistent.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sparse algebraic equation system from one or more "
            "functional identities written with f(...)."
        )
    )
    parser.add_argument(
        "identity_file",
        nargs="?",
        default=str(DEFAULT_SPEC_PATH),
        help=(
            "Path to the identity specification file. "
            f"Defaults to {DEFAULT_SPEC_PATH}."
        ),
    )
    parser.add_argument(
        "--sparse-out",
        default="simplified_equations.txt",
        help="Output path for the deduplicated sparse equations.",
    )
    parser.add_argument(
        "--matrix-out",
        default="simplified_equations_matrix.csv",
        help="Output path for the dense CSV matrix export.",
    )
    parser.add_argument(
        "--raw-out",
        default="all_512_raw_equations.txt",
        help="Output path for instantiated equations before deduplication.",
    )
    parser.add_argument(
        "--no-substitute-fixed",
        action="store_true",
        help="Keep fixed-value equations like x_k = 0/1 instead of substituting them.",
    )
    parser.add_argument(
        "--skip-rank",
        action="store_true",
        help="Skip the optional Sympy rank analysis.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=40,
        help="How many deduplicated equations to print to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    spec = load_spec(Path(args.identity_file))
    instantiated = instantiate_all_identities(spec)

    constraints: List[Constraint] = [instance.constraint for instance in instantiated]
    fixed_values = extract_fixed_values(constraints)

    if not args.no_substitute_fixed and fixed_values:
        substituted: List[Constraint] = []
        for constraint in constraints:
            rewritten = substitute_constraint_fixed_values(constraint, fixed_values)
            if rewritten is not None:
                substituted.append(rewritten)
        constraints = substituted

    eq_counts = deduplicate_constraints(constraints)
    matrix, rhs, multiplicities, senses, ordered_equations = constraints_to_matrix(
        eq_counts, spec.truth_table_size
    )

    sparse_out = Path(args.sparse_out)
    matrix_out = Path(args.matrix_out)
    raw_out = Path(args.raw_out)

    save_sparse_equations(sparse_out, eq_counts)
    save_dense_matrix_csv(matrix_out, matrix, rhs, multiplicities, senses)
    save_instantiated_equations(raw_out, instantiated)

    print(
        "Loaded "
        f"{len(spec.identities)} identities over variables ({', '.join(spec.variable_order)})"
    )
    print(f"Instantiated constraints before deduplication: {len(instantiated)}")
    print(f"Fixed truth-table entries detected: {len(fixed_values)}")
    print(f"Number of distinct simplified constraints: {len(eq_counts)}")
    print()

    print(f"First {min(args.print_limit, len(ordered_equations))} distinct constraints:")
    print("---------------------------")
    items = sorted(
        eq_counts.items(),
        key=lambda item: (
            0 if isinstance(item[0], Equation) else 1,
            item[0].support_size() if isinstance(item[0], Equation) else 2,
            item[0].const if isinstance(item[0], Equation) else 0,
            item[0].coeffs if isinstance(item[0], Equation) else ((item[0].lhs, item[0].rhs),),
        ),
    )
    for index, (equation, count) in enumerate(items[: args.print_limit], start=1):
        print(f"[{index}] multiplicity={count:3d}   {equation.to_sparse_string()}")

    print()
    print("Saved files:")
    print(f"  {sparse_out}")
    print(f"  {matrix_out}")
    print(f"  {raw_out}")

    if not args.skip_rank and all(sense == "=" for sense in senses):
        try_sympy_rank(matrix, rhs, spec.truth_table_size)
    elif not args.skip_rank:
        print("sympy rank analysis skipped because the instantiated system contains non-equality constraints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
