from abc import ABC, abstractmethod
import ast
import asyncio
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable, Literal, NamedTuple, override
from collections.abc import Sequence
from collections.abc import Iterable
import contextlib
import io
import multiprocessing
import queue
import subprocess
import tempfile
import time
import os
import traceback


def ast_replace(node: ast.AST, **changes) -> ast.AST:
    node = copy(node)
    for field, value in changes.items():
        setattr(node, field, value)
    return node


def parse_stmts(s: str) -> list[ast.stmt]:
    mod = ast.parse(s)
    return mod.body


def parse_expr(s: str) -> ast.expr:
    mod = ast.parse(s, mode="eval")
    return mod.body


class ASTRewrite(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        raise NotImplementedError

    @abstractmethod
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> "ast.AST | list[ast.AST] | None":
        raise NotImplementedError


@dataclass
class GeneralRemove(ASTRewrite):
    name: str
    target_type: type[ast.AST]
    inside_list: bool = True
    replace_with: "ast.AST | list[ast.AST] | None" = None

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return isinstance(node, self.target_type) and (not self.inside_list or inside_list)

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> None:
        return deepcopy(self.replace_with)


def expr_to_zeros(target: ast.expr) -> ast.expr:
    if isinstance(target, ast.Tuple):
        zeros = [ast.Constant(value=0) for _ in target.elts]
        return ast.Tuple(elts=zeros, ctx=ast.Load())
    else:
        return ast.Constant(value=0)


class CallFwdArg1(ASTRewrite):
    @override
    def get_name(self) -> str:
        return "call-fwd-arg1"

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return isinstance(node, ast.Call) and len(node.args) >= 1

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> ast.AST:
        assert isinstance(node, ast.Call)
        return node.args[0]


class AttachFullFuncArgs(ASTRewrite):
    @override
    def get_name(self) -> str:
        return "attach-full-func-args"

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return isinstance(node, ast.FunctionDef) and (node.args.vararg is None or node.args.kwarg is None)

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> ast.AST:
        assert isinstance(node, ast.FunctionDef)
        node = copy(node)
        node.args = copy(node.args)
        if node.args.vararg is None:
            node.args.vararg = ast.arg(arg="args")
        if node.args.kwarg is None:
            node.args.kwarg = ast.arg(arg="kwargs")
        return node


@dataclass
class IntConstApply(ASTRewrite):
    matcher: Callable[[int], bool]
    apply: Callable[[int], ast.AST]
    name: str

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, int) and self.matcher(node.value)

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> ast.AST:
        assert isinstance(node, ast.Constant) and isinstance(node.value, int)
        return ast_replace(node, value=self.apply(node.value))


@dataclass
class BinOpFwdArg(ASTRewrite):
    forward: Literal["left", "right"] = "left"

    @override
    def get_name(self) -> str:
        return f"binop-fwd-arg-{self.forward}"

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return isinstance(node, ast.BinOp)

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> ast.AST:
        assert isinstance(node, ast.BinOp)
        if self.forward == "left":
            return node.left
        else:
            return node.right


def _as_expr_placeholder(temp: ast.AST) -> "str | None":
    if isinstance(temp, ast.Name):
        return temp.id
    else:
        return None


def _as_stmt_placeholder(temp: ast.AST) -> "str | None":
    if isinstance(temp, ast.Expr) and isinstance(temp.value, ast.Name):
        return temp.value.id
    else:
        return None


def _ast_match(temp: ast.AST, node: ast.expr, placeholders: set[str]):
    ph_expr = _as_expr_placeholder(temp)
    if ph_expr is not None and ph_expr in placeholders:
        return {ph_expr: node}
    if type(temp) is not type(node):
        return False
    result = {}
    for field, value in ast.iter_fields(temp):
        if isinstance(value, list):
            if len(value) == 1:
                ph_stmts = _as_stmt_placeholder(value[0])
                if ph_stmts is not None and ph_stmts in placeholders:
                    result.update({ph_stmts: getattr(node, field)})
                    continue
            if not isinstance(getattr(node, field), list):
                return False
            if len(value) != len(getattr(node, field)):
                return False
            for v1, v2 in zip(value, getattr(node, field)):
                sub_result = _ast_match(v1, v2, placeholders)
                if sub_result is False:
                    return False
                result.update(sub_result)
        elif isinstance(value, ast.AST):
            if not isinstance(getattr(node, field), ast.AST):
                return False
            sub_result = _ast_match(value, getattr(node, field), placeholders)
            if sub_result is False:
                return False
            result.update(sub_result)
        else:
            if value != getattr(node, field):
                return False
    return result


def _ast_replace(temp: ast.expr, repl: dict[str, ast.AST]) -> ast.expr:
    ph_expr = _as_expr_placeholder(temp)
    if ph_expr is not None and ph_expr in repl:
        return deepcopy(repl[ph_expr])
    ph_stmts = _as_stmt_placeholder(temp)
    if ph_stmts is not None and ph_stmts in repl:
        return deepcopy(repl[ph_stmts])
    temp = copy(temp)
    for field, value in ast.iter_fields(temp):
        if isinstance(value, list):
            if len(value) == 1:
                ph_stmts = _as_stmt_placeholder(value[0])
                if ph_stmts is not None and ph_stmts in repl:
                    setattr(temp, field, deepcopy(repl[ph_stmts]))
                    continue
            new_values = []
            for v in value:
                res = _ast_replace(v, repl)
                if res is None:
                    continue
                if isinstance(res, ast.AST):
                    new_values.append(res)
                else:
                    new_values.extend(res)
            setattr(temp, field, new_values)
        elif isinstance(value, ast.AST):
            setattr(temp, field, _ast_replace(value, repl))
    return temp


ASTPatKind = Literal["expr", "stmt"]


@dataclass
class ASTPat:
    tree: "ast.expr | list[ast.stmt]"
    placeholders: set[str]

    @classmethod
    def from_code(cls, kind: ASTPatKind, code: str, placeholders: set[str]) -> "ASTPat":
        if kind == "expr":
            tree = parse_expr(code)
        elif kind == "stmt":
            tree = parse_stmts(code)
            if len(tree) == 1:
                tree = tree[0]
        else:
            raise ValueError(f"Unknown AST pattern kind: {kind}")
        return cls(tree, placeholders)

    def match_placeholders(self, node: "ast.AST | list[ast.AST]") -> "dict[str, ast.AST] | bool":
        return _ast_match(self.tree, node, self.placeholders)

    def match(self, node: ast.AST) -> bool:
        return self.match_placeholders(node) is not False

    def replace(self, repl: dict[str, ast.AST]) -> ast.AST:
        if isinstance(self.tree, list):
            replaced_stmts = []
            for stmt in self.tree:
                replaced = _ast_replace(stmt, repl)
                if isinstance(replaced, ast.AST):
                    replaced_stmts.append(replaced)
                else:
                    replaced_stmts.extend(replaced)
            return replaced_stmts
        else:
            return _ast_replace(self.tree, repl)


@dataclass
class ASTPatRewrite(ASTRewrite):
    name: str
    match_pat: ASTPat
    rewrite_pat: ASTPat
    checker: "Callable[[dict[str, ast.AST]], bool] | dict[str, Callable[[ast.AST], bool]] | None" = None
    derived: "dict[str, Callable[[dict[str, ast.AST]], ast.AST]] | None" = None

    @classmethod
    def from_code(
        cls,
        name: str,
        kind: ASTPatKind,
        match: str,
        rewrite: str,
        placeholders: set[str],
        checker: "Callable[[dict[str, ast.AST]], bool] | dict[str, Callable[[ast.AST], bool]] | None" = None,
        derived: "dict[str, Callable[[dict[str, ast.AST]], ast.AST]] | None" = None,
    ) -> "ASTPatRewrite":
        match_pat = ASTPat.from_code(kind, match, placeholders)
        rewrite_pat = ASTPat.from_code(kind, rewrite, placeholders)
        return cls(name, match_pat, rewrite_pat, checker, derived)

    @override
    def get_name(self) -> str:
        return self.name

    def match_placeholders(self, node: ast.AST):
        ph = self.match_pat.match_placeholders(node)
        if ph is False:
            return False
        if self.derived is not None:
            for k, v in self.derived.items():
                ph[k] = v(ph)
        if self.checker is not None:
            if isinstance(self.checker, dict):
                for k, v in self.checker.items():
                    if k not in ph or not v(ph[k]):
                        return False
            else:
                return self.checker(ph)
        return ph

    @override
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool:
        return self.match_placeholders(node) is not False

    def _rewrite(self, node: ast.AST):
        # this function is for debugging purpose
        repl = self.match_placeholders(node)
        assert repl is not False
        replaced = self.rewrite_pat.replace(repl)
        return replaced

    @override
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> ast.AST:
        return self._rewrite(node)


class ASTMutator:
    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(copy(node)):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value, node, field, True)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value, node, field, False)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit(self, node: ast.AST, parent: "ast.AST | None", field: "str | None", inside_list: bool):
        return self.generic_visit(node)


@dataclass
class LabeledRewrite:
    label: int
    rewrite: ASTRewrite


class RewriteAttacher(ASTMutator):
    def __init__(self, rewrites: list[ASTRewrite]):
        self.rewrites = rewrites
        self.uid_counter = 0
        self.rewrite_counter = 0
        self.rewrite_names = Counter()

    @override
    def visit(self, node: ast.AST, parent: "ast.AST | None", field: "str | None", inside_list: bool):
        node = copy(node)
        node._dd_uid = self.uid_counter
        self.uid_counter += 1
        node._dd_rewrites = []
        for r in self.rewrites:
            if r.match(node, parent, field, inside_list):
                lr = LabeledRewrite(self.rewrite_counter, r)
                self.rewrite_counter += 1
                self.rewrite_names[lr.rewrite.get_name()] += 1
                node._dd_rewrites.append(lr)
        res = self.generic_visit(node)
        return res


def attach_rewrites(tree: ast.AST, rewrites: list[ASTRewrite]) -> tuple[ast.AST, int, int]:
    attacher = RewriteAttacher(rewrites)
    new_tree = attacher.visit(tree, None, None, False)
    print("Rewrites:", attacher.rewrite_names)
    return new_tree, attacher.uid_counter, attacher.rewrite_counter


class RewriteApplier(ASTMutator):
    def __init__(self, target_labels: set[int]):
        self.target_labels = target_labels
        self.applied_rewrites: set[int] = set()
        self.visited: set[int] = set()

    @override
    def visit(self, node: ast.AST, parent: "ast.AST | None", field: "str | None", inside_list: bool):
        orig_uid = getattr(node, "_dd_uid", None)
        if orig_uid in self.visited:
            return self.generic_visit(node)
        self.visited.add(orig_uid)

        node = copy(node)
        for lr in getattr(node, "_dd_rewrites", []):
            lr: LabeledRewrite
            if lr.label in self.target_labels:
                node = lr.rewrite.rewrite(node, parent, field, inside_list)
                self.applied_rewrites.add(lr.label)
                break

        if node is None:
            return None
        elif isinstance(node, ast.AST):
            # After rewriting this node, traverse its children without
            # re-applying rewrite selection logic to the node itself.
            return self.generic_visit(node)
        else:
            new_items = []
            for item in node:
                if isinstance(item, ast.AST):
                    res = self.visit(item, parent, field, inside_list)
                    if res is None:
                        continue
                    elif isinstance(res, ast.AST):
                        new_items.append(res)
                    else:
                        new_items.extend(res)
            return new_items


def apply_rewrites(tree: ast.AST, target_labels: set[int]) -> tuple[ast.AST, set[int]]:
    applier = RewriteApplier(target_labels)
    new_tree = applier.visit(deepcopy(tree), None, None, False)
    return new_tree, applier.applied_rewrites


def test_rewrite(rewrite: ASTRewrite, code: str):
    tree = ast.parse(code)
    tree, _, num_matched = attach_rewrites(tree, [rewrite])
    tree, _ = apply_rewrites(tree, set(i for i in range(num_matched)))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


@dataclass
class Task:
    source: str
    applied: list[int]
    masked: list[int]

    def with_source(self, source: str) -> "Task":
        return Task(source, self.applied, self.masked)


class PDD:
    def __init__(self, all_labels: list[int], init_proba: float = 0.93):
        self.all_labels = all_labels
        self.probas = {label: init_proba for label in all_labels}

    def apply(self, target_labels: set[int]) -> set[int]:
        return target_labels

    @staticmethod
    def _update_probas(probas: dict[int, float], task: Task, is_interesting: bool):
        if is_interesting:
            for label in task.applied:
                probas[label] = 1.0
            for label in task.masked:
                probas[label] = 0.0
        else:
            prod = 1.0
            for label in task.applied:
                if probas[label] > 0:
                    prod *= probas[label]
            denorm = 1.0 - prod
            for label in task.applied:
                p = probas[label]
                if p >= 1.0:
                    continue
                probas[label] = 1.0 - (1.0 - p) / denorm if denorm > 0.0 else 0.0

    def generator(self) -> Iterable[Task]:
        probas = deepcopy(self.probas)
        while True:
            choices = sorted(probas.items(), key=lambda x: (x[1], x[0]), reverse=True)
            selected = []
            selected_count, prod = 0.0, 1.0
            for label, p in choices:
                if p >= 1.0:
                    selected.append(label)
                    continue
                if (selected_count + 1) * prod * p > selected_count * prod:
                    selected.append(label)
                    selected_count, prod = selected_count + 1, prod * p
                else:
                    break
            applied = self.apply(set(selected))
            masked = set(selected).difference(applied)
            task = Task(source=None, applied=list(applied), masked=list(masked))
            if selected_count * prod == 0 or all(probas[label] >= 1.0 for label in applied):
                break
            yield deepcopy(task)
            self._update_probas(probas, task, is_interesting=False)

    def update(self, task: Task, is_interesting: bool):
        self._update_probas(self.probas, task, is_interesting)


class TaskManager(ABC):
    @abstractmethod
    def task_generator(self) -> Iterable[Task]: ...

    @abstractmethod
    def task_update(self, task: Task, is_interesting: bool): ...

    @classmethod
    @abstractmethod
    def from_source(cls, source: str, *args, **kwargs) -> "TaskManager": ...


class ASTPDD(TaskManager, PDD):
    def __init__(self, tree: ast.AST, rewrites: list[ASTRewrite], init_proba: float = 0.93):
        self.tree, _, total_rewrites = attach_rewrites(tree, rewrites)
        all_labels = [i for i in range(total_rewrites)]
        super().__init__(all_labels, init_proba)

    @override
    @classmethod
    def from_source(cls, source, *args, **kwargs):
        return cls(ast.parse(source), *args, **kwargs)

    def apply(self, target_labels: set[int]) -> set[int]:
        _, applied = apply_rewrites(self.tree, target_labels)
        return applied

    @override
    def task_generator(self) -> Iterable[Task]:
        for task in self.generator():
            new_tree, _ = apply_rewrites(self.tree, task.applied)
            try:
                new_tree = deepcopy(new_tree)
                ast.fix_missing_locations(new_tree)
                source = ast.unparse(new_tree)
            except Exception as _:
                continue
            yield task.with_source(source)
            # self.update(task, is_interesting=False)

    @override
    def task_update(self, task: Task, is_interesting: bool):
        self.update(task, is_interesting)


def ruff_fix_code(code_string: str, fix_lint: bool = True, format_code: bool = True) -> str:
    ruff_executable = shutil.which("ruff")
    if not ruff_executable:
        raise FileNotFoundError("Unable to find ruff")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code_string)
        tmp_path = tmp.name

    try:
        if fix_lint:
            print("Running ruff fix on:", tmp_path)
            subprocess.run([ruff_executable, "check", "--fix", "--unsafe-fixes", tmp_path], capture_output=True, check=False)

        if format_code:
            print("Running ruff format on:", tmp_path)
            subprocess.run([ruff_executable, "format", tmp_path], capture_output=True, check=False)

        with open(tmp_path) as f:
            fixed_code = f.read()

        return fixed_code

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class LinePDD(TaskManager, PDD):
    def __init__(self, source: str, init_proba: float = 0.93):
        lines = [line for line in source.splitlines() if line.strip() != ""]
        self.lines = lines
        all_labels = [i for i in range(len(lines))]
        super().__init__(all_labels, init_proba)

    @override
    @classmethod
    def from_source(cls, source, *args, **kwargs):
        return cls(source, *args, **kwargs)

    @override
    def task_generator(self) -> Iterable[Task]:
        for task in self.generator():
            new_lines = [line for idx, line in enumerate(self.lines) if idx not in task.applied]
            source = "\n".join(new_lines)
            try:
                ast.parse(source)
            except Exception as _:
                # self.update(task, is_interesting=False)
                continue
            yield task.with_source(source)

    @override
    def task_update(self, task: Task, is_interesting: bool):
        self.update(task, is_interesting)


class Ruff(TaskManager):
    def __init__(self, source: str, fix_lint: bool = True, format_code: bool = True):
        self.source = source
        self.fix_lint = fix_lint
        self.format_code = format_code
        self.finished = False

    @override
    @classmethod
    def from_source(cls, source: str, *args, **kwargs) -> "Ruff":
        return cls(source)

    @override
    def task_generator(self):
        if self.finished:
            return
        self.finished = True
        try:
            fixed_code = ruff_fix_code(self.source, fix_lint=self.fix_lint, format_code=self.format_code)
            yield Task(source=fixed_code, applied=[], masked=[])
        except FileNotFoundError as _:
            return

    @override
    def task_update(self, task: Task, is_interesting: bool):
        pass


def _worker_loop(input_queue, output_queue):
    while True:
        try:
            task = input_queue.get()
            if task is None:
                break

            capture_out = io.StringIO()
            capture_err = io.StringIO()
            success = False
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
                f.write(task)
                f.flush()
                try:
                    with contextlib.redirect_stdout(capture_out), contextlib.redirect_stderr(capture_err):
                        code = compile(task, f.name, "exec")
                        exec(code, {"__builtins__": __builtins__})
                    success = True
                except SystemExit as e:
                    capture_err.write(f"SystemExit: Code {e.code}\n")
                except Exception:
                    traceback.print_exc(file=capture_err)

            output_queue.put((capture_out.getvalue(), capture_err.getvalue(), success))
        except KeyboardInterrupt:
            break
        except Exception as e:
            output_queue.put(("", f"Critical: {e}", False))


# This class is written by Gemini
class AsyncPythonRunner:
    def __init__(self):
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self.lock = asyncio.Lock()

    def start_proc(self):
        if self.process and self.process.is_alive():
            return
        ctx = multiprocessing.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(target=_worker_loop, args=(self.input_queue, self.output_queue), daemon=True)
        self.process.start()

    def stop_proc(self):
        if self.process:
            # Try to send a stop signal.
            # Note: if the queue is full or broken, put may block, so wrap it in a try.
            with contextlib.suppress(Exception):
                self.input_queue.put_nowait(None)

            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.terminate()
        self.process = None

    def __enter__(self):
        self.start_proc()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_proc()

    async def run(self, code: str, timeout: float = 5.0):
        async with self.lock:
            if not self.process or not self.process.is_alive():
                self.start_proc()

            try:
                self.input_queue.put(code)
            except Exception as e:
                # Rare case: the pipe is broken.
                return "", f"Queue Error: {e}", False

            start_time = time.time()
            while True:
                # 1. Check whether we timed out.
                if time.time() - start_time > timeout:
                    self._handle_timeout(timeout)
                    return "", f"TimeoutError: Exceeded {timeout}s", False

                # 2. Check whether the child process is still alive (avoid hanging if it segfaults).
                if not self.process.is_alive():
                    # Try one last read (in case the result was just written before the process exited).
                    try:
                        return self.output_queue.get_nowait()
                    except queue.Empty:
                        self.process = None  # Mark as needing restart.
                        return "", "Error: Worker process died unexpectedly", False

                # 3. Try to read results in a non-blocking way.
                try:
                    # get_nowait raises queue.Empty immediately if the queue is empty.
                    result = self.output_queue.get_nowait()
                    return result
                except queue.Empty:
                    # No data in the queue yet, sleep briefly and yield control back to the event loop.
                    # A 0.05s delay is perfectly acceptable for interactive usage.
                    await asyncio.sleep(0.05)

    def _handle_timeout(self, timeout):
        """Handle cleanup logic when a timeout happens."""
        # We must force-terminate because exec may still be stuck in a tight loop.
        if self.process and self.process.is_alive():
            self.process.terminate()
            # Give the OS a bit of time to reclaim resources.
            self.process.join(timeout=0.5)

        # Mark as None so that the next run triggers start_proc and restarts the worker.
        self.process = None
        self.input_queue = None
        self.output_queue = None


class SubProcRunner:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    async def run(self, code: str, timeout: float = 5.0):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)

        def run_subprocess(args):
            try:
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,  # Decodes output as strings (Python 3.5+)
                    timeout=timeout,  # Timeout
                    check=False,  # Do not raise exception for non-zero exit codes
                )
                return proc.stdout, proc.stderr, proc.returncode == 0
            except subprocess.TimeoutExpired:
                return "", f"TimeoutError: Exceeded {timeout}s", False

        result = await asyncio.get_running_loop().run_in_executor(None, run_subprocess, ["python3", f.name])
        with contextlib.suppress(OSError):
            os.remove(f.name)
        return result


def clean_empty_pass(code: str) -> str:
    tree = ast.parse(code)

    class PassRemover(ast.NodeTransformer):
        def clean_body(self, body: list[ast.stmt], keep_one=True) -> list[ast.stmt]:
            if body is None:
                return None
            res = [stmt for stmt in body if not isinstance(stmt, ast.Pass)]
            if not res and keep_one:
                return [ast.Pass()]
            return res

        def visit_For(self, node: ast.For) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_If(self, node: ast.If) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            node.orelse = self.clean_body(node.orelse, keep_one=False)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_Module(self, node: ast.Module) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_With(self, node: ast.With) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_While(self, node: ast.While) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_Try(self, node: ast.Try) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            node.orelse = self.clean_body(node.orelse)
            node.finalbody = self.clean_body(node.finalbody)
            for handler in node.handlers:
                handler.body = self.clean_body(handler.body)
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
            self.generic_visit(node)
            node.body = self.clean_body(node.body)
            return node

    new_tree = PassRemover().visit(tree)
    return ast.unparse(new_tree)


JobBackend = Literal["subproc", "runner"]


@dataclass
class ParTaskManager:
    err_msg: str
    text: str
    output_file: Path
    timeout: int = 60
    num_workers: int = 1
    backend: JobBackend = "runner"
    allow_larger: bool = False

    def __post_init__(self):
        self.worker_tasks: list[asyncio.Task] = []
        self.stopped = False
        self.task_manager: TaskManager | None = None
        self.generator: Iterable[Task] | None = None
        self.condition = asyncio.Condition()
        self.waiting_workers = 0
        self.finished = True
        self.task_counter = 0
        self.updated = False

    @property
    def text_len(self):
        return len(self.text)

    def reset(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.generator = task_manager.task_generator()
        self.finished = False

    async def get_next_task(self) -> "Task | None":
        async with self.condition:
            while True:
                if self.stopped:
                    return None
                if self.finished or self.generator is None:
                    await self.condition.wait()
                    continue
                try:
                    result = deepcopy(next(self.generator))
                    self.task_counter += 1
                    if self.task_counter % self.num_workers == 0:
                        print(f"Dispatched {self.task_counter} tasks")
                    return result
                except StopIteration:
                    self.waiting_workers += 1
                    if self.waiting_workers == self.num_workers:
                        self.finished = True
                        self.generator = None
                        self.condition.notify_all()
                    await self.condition.wait()
                    self.waiting_workers -= 1

    async def submit_result(self, task: Task, is_interested: bool):
        async with self.condition:
            self.task_manager.task_update(task, is_interested)
            if is_interested:
                self.generator = self.task_manager.task_generator()
                self.condition.notify_all()
                text = self.post_proc(task.source)
                if len(text) <= self.text_len or self.allow_larger:
                    print("Accept length", len(text))
                    self.text = text
                    self.output_file.write_text(text)
                    self.updated = True

    def post_proc(self, text):
        return clean_empty_pass(text)

    async def worker(self, wid: int):
        runner = AsyncPythonRunner() if self.backend == "runner" else SubProcRunner()
        with runner:
            while True:
                task = await self.get_next_task()
                if task is None:
                    break
                out, err, ok = await runner.run(task.source, timeout=self.timeout)
                is_interested = self.err_msg in out or self.err_msg in err
                await self.submit_result(task, is_interested)

    async def start_workers(self):
        if self.worker_tasks:
            return
        self.stopped = False
        self.worker_tasks = [asyncio.create_task(self.worker(wid)) for wid in range(self.num_workers)]

    async def stop_workers(self):
        if not self.worker_tasks:
            return
        self.stopped = True
        async with self.condition:
            self.condition.notify_all()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks = []
        self.generator = None

    async def run_async(self, task_manager: TaskManager):
        await self.start_workers()
        self.reset(task_manager)
        best_length = self.text_len
        async with self.condition:
            self.condition.notify_all()
            while not self.finished:
                await self.condition.wait()
        return self.text_len < best_length

    async def run_with(self, cls: type[TaskManager], *args, **kwargs):
        allow_larger = kwargs.pop("allow_larger", False)
        if allow_larger:
            self.allow_larger = True
            self.updated = False
        task_manager = cls.from_source(self.text, *args, **kwargs)
        res = await self.run_async(task_manager)
        self.allow_larger = False
        if allow_larger:
            return self.updated
        return res


class Args(NamedTuple):
    source: Path
    err_msg: str
    output: Path
    backend: JobBackend
    timeout: int
    jobs: int


async def main(args: Args):
    if not args.source.exists() or not args.source.is_file():
        raise FileNotFoundError(f"Source file '{args.source}' does not exist or is not a regular file.")
    if not os.access(args.source, os.R_OK):
        raise OSError(f"Source file '{args.source}' is not readable.")
    try:
        source = args.source.read_text()
    except OSError as e:
        raise OSError(f"Failed to read source file '{args.source}': {e}") from e

    manager = ParTaskManager(
        err_msg=args.err_msg,
        text=source,
        output_file=args.output,
        timeout=args.timeout,
        backend=args.backend,
        num_workers=args.jobs,
    )

    # remove any statement

    for_bind_0 = ASTPatRewrite.from_code(
        name="for-bind-0",
        kind="stmt",
        match="for VARS in EXPR: BODY",
        rewrite="VARS = ZEROS\nBODY",
        placeholders={"VARS", "EXPR", "BODY", "ZEROS"},
        derived={
            "ZEROS": lambda ph: expr_to_zeros(ph["VARS"]),
        },
    )

    with_bind_0 = ASTPatRewrite.from_code(
        name="with-bind-0",
        kind="stmt",
        match="with EXPR as VARS: BODY",
        rewrite="with EXPR:\n  VARS = ZEROS\n  BODY",
        placeholders={"VARS", "EXPR", "BODY", "ZEROS"},
        derived={
            "ZEROS": lambda ph: expr_to_zeros(ph["VARS"]),
        },
    )

    assign_rhs_1 = ASTPatRewrite.from_code(
        name="assign-rhs-1",
        kind="stmt",
        match="VAR = EXPR",
        rewrite="VAR = 1",
        placeholders={"VAR", "EXPR"},
    )

    if_remover_1 = ASTPatRewrite.from_code(
        name="if-remover-1",
        kind="stmt",
        match="if COND: BODY",
        rewrite="BODY",
        placeholders={"COND", "BODY"},
    )

    if_remover_2 = ASTPatRewrite.from_code(
        name="if-remover-2",
        kind="stmt",
        match="if COND: BODY\nelse: ELSE_BODY",
        rewrite="BODY",
        placeholders={"COND", "BODY", "ELSE_BODY"},
    )

    if_remover_3 = ASTPatRewrite.from_code(
        name="if-remover-3",
        kind="stmt",
        match="if COND: BODY\nelse: ELSE_BODY",
        rewrite="ELSE_BODY",
        placeholders={"COND", "BODY", "ELSE_BODY"},
    )

    # replace all integer constant x with x // 2
    int_reduce = IntConstApply(lambda x: x > 1, lambda x: x // 2, "int-reduce-2")

    # 1. first, we only do statement level fast reductions
    fast_reducers = [
        if_remover_1,
        if_remover_2,
        if_remover_3,
        for_bind_0,
        GeneralRemove("stmt-remover", ast.stmt, replace_with=ast.Pass()),
    ]

    # 2. canonicalizer enables more simplifications
    canonicalizers = [
        with_bind_0,
        AttachFullFuncArgs(),
    ]

    # 3. simplifiers
    simplifiers = [
        assign_rhs_1,
        CallFwdArg1(),
        BinOpFwdArg("left"),
        BinOpFwdArg("right"),
        GeneralRemove("func-arg-remover", ast.arg),
    ] + fast_reducers

    # 4. finally apply expr level slow reductions
    slow_reducers = [
        GeneralRemove("func-arg-remover", ast.arg),
        GeneralRemove("general-expr-remover", ast.expr),
        GeneralRemove("general-keyword-remover", ast.keyword),
    ] + fast_reducers

    await manager.start_workers()
    manager.text = manager.post_proc(manager.text)
    try:
        while True:
            changed = False
            while await manager.run_with(ASTPDD, fast_reducers):
                changed = True
            await manager.run_with(ASTPDD, canonicalizers, allow_larger=True)
            while await manager.run_with(ASTPDD, simplifiers):
                changed = True
            while await manager.run_with(ASTPDD, [int_reduce], allow_larger=True):
                changed = True
            while await manager.run_with(ASTPDD, slow_reducers):
                changed = True
            if not changed:
                break
    finally:
        await manager.stop_workers()


def cli_main(argv: "Sequence[str] | None" = None) -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        usage="python -m tilelang.autodd source --err-msg MSG -o OUTPUT [--backend {runner,subproc}] [--timeout SEC] [-j N]",
        description="Delta-debug the provided Python source until the target error message remains reproducible.",
        epilog="Author: Kexing Zhou <zhoukexing@pku.edu.cn>",
    )
    parser.add_argument("source", type=Path, help="Input python source file")
    parser.add_argument("--err-msg", type=str, required=True, help="Error message to look for")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--backend", default="runner", choices=["runner", "subproc"], help="Backend for running code: runner is faster, subproc is stable"
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for each task in seconds (default: 60)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel jobs (default: 1)")
    ns = parser.parse_args(argv)

    args = Args(
        source=ns.source,
        err_msg=ns.err_msg,
        output=ns.output,
        backend=ns.backend,
        timeout=ns.timeout,
        jobs=ns.jobs,
    )
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
