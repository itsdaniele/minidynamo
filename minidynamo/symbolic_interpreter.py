"""
Symbolic Bytecode Interpreter — the core of minidynamo.

This is a simplified version of TorchDynamo's InstructionTranslator
(found in torch/_dynamo/symbolic_convert.py). It re-implements CPython's
bytecode interpreter, but instead of executing operations on real values,
it executes them on VariableTrackers — symbolic wrappers that record
operations into a Graph.

How it works:
  1. Takes a Python function + example arguments
  2. Uses dis.get_instructions() to read the function's bytecode
  3. Creates TensorVariable placeholders for each tensor argument
  4. Walks instructions one by one, dispatching to handler methods
  5. Each handler manipulates the symbolic stack/locals (just like CPython)
  6. When tensor operations are encountered, they become Graph nodes
  7. Returns the completed Graph

CPython is a stack machine: it has a value stack and a locals array.
Our interpreter models a small subset; its stack holds VariableTrackers
instead of real Python objects.

Real Dynamo handles many more instructions, including control flow.
This interpreter rejects jumps and branches.
"""

import dis
import inspect
import operator
import sys

import torch

from .graph import Graph
from .guards import Guard, GuardSet
from .variable_tracker import (
    ConstantVariable,
    MethodVariable,
    TensorVariable,
    TorchVariable,
)

# ── Supported torch functions ──────────────────────────────────────────
# These are the torch functions we know how to trace. When the interpreter
# sees a call to one of these with tensor arguments, it records a graph node
# and executes the operation on example tensors to propagate metadata.
SUPPORTED_TORCH_FUNCTIONS = {
    # Unary
    torch.abs,
    torch.neg,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    torch.exp,
    torch.log,
    torch.sin,
    torch.cos,
    torch.sqrt,
    # Binary
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.pow,
    torch.matmul,
    torch.mm,
    # Reduction
    torch.sum,
    torch.mean,
}

# Only operations known to be deterministic and free of mutation. Calling
# arbitrary Python during example propagation can duplicate side effects.
SUPPORTED_TENSOR_METHODS = {
    "abs",
    "neg",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sin",
    "cos",
    "sqrt",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",
    "mm",
    "sum",
    "mean",
    "max",
    "min",
    "reshape",
    "view",
    "flatten",
    "transpose",
    "permute",
    "contiguous",
    "clone",
    "detach",
    "to",
    "float",
    "double",
}
CONSTANT_FUNCTIONS = {len, int, float, bool, abs}


def validate_function(fn):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("minidynamo requires Python 3.10 bytecode.")
    if not inspect.isfunction(fn):
        raise TypeError("minidynamo expects a Python function.")
    code = fn.__code__
    if code.co_freevars:
        raise NotImplementedError(
            "Closures are not supported; use constants in the function body."
        )
    if code.co_kwonlyargcount or code.co_flags & (
        inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
    ):
        raise NotImplementedError("Only fixed positional parameters are supported.")
    if fn.__defaults__:
        raise NotImplementedError("Default arguments are not supported.")


def validate_inputs(fn, args):
    if len(args) != fn.__code__.co_argcount:
        raise TypeError(
            f"{fn.__name__} expects {fn.__code__.co_argcount} positional arguments, got {len(args)}"
        )
    if not all(
        type(arg) is torch.Tensor and arg.layout == torch.strided and not arg.__dict__
        for arg in args
    ):
        raise TypeError(
            "minidynamo only supports positional tensor arguments "
            "(plain, strided tensors without custom attributes)."
        )


def is_constant(value):
    if type(value) in (int, float, bool, str, type(None), torch.dtype, torch.device):
        return True
    return type(value) in (tuple, torch.Size) and all(is_constant(v) for v in value)


# Map Python binary operators to their torch equivalents.
# When we see BINARY_ADD on two TensorVariables, we record torch.add in the graph.
BINARY_OP_TO_TORCH = {
    operator.add: torch.add,
    operator.sub: torch.sub,
    operator.mul: torch.mul,
    operator.truediv: torch.div,
    operator.pow: torch.pow,
}


class SymbolicInterpreter:
    """
    A symbolic bytecode interpreter.

    This mirrors CPython's evaluation loop but operates on VariableTrackers.
    It has:
      - self.stack: the value stack (list of VariableTrackers)
      - self.locals: local variables (dict of name -> VariableTracker)
      - self.graph: the Graph being built

    For each bytecode instruction, there's an op_XXXX handler method.
    """

    def __init__(self, fn, example_args):
        validate_function(fn)
        validate_inputs(fn, example_args)
        self.fn = fn
        self.graph = Graph()
        self.stack = []
        self.locals = {}
        self.globals = fn.__globals__
        # CPython binds this namespace when the function is created. It can
        # differ from the process-wide builtins or globals['__builtins__'].
        self.builtins = fn.__builtins__
        self.guards = GuardSet.from_example_inputs(example_args)

        # Create placeholder nodes for each function argument.
        # This is where tracing begins: each tensor arg becomes a
        # TensorVariable with a placeholder node in the graph.
        code = fn.__code__
        arg_names = code.co_varnames[: code.co_argcount]
        for name, example in zip(arg_names, example_args):
            if isinstance(example, torch.Tensor):
                node = self.graph.placeholder(name)
                self.locals[name] = TensorVariable(node, example)
            else:
                self.locals[name] = ConstantVariable(example)

    def run(self):
        """
        Execute the bytecode symbolically and return the captured Graph.

        This is the main loop — analogous to CPython's _PyEval_EvalFrameDefault.
        We fetch one instruction at a time, dispatch to the handler, and repeat.
        """
        instructions = list(dis.get_instructions(self.fn))

        for inst in instructions:
            # Find the handler method for this opcode
            handler = getattr(self, f"op_{inst.opname}", None)
            if handler is None:
                raise NotImplementedError(
                    f"Unsupported bytecode: {inst.opname}. "
                    f"minidynamo only supports straight-line tensor code."
                )
            handler(inst)

        return self.graph

    # ── Stack helpers ──────────────────────────────────────────────────
    # These mirror CPython's internal stack operations.

    def push(self, val):
        """Push a VariableTracker onto the stack."""
        self.stack.append(val)

    def pop(self):
        """Pop and return the top of stack."""
        return self.stack.pop()

    def popn(self, n):
        """Pop n items from the stack, returned in original push order."""
        if n == 0:
            return []
        items = self.stack[-n:]
        self.stack[-n:] = []
        return items

    # ── Load/Store instructions ────────────────────────────────────────

    def op_LOAD_FAST(self, inst):
        """
        Push a local variable onto the stack.
        Bytecode: LOAD_FAST varname
        CPython equivalent: push locals[varname]
        """
        self.push(self.locals[inst.argval])

    def op_STORE_FAST(self, inst):
        """
        Pop the top of stack and store it in a local variable.
        Bytecode: STORE_FAST varname
        CPython equivalent: locals[varname] = pop()
        """
        self.locals[inst.argval] = self.pop()

    def op_LOAD_CONST(self, inst):
        """
        Push a constant value onto the stack.
        Bytecode: LOAD_CONST value
        Examples: numbers (2, 3.14), None, tuples of constants
        """
        self.push(ConstantVariable(inst.argval))

    def op_LOAD_GLOBAL(self, inst):
        """
        Push a global variable onto the stack.
        Bytecode: LOAD_GLOBAL name

        This is how we see `torch` — when the function references the global
        `torch`, Python emits LOAD_GLOBAL torch. We wrap it as a TorchVariable
        so that later LOAD_ATTR/CALL operations know to record graph nodes.
        """
        name = inst.argval
        namespace = self.globals if name in self.globals else self.builtins
        if name not in namespace:
            raise NameError(name)
        value = namespace[name]
        # Identity is conservative for immutable constants: rebinding to an
        # equal object may retrace, but can never reuse a stale value.
        self.guards.add(
            Guard(
                lambda *args, ns=namespace, globs=self.globals, key=name, expected=value: (
                    ns.get(key, _MISSING) is expected
                    and (ns is globs or key not in globs)
                ),
                f"global {name} is unchanged",
            )
        )
        self.push(self._wrap_global(value))

    def _wrap_global(self, value):
        if value is torch or any(
            value is fn for fn in SUPPORTED_TORCH_FUNCTIONS | CONSTANT_FUNCTIONS
        ):
            return TorchVariable(value)
        if is_constant(value):
            return ConstantVariable(value)
        raise NotImplementedError(
            "Unsupported global or callable; use supported torch operations and immutable constants."
        )

    # ── Attribute access ───────────────────────────────────────────────

    def op_LOAD_ATTR(self, inst):
        """
        Pop an object, push getattr(object, name).
        Bytecode: LOAD_ATTR name

        Three cases:
        1. TorchVariable(torch).relu -> TorchVariable(torch.relu)
        2. TensorVariable.shape     -> ConstantVariable((3, 4))
        3. TensorVariable.sum       -> MethodVariable(tensor, "sum")
        """
        obj = self.pop()
        attr_name = inst.argval

        if isinstance(obj, TorchVariable):
            if obj.value is not torch:
                raise NotImplementedError(
                    "Only attributes of the torch module are supported."
                )
            attr = getattr(obj.value, attr_name)
            self.guards.add(
                Guard(
                    lambda *args, name=attr_name, expected=attr: (
                        getattr(torch, name, _MISSING) is expected
                    ),
                    f"torch.{attr_name} is unchanged",
                )
            )
            self.push(self._wrap_global(attr))

        elif isinstance(obj, TensorVariable):
            if attr_name in ("shape", "dtype", "device", "ndim"):
                # Metadata is specialized; input guards protect this value.
                val = getattr(obj.example_value, attr_name)
                self.push(ConstantVariable(val))
            elif attr_name in SUPPORTED_TENSOR_METHODS | {"dim", "size", "numel"}:
                self.push(MethodVariable(obj, attr_name))
            else:
                raise NotImplementedError(
                    f"Unsupported tensor attribute or method: {attr_name}"
                )
        else:
            raise NotImplementedError(f"Unsupported attribute: {attr_name}")

    def op_LOAD_METHOD(self, inst):
        """
        Like LOAD_ATTR but specifically for method calls.
        Bytecode: LOAD_METHOD name

        In Python 3.10, `x.sum()` generates LOAD_METHOD + CALL_METHOD
        instead of LOAD_ATTR + CALL_FUNCTION. The semantics are the same
        for our purposes.
        """
        self.op_LOAD_ATTR(inst)

    # ── Function/method calls ──────────────────────────────────────────

    def op_CALL_FUNCTION(self, inst):
        """
        Call a function with positional arguments.
        Bytecode: CALL_FUNCTION nargs

        Stack before: [fn, arg0, arg1, ..., argN-1]
        Stack after:  [result]
        """
        nargs = inst.argval
        args = self.popn(nargs)
        fn = self.pop()
        result = self._handle_call(fn, args)
        self.push(result)

    def op_CALL_METHOD(self, inst):
        """
        Call a method.
        Bytecode: CALL_METHOD nargs

        Stack before: [method_or_self, arg0, ..., argN-1]
        Stack after:  [result]

        For LOAD_METHOD + CALL_METHOD, the method object is on the stack.
        """
        nargs = inst.argval
        args = self.popn(nargs)
        method = self.pop()
        result = self._handle_call(method, args)
        self.push(result)

    def _handle_call(self, fn, args):
        """
        Dispatch a function/method call.

        Given a callable VariableTracker
        and its arguments, we either:
          a) Record a graph node (if it's a tensor operation)
          b) Evaluate it concretely (if it's a pure Python operation on constants)
        """
        if isinstance(fn, TorchVariable) and fn.value in SUPPORTED_TORCH_FUNCTIONS:
            return self._call_torch_function(fn.value, args)

        if isinstance(fn, MethodVariable):
            return self._call_tensor_method(fn, args)

        if isinstance(fn, TorchVariable) and fn.value in CONSTANT_FUNCTIONS:
            if not all(isinstance(a, ConstantVariable) for a in args):
                raise NotImplementedError(
                    "Python calls on tensor data are not supported."
                )
            concrete_args = [self._to_concrete(a) for a in args]
            result = fn.value(*concrete_args)
            return self._wrap_result(result)

        raise RuntimeError(f"Don't know how to call {type(fn).__name__}: {fn}")

    def _call_torch_function(self, torch_fn, args):
        """
        Record a torch function call in the graph.

        1. Extract node references (for tensors) or raw values (for constants)
        2. Create a call_function node in the graph
        3. Compute the example output for shape propagation
        4. Return a new TensorVariable wrapping the node
        """
        if not any(isinstance(arg, TensorVariable) for arg in args):
            raise NotImplementedError(
                "Tensor operations require a tensor argument; tensor factories are unsupported."
            )

        # Build graph args: Node for tensors, raw value for constants
        graph_args = tuple(
            a.node if isinstance(a, TensorVariable) else self._to_concrete(a)
            for a in args
        )

        # Record in graph
        node = self.graph.call_function(torch_fn, graph_args)

        # Compute example value (for shape/dtype tracking)
        example_args = tuple(
            a.example_value if isinstance(a, TensorVariable) else self._to_concrete(a)
            for a in args
        )
        example_result = torch_fn(*example_args)
        if type(example_result) is not torch.Tensor:
            raise NotImplementedError("Operations must return a single tensor.")
        return TensorVariable(node, example_result)

    def _call_tensor_method(self, method_var, args):
        """
        Record a tensor method call in the graph.

        For x.sum(): method_var.tensor_var is x, method_var.method_name is "sum"
        """
        tensor_var = method_var.tensor_var
        method_name = method_var.method_name

        if method_name in {"dim", "size", "numel"}:
            if not all(isinstance(a, ConstantVariable) for a in args):
                raise NotImplementedError(
                    "Metadata queries require constant arguments."
                )
            return ConstantVariable(
                getattr(tensor_var.example_value, method_name)(*(a.value for a in args))
            )

        # Graph args: self tensor node + method args
        graph_args = (tensor_var.node,) + tuple(
            a.node if isinstance(a, TensorVariable) else self._to_concrete(a)
            for a in args
        )

        # Record in graph
        node = self.graph.call_method(method_name, graph_args)

        # Compute example value
        example_args = tuple(
            a.example_value if isinstance(a, TensorVariable) else self._to_concrete(a)
            for a in args
        )
        example_result = getattr(tensor_var.example_value, method_name)(*example_args)
        if type(example_result) is not torch.Tensor:
            raise NotImplementedError("Operations must return a single tensor.")
        return TensorVariable(node, example_result)

    # ── Binary operations ──────────────────────────────────────────────

    def op_BINARY_ADD(self, inst):
        self._binary_op(operator.add)

    def op_BINARY_SUBTRACT(self, inst):
        self._binary_op(operator.sub)

    def op_BINARY_MULTIPLY(self, inst):
        self._binary_op(operator.mul)

    def op_BINARY_TRUE_DIVIDE(self, inst):
        self._binary_op(operator.truediv)

    def op_BINARY_POWER(self, inst):
        self._binary_op(operator.pow)

    def _binary_op(self, py_op):
        """
        Handle a binary operation like + - * / **.

        Stack before: [left, right]
        Stack after:  [result]

        If either operand is a tensor, we record torch.add/sub/mul/div in
        the graph. If both are constants, we just compute the result.
        """
        right = self.pop()
        left = self.pop()

        has_tensor = isinstance(left, TensorVariable) or isinstance(
            right, TensorVariable
        )

        if has_tensor:
            # Record as a torch operation in the graph
            # Tensor.__rsub__ and Tensor.__rtruediv__ support scalars on the
            # left where the corresponding torch function may not.
            torch_fn = (
                py_op
                if isinstance(left, ConstantVariable)
                and py_op in (operator.sub, operator.truediv)
                else BINARY_OP_TO_TORCH[py_op]
            )
            result = self._call_torch_function(torch_fn, [left, right])
            self.push(result)
        else:
            # Both constants — evaluate directly
            result = py_op(self._to_concrete(left), self._to_concrete(right))
            self.push(ConstantVariable(result))

    # ── Unary operations ───────────────────────────────────────────────

    def op_UNARY_NEGATIVE(self, inst):
        """Negate the top of stack: push(-pop())."""
        operand = self.pop()
        if isinstance(operand, TensorVariable):
            result = self._call_torch_function(torch.neg, [operand])
            self.push(result)
        else:
            self.push(ConstantVariable(-self._to_concrete(operand)))

    # ── Return ─────────────────────────────────────────────────────────

    def op_RETURN_VALUE(self, inst):
        """
        Return from the function.
        Bytecode: RETURN_VALUE

        Pops the return value and marks it as the graph output.
        This is always the last instruction we process.
        """
        retval = self.pop()
        if isinstance(retval, TensorVariable):
            self.graph.output(retval.node)
        else:
            raise RuntimeError(
                "minidynamo expects the function to return a tensor. "
                f"Got {type(retval).__name__} instead."
            )

    # ── Misc instructions ──────────────────────────────────────────────

    def op_POP_TOP(self, inst):
        """Discard the top of stack. Used after statements with unused results."""
        self.pop()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _to_concrete(self, var):
        """Extract the concrete Python value from a VariableTracker."""
        if isinstance(var, ConstantVariable):
            return var.value
        if isinstance(var, TensorVariable):
            return var.example_value
        if isinstance(var, TorchVariable):
            return var.value
        raise RuntimeError(f"Cannot make {type(var).__name__} concrete")

    def _wrap_result(self, value):
        """Wrap a Python value back into the appropriate VariableTracker."""
        if isinstance(value, torch.Tensor):
            raise NotImplementedError(
                "A call evaluated at trace time returned a new tensor. "
                "Tensor factories like torch.ones or torch.randn inside the "
                "traced function are not supported: they would need their own "
                "graph nodes to be replayed at run time."
            )
        return ConstantVariable(value)


_MISSING = object()
