"""
Variable Trackers — symbolic values used during bytecode tracing.

When the symbolic interpreter executes bytecode, it doesn't work with real
Python objects. Instead, every value on the stack and in local variables is
a VariableTracker — a symbolic wrapper that knows:

  1. What kind of value it represents (tensor, constant, torch function, etc.)
  2. How it connects to the computation graph (for tensors: which Node)
  3. Enough metadata to generate guards (for tensors: shape, dtype, device)

In real TorchDynamo, VariableTracker has many subclasses covering
Python types (lists, dicts, iterators, user classes, etc.). We only need 4.

Hierarchy:
  VariableTracker          — base class
  ├── TensorVariable       — a symbolic tensor (references a graph Node)
  ├── ConstantVariable     — a concrete Python value (int, float, str, None, ...)
  ├── TorchVariable        — the torch module or a torch function
  └── MethodVariable       — a bound tensor method (e.g., x.sum)
"""


class VariableTracker:
    """
    Base class for all symbolic values.

    Every value the symbolic interpreter manipulates is one of these.
    Think of it as: "I'm not a real value — I'm a *description* of a value
    that will exist at runtime."
    """

    pass


class TensorVariable(VariableTracker):
    """
    A symbolic tensor — the most important variable type.

    It holds both a symbolic reference and real example data:
      - node: the graph Node that produces this tensor
      - example_value: a real tensor with the same metadata (shape, dtype, device)
        used for two purposes:
          1. Generating guards ("this arg must be shape [4, 3]")
          2. Propagating shapes through operations (we execute example operations
             to obtain the output metadata)

    When the interpreter sees `x + y` where x and y are TensorVariables,
    it records the operation and computes an example result:
      1. Creates a new Node in the graph: call_function(torch.add, (x.node, y.node))
      2. Computes the example output: torch.add(x.example_value, y.example_value)
      3. Returns a new TensorVariable(node=new_node, example_value=example_output)
    """

    def __init__(self, node, example_value):
        self.node = node
        self.example_value = example_value


class ConstantVariable(VariableTracker):
    """
    A concrete Python value — fully known at trace time.

    Examples: the `2` in `x * 2`, a dtype like `torch.float32`, a shape tuple.

    Constants don't create graph nodes. They get embedded directly into the
    graph as literal values. When we see `torch.mul(tensor_var, const_var)`,
    the graph node's args will be `(Node, 2)` — the constant is inlined.

    In real Dynamo, constants are "specialized" — their exact value becomes
    part of the guard. If you call fn(x, 2) and later fn(x, 3), the guard
    for the first cached version fails (because 2 != 3), triggering recompilation.
    """

    def __init__(self, value):
        self.value = value


class TorchVariable(VariableTracker):
    """
    A reference to the torch module or a callable from it.

    Covers two cases:
      - The module itself: when the interpreter sees `LOAD_GLOBAL torch`,
        it pushes TorchVariable(torch)
      - A function: when it then sees `LOAD_ATTR relu`, it resolves
        torch.relu and pushes TorchVariable(torch.relu)

    When this gets called (CALL_FUNCTION), the interpreter checks: "is this
    a torch function that operates on tensors?" If yes, it records a
    call_function node and executes an example call for metadata.
    """

    def __init__(self, value):
        self.value = value


class MethodVariable(VariableTracker):
    """
    A bound method on a tensor, like x.sum or x.reshape.

    Created when the interpreter sees LOAD_METHOD/LOAD_ATTR on a TensorVariable.
    For example, when tracing `x.sum()`:
      1. LOAD_FAST x        -> pushes TensorVariable(x_node, x_example)
      2. LOAD_METHOD sum     -> pushes MethodVariable(tensor_var=x, method_name="sum")
      3. CALL_METHOD 0       -> creates graph node: call_method("sum", args=(x.node,))

    We need to remember both which tensor it's bound to (tensor_var) and
    which method (method_name) so we can record the right graph node.
    """

    def __init__(self, tensor_var, method_name):
        self.tensor_var = tensor_var
        self.method_name = method_name
