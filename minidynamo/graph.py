"""
Computation Graph IR — a minimal version of torch.fx.Graph.

This is the data structure that stores the captured computation. When we
"trace" a function, we walk its bytecode and record every tensor operation
as a Node in this Graph. The result is a simple DAG (directed acyclic graph)
that can be compiled into optimized code.

In real TorchDynamo, this role is played by torch.fx.Graph and torch.fx.Node.
"""


class Node:
    """
    A single operation in the computation graph.

    Fields:
        name   — unique identifier, e.g. "add_0", "x"
        op     — one of: "placeholder", "call_function", "call_method", "output"
        target — what to call:
                   - for call_function: the actual callable (e.g. torch.add)
                   - for call_method: the method name as a string (e.g. "sum")
                   - for placeholder/output: just the name
        args   — positional arguments (can reference other Nodes)
        kwargs — keyword arguments
    """

    def __init__(self, name, op, target, args=(), kwargs=None):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def __repr__(self):
        return f"Node({self.name!r}, op={self.op!r})"


class Graph:
    """
    An ordered sequence of Nodes representing a computation.

    The graph is always structured as:
      1. Placeholder nodes (inputs)
      2. Operation nodes (call_function / call_method)
      3. One output node (the return value)
    """

    def __init__(self):
        self.nodes = []
        self.inputs = []  # placeholder nodes only
        self._name_counter = {}
        self._used_names = set()

    def _unique_name(self, base):
        """Generate a unique name like 'add_0', 'add_1', etc."""
        count = self._name_counter.get(base, 0)
        while f"{base}_{count}" in self._used_names:
            count += 1
        self._name_counter[base] = count + 1
        name = f"{base}_{count}"
        self._used_names.add(name)
        return name

    def placeholder(self, name):
        """
        Create an input placeholder node.

        In real FX, this corresponds to function arguments. When we later
        compile the graph, placeholders become parameters of the generated function.
        """
        if name in self._used_names:
            raise ValueError(f"Duplicate graph name: {name}")
        self._used_names.add(name)
        node = Node(name, op="placeholder", target=name)
        self.nodes.append(node)
        self.inputs.append(node)
        return node

    def call_function(self, target, args, kwargs=None):
        """
        Record a function call like torch.add(x, y).

        `target` is the actual Python callable.
        `args` is a tuple that can contain Node references (for tensor args)
        or plain Python values (for constants like the `2` in `x * 2`).
        """
        name = self._unique_name(target.__name__)
        node = Node(name, op="call_function", target=target, args=args, kwargs=kwargs)
        self.nodes.append(node)
        return node

    def call_method(self, method_name, args, kwargs=None):
        """
        Record a method call like x.sum() or x.reshape(3, 4).

        `args[0]` is always the self Node (the tensor the method is called on).
        `args[1:]` are the method arguments.
        """
        name = self._unique_name(method_name)
        node = Node(
            name, op="call_method", target=method_name, args=args, kwargs=kwargs
        )
        self.nodes.append(node)
        return node

    def output(self, node):
        """Mark a node as the return value of the graph."""
        out = Node("output", op="output", target="output", args=(node,))
        self.nodes.append(out)
        return out

    def __str__(self):
        """Pretty-print the graph. Similar to graph.print_tabular() in FX."""
        lines = ["Graph:"]
        for node in self.nodes:
            if node.op == "placeholder":
                lines.append(f"  {node.name} = placeholder")

            elif node.op == "call_function":
                args_str = _format_args(node.args)
                fn_name = _qualified_name(node.target)
                lines.append(f"  {node.name} = {fn_name}({args_str})")

            elif node.op == "call_method":
                self_node = node.args[0]
                rest = node.args[1:]
                args_str = _format_args(rest)
                lines.append(
                    f"  {node.name} = {self_node.name}.{node.target}({args_str})"
                )

            elif node.op == "output":
                lines.append(f"  return {node.args[0].name}")

        return "\n".join(lines)


def _format_args(args):
    """Format a tuple of args for printing."""
    parts = []
    for a in args:
        if isinstance(a, Node):
            parts.append(a.name)
        else:
            parts.append(repr(a))
    return ", ".join(parts)


def _qualified_name(fn):
    """Get a readable name for a function like 'torch.add'."""
    module = getattr(fn, "__module__", "")
    name = getattr(fn, "__name__", str(fn))
    if module:
        return f"{module}.{name}"
    return name
