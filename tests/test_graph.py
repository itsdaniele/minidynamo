"""Tests for the computation graph IR."""

import torch
from minidynamo.graph import Graph, Node, _format_args, _qualified_name


def test_node_creation():
    """Node stores all fields correctly."""
    node = Node(
        "add_0", op="call_function", target=torch.add, args=(1, 2), kwargs={"alpha": 1}
    )
    assert node.name == "add_0"
    assert node.op == "call_function"
    assert node.target is torch.add
    assert node.args == (1, 2)
    assert node.kwargs == {"alpha": 1}


def test_node_default_kwargs():
    """kwargs defaults to empty dict when not provided."""
    node = Node("x", op="placeholder", target="x")
    assert node.kwargs == {}
    assert node.args == ()


def test_node_repr():
    node = Node("add_0", op="call_function", target=torch.add)
    assert repr(node) == "Node('add_0', op='call_function')"


def test_unique_name():
    """_unique_name generates incrementing suffixes."""
    g = Graph()
    assert g._unique_name("add") == "add_0"
    assert g._unique_name("add") == "add_1"
    assert g._unique_name("add") == "add_2"
    assert g._unique_name("mul") == "mul_0"
    assert g._unique_name("add") == "add_3"


def test_placeholder():
    """placeholder creates input nodes and tracks them in graph.inputs."""
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")

    assert x.op == "placeholder"
    assert x.name == "x"
    assert x.target == "x"

    assert len(g.nodes) == 2
    assert g.inputs == [x, y]


def test_call_function():
    """call_function records a function call with node references and constants."""
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")

    # torch.add(x, y) — both args are Nodes
    add_node = g.call_function(torch.add, args=(x, y))
    assert add_node.op == "call_function"
    assert add_node.target is torch.add
    assert add_node.name == "add_0"
    assert add_node.args == (x, y)

    # torch.mul(add_result, 2) — one Node, one constant
    mul_node = g.call_function(torch.mul, args=(add_node, 2))
    assert mul_node.name == "mul_0"
    assert mul_node.args == (add_node, 2)

    assert len(g.nodes) == 4  # 2 placeholders + add + mul


def test_call_function_unique_names():
    """Multiple calls to the same function get distinct names."""
    g = Graph()
    x = g.placeholder("x")
    n1 = g.call_function(torch.add, args=(x, x))
    n2 = g.call_function(torch.add, args=(n1, x))
    assert n1.name == "add_0"
    assert n2.name == "add_1"


def test_call_method():
    """call_method records a method call with self as args[0]."""
    g = Graph()
    x = g.placeholder("x")

    sum_node = g.call_method("sum", args=(x,))
    assert sum_node.op == "call_method"
    assert sum_node.target == "sum"
    assert sum_node.name == "sum_0"
    assert sum_node.args[0] is x

    # Method with extra args: x.reshape(3, 4)
    reshape_node = g.call_method("reshape", args=(x, 3, 4))
    assert reshape_node.args == (x, 3, 4)


def test_output():
    """output marks the return value of the graph."""
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)

    assert out.op == "output"
    assert out.args == (x,)
    assert g.nodes[-1] is out


def test_full_graph_structure():
    """Build a complete graph and verify node ordering."""
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(torch.add, args=(x, y))
    result = g.call_method("sum", args=(add,))
    g.output(result)

    ops = [n.op for n in g.nodes]
    assert ops == [
        "placeholder",
        "placeholder",
        "call_function",
        "call_method",
        "output",
    ]


def test_str_placeholders():
    g = Graph()
    g.placeholder("x")
    g.placeholder("y")
    text = str(g)
    assert "x = placeholder" in text
    assert "y = placeholder" in text


def test_str_call_function():
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    g.call_function(torch.add, args=(x, y))
    text = str(g)
    # Should show something like: add_0 = torch.add(x, y)
    assert "add_0" in text
    assert "x, y" in text


def test_str_call_function_with_constant():
    g = Graph()
    x = g.placeholder("x")
    g.call_function(torch.mul, args=(x, 2))
    text = str(g)
    # Should show: mul_0 = torch.mul(x, 2)
    assert "mul_0" in text
    assert "x, 2" in text


def test_str_call_method():
    g = Graph()
    x = g.placeholder("x")
    g.call_method("sum", args=(x,))
    text = str(g)
    # Should show: sum_0 = x.sum()
    assert "x.sum()" in text


def test_str_call_method_with_args():
    g = Graph()
    x = g.placeholder("x")
    g.call_method("reshape", args=(x, 3, 4))
    text = str(g)
    # Should show: reshape_0 = x.reshape(3, 4)
    assert "x.reshape(3, 4)" in text


def test_str_output():
    g = Graph()
    x = g.placeholder("x")
    g.output(x)
    text = str(g)
    assert "return x" in text


def test_str_full_graph():
    """End-to-end: the printed graph matches expected output."""
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(torch.add, args=(x, y))
    mul = g.call_function(torch.mul, args=(add, 2))
    result = g.call_method("sum", args=(mul,))
    g.output(result)

    text = str(g)
    lines = text.strip().split("\n")
    assert lines[0] == "Graph:"
    assert "x = placeholder" in lines[1]
    assert "y = placeholder" in lines[2]
    assert "add_0" in lines[3] and "x, y" in lines[3]
    assert "mul_0" in lines[4] and "add_0, 2" in lines[4]
    assert "mul_0.sum()" in lines[5]
    assert "return sum_0" in lines[6]


def test_format_args_nodes():
    a = Node("a", op="placeholder", target="a")
    b = Node("b", op="placeholder", target="b")
    assert _format_args((a, b)) == "a, b"


def test_format_args_mixed():
    a = Node("a", op="placeholder", target="a")
    assert _format_args((a, 2, "hello")) == "a, 2, 'hello'"


def test_format_args_empty():
    assert _format_args(()) == ""


def test_qualified_name():
    assert _qualified_name(torch.add) == "torch.add"
    assert _qualified_name(torch.relu) == "torch.relu"


def test_graph_inputs_only_tracks_placeholders():
    """graph.inputs should not include operation nodes."""
    g = Graph()
    x = g.placeholder("x")
    add = g.call_function(torch.add, args=(x, x))
    g.output(add)

    assert g.inputs == [x]
    assert len(g.nodes) == 3  # placeholder + call_function + output
