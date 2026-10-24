"""Tests for the variable tracker hierarchy."""

import torch
from minidynamo.graph import Graph
from minidynamo.variable_tracker import (
    VariableTracker,
    TensorVariable,
    ConstantVariable,
    TorchVariable,
    MethodVariable,
)


# --- TensorVariable ---


def test_tensor_variable_holds_node_and_example():
    """TensorVariable links a graph node to an example tensor."""
    g = Graph()
    node = g.placeholder("x")
    example = torch.randn(3, 4)

    tv = TensorVariable(node, example)
    assert tv.node is node
    assert tv.example_value is example


def test_tensor_variable_is_variable_tracker():
    g = Graph()
    node = g.placeholder("x")
    tv = TensorVariable(node, torch.randn(2))
    assert isinstance(tv, VariableTracker)


def test_tensor_variable_example_metadata():
    """We can read shape/dtype/device from the example value."""
    g = Graph()
    node = g.placeholder("x")
    example = torch.zeros(5, 3, dtype=torch.float64)

    tv = TensorVariable(node, example)
    assert tv.example_value.shape == (5, 3)
    assert tv.example_value.dtype == torch.float64


# --- ConstantVariable ---


def test_constant_variable_int():
    cv = ConstantVariable(42)
    assert cv.value == 42


def test_constant_variable_float():
    cv = ConstantVariable(3.14)
    assert cv.value == 3.14


def test_constant_variable_none():
    cv = ConstantVariable(None)
    assert cv.value is None


def test_constant_variable_string():
    cv = ConstantVariable("hello")
    assert cv.value == "hello"


def test_constant_variable_tuple():
    """Constants can hold compound values like shape tuples."""
    cv = ConstantVariable((3, 4, 5))
    assert cv.value == (3, 4, 5)


def test_constant_variable_is_variable_tracker():
    assert isinstance(ConstantVariable(1), VariableTracker)


# --- TorchVariable ---


def test_torch_variable_module():
    """TorchVariable can wrap the torch module itself."""
    tv = TorchVariable(torch)
    assert tv.value is torch


def test_torch_variable_function():
    """TorchVariable can wrap a specific torch function."""
    tv = TorchVariable(torch.relu)
    assert tv.value is torch.relu
    assert callable(tv.value)


def test_torch_variable_is_variable_tracker():
    assert isinstance(TorchVariable(torch.add), VariableTracker)


# --- MethodVariable ---


def test_method_variable():
    """MethodVariable binds a tensor variable to a method name."""
    g = Graph()
    node = g.placeholder("x")
    tensor_var = TensorVariable(node, torch.randn(3))

    mv = MethodVariable(tensor_var, "sum")
    assert mv.tensor_var is tensor_var
    assert mv.method_name == "sum"


def test_method_variable_preserves_tensor_reference():
    """The tensor_var in MethodVariable still has its node and example."""
    g = Graph()
    node = g.placeholder("x")
    example = torch.randn(4, 5)
    tensor_var = TensorVariable(node, example)

    mv = MethodVariable(tensor_var, "reshape")
    assert mv.tensor_var.node is node
    assert mv.tensor_var.example_value is example


def test_method_variable_is_variable_tracker():
    g = Graph()
    node = g.placeholder("x")
    tv = TensorVariable(node, torch.randn(2))
    assert isinstance(MethodVariable(tv, "sum"), VariableTracker)


# --- Usage pattern: simulating what the interpreter does ---


def test_simulated_trace_x_plus_y():
    """
    Simulate what the interpreter does for `z = x + y`:

    1. LOAD_FAST x  -> push TensorVariable for x
    2. LOAD_FAST y  -> push TensorVariable for y
    3. BINARY_ADD   -> pop both, create graph node, push result TensorVariable
    """
    g = Graph()

    # Step 1-2: create input variables (interpreter does this at start)
    x_node = g.placeholder("x")
    y_node = g.placeholder("y")
    x_example = torch.tensor([1.0, 2.0, 3.0])
    y_example = torch.tensor([4.0, 5.0, 6.0])
    x_var = TensorVariable(x_node, x_example)
    y_var = TensorVariable(y_node, y_example)

    # Step 3: BINARY_ADD — what the interpreter does:
    #   a) record the operation in the graph
    add_node = g.call_function(torch.add, args=(x_var.node, y_var.node))
    #   b) compute example output for shape propagation
    add_example = torch.add(x_var.example_value, y_var.example_value)
    #   c) wrap result as a new TensorVariable
    z_var = TensorVariable(add_node, add_example)

    # Verify: the graph captured the operation
    assert z_var.node.op == "call_function"
    assert z_var.node.target is torch.add
    assert z_var.node.args == (x_node, y_node)
    # Verify: example value is correct
    assert torch.equal(z_var.example_value, torch.tensor([5.0, 7.0, 9.0]))


def test_simulated_trace_x_times_constant():
    """
    Simulate `w = x * 2`:

    The constant 2 becomes a ConstantVariable. When used in a binary op
    with a tensor, its .value (not a Node) goes into the graph args.
    """
    g = Graph()
    x_node = g.placeholder("x")
    x_var = TensorVariable(x_node, torch.tensor([1.0, 2.0]))
    const_var = ConstantVariable(2)

    # BINARY_MULTIPLY: tensor * constant
    # The constant's raw value goes into args (not a Node)
    mul_node = g.call_function(torch.mul, args=(x_var.node, const_var.value))
    mul_example = torch.mul(x_var.example_value, const_var.value)
    w_var = TensorVariable(mul_node, mul_example)

    assert w_var.node.args == (x_node, 2)
    assert torch.equal(w_var.example_value, torch.tensor([2.0, 4.0]))


def test_simulated_trace_method_call():
    """
    Simulate `result = x.sum()`:

    1. LOAD_FAST x     -> TensorVariable
    2. LOAD_METHOD sum -> MethodVariable(x, "sum")
    3. CALL_METHOD 0   -> graph.call_method("sum", args=(x.node,))
    """
    g = Graph()
    x_node = g.placeholder("x")
    x_var = TensorVariable(x_node, torch.tensor([1.0, 2.0, 3.0]))

    # Step 2: LOAD_METHOD
    method_var = MethodVariable(x_var, "sum")

    # Step 3: CALL_METHOD — interpreter uses method_var to build the node
    sum_node = g.call_method(
        method_var.method_name,
        args=(method_var.tensor_var.node,),
    )
    sum_example = method_var.tensor_var.example_value.sum()
    result_var = TensorVariable(sum_node, sum_example)

    assert result_var.node.op == "call_method"
    assert result_var.node.target == "sum"
    assert result_var.node.args[0] is x_node
    assert result_var.example_value.item() == 6.0
