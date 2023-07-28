"""Test for utils."""
from zss import Node

from pytransins.utils import compare_markup, convert_to_nodes


def test_convert_to_nodes():
    """Test for node conversion."""
    test_input = "<h>this is <b>a</b> test</h>"
    expected_graph = Node(
        "<[document]>",
        [Node("<h>", [Node("NULL"), Node("<b>", [Node("NULL")]), Node("NULL")])],
    )
    result = convert_to_nodes(test_input)
    assert str(result) == str(expected_graph)


def test_convert_to_nodes_faulty():
    """Test for faulty markup string."""
    test_input = "<h>some text"
    expected_graph = Node("<[document]>", [Node("<h>", [Node("NULL")])])

    result = convert_to_nodes(test_input)

    assert result == expected_graph


def test_compare_markup(mocker):
    """Test for markup comparison."""
    mocker.patch("pytransins.utils.convert_to_nodes", return_value=Node("root_node"))
    result = compare_markup(
        "<h>this is a test</h>", "<p>this is <b>another</b> test</p>"
    )

    assert result == 0

    mocker.patch("pytransins.utils.simple_distance", return_value=1)
    result = compare_markup("<h>this is a test</h>", "<h>this is a test</h>")
    assert result == 1
