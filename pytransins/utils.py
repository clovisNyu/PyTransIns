"""Set of utility functions to be used with the transins library."""
from bs4 import BeautifulSoup
from bs4.element import Tag
from zss import Node, simple_distance


def is_tag(element: str) -> bool:
    """Check if string provided is a markup tag."""
    return element.startswith("<") and element.endswith(">")


def get_opening_tag(element: Tag) -> str:
    """
    Convert a tag to the string representation of the opening tag.

    Parameters
    ----------
    element : Tag
        BeautifulSoup Tag element to convert


    Returns
    -------
    output : str
        Opening tag with attributes of the element provided.
    """
    raw_attrs = {
        k: v if not isinstance(v, list) else " ".join(v)
        for k, v in element.attrs.items()
    }
    attrs = " ".join((f'{k}="{v}"' for k, v in raw_attrs.items()))
    if attrs:
        return f"<{element.name} {attrs}>"
    else:
        return f"<{element.name}>"


def _convert_to_nodes(element: Tag):
    """
    Recursive conversion of Document Tree into a graph understood by zss library.

    Parameters
    ----------
    element : Tag
        BeautifulSoup Tag element to recurse down


    Returns
    -------
    node : Node
        zss Node object that stores the subtree from the given element.
    """
    try:
        children = element.contents

    except Exception:  # Unable to get children. Can happen for some tags like script.
        return Node("NULL", [])

    sub_doc = []
    for child in children:
        sub_doc.append(_convert_to_nodes(child))

    return Node(get_opening_tag(element), sub_doc)


def convert_to_nodes(markup: str) -> Node:
    """
    Convert markup document tree into a graph understood by zss library.

    Parameters
    ----------
    markup : str
        markup document as a string to be converted


    Returns
    -------
    tree : Node
        zss Node object that stores the entire tree.
    """
    soup = BeautifulSoup(markup, "html.parser")
    tree = _convert_to_nodes(soup)
    return tree


def compare_markup(src: str, tgt: str) -> int:
    """
    Compute the tree edit distance between 2 markup documents.

    Parameters
    ----------
    src : str
        Source markup document
    tgt : str
        Target markup document


    Returns
    -------
    score : int
        Tree edit distance as given by the ZSS algorithm
    """
    src_tree = convert_to_nodes(src)
    tgt_tree = convert_to_nodes(tgt)
    score = simple_distance(src_tree, tgt_tree)
    return score
