"""Set of utility functions to be used with the transins library."""
from io import BytesIO

from lxml import etree
from lxml.etree import XMLParser, _Element
from zss import Node, simple_distance


def is_tag(element: str) -> bool:
    """Check if string provided is a markup tag."""
    return element.startswith("<") and element.endswith(">")


def get_opening_tag(element: _Element) -> str:
    """
    Convert a tag to the string representation of the opening tag.

    Parameters
    ----------
    element : _Element
        lxml _Element to convert


    Returns
    -------
    output : str
        Opening tag with attributes of the element provided.
    """
    attrs = element.items()

    if attrs:
        return (
            f"<{element.tag} "
            + " ".join(f'{key}="{value}"' for key, value in attrs)
            + ">"
        )

    return f"<{element.tag}>"


def _convert_to_nodes(element: _Element):
    """
    Recursive conversion of Document Tree into a graph understood by zss library.

    Parameters
    ----------
    element : _Element
        lxml _Element element to recurse down


    Returns
    -------
    node : Node
        zss Node object that stores the subtree from the given element.
    """
    try:
        children = element.getchildren()

    except Exception:  # Unable to get children. Can happen for some tags like script.
        if element.text:
            return Node(get_opening_tag(element), [Node("NULL", [])])

        return Node("NULL", [])

    sub_doc = []
    if element.text:
        sub_doc.append(Node("NULL", []))

    for child in children:
        sub_doc.append(_convert_to_nodes(child))

        if child.tail:
            sub_doc.append(Node("NULL", []))

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
    parser = XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(BytesIO(markup.encode("utf-8")), parser=parser).getroot()

    tree = _convert_to_nodes(root)
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
