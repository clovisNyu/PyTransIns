"""Test file for transins."""
from pytransins.tokenizer import MosesTokenizer, Tokenizer
from pytransins.transins import TransIns


def test_transins_invalid_tokenizer():
    """Test for initial tokenizer checks."""
    try:
        TransIns(Tokenizer())
        raise AssertionError

    except TypeError as exc:
        assert str(exc) == "Tokenizer does not support any languages"

    try:
        mock_tokenizer = Tokenizer()
        mock_tokenizer.languages = ["en"]

        TransIns(mock_tokenizer)
        raise AssertionError

    except NotImplementedError:
        assert True


transins = TransIns(MosesTokenizer())  # Use this for rest of tests.


def test_transins_extract_markup_simple():
    """Test for simple markup extraction."""
    transins.reset()
    test_input = "<h>this is a test</h>"

    transins.extract_markup(test_input)

    assert transins.tokens == ["this", "is", "a", "test"]
    assert transins.tag_map == {0: [1, 0], 1: [1, 0], 2: [1, 0], 3: [1, 0]}
    assert transins.no_token_tags == {}
    assert transins.tag_id_map == {1: "<h>"}


def test_transins_extract_markup_self_closing():
    """Test for handling of self closing tags."""
    transins.reset()
    test_input = "<h>this is <br> a <img src='http://localhost' /> test</h>"

    transins.extract_markup(test_input)

    assert transins.tokens == ["this", "is", "a", "test"]
    assert transins.tag_map == {0: [1, 0], 1: [1, 0], 2: [1, 0], 3: [1, 0]}
    assert transins.no_token_tags == {2: [2], 3: [3]}
    assert transins.tag_id_map == {
        1: "<h>",
        2: "<br />",
        3: '<img src="http://localhost" />',
    }


def test_transins_extract_markup_untagged_tokens():
    """Test for handling of untagged tokens."""
    transins.reset()
    test_input = "<h>this is a <p></p> test</h>"

    transins.extract_markup(test_input)

    assert transins.tokens == ["this", "is", "a", "test"]
    assert transins.tag_map == {0: [1, 0], 1: [1, 0], 2: [1, 0], 3: [1, 0]}
    assert transins.no_token_tags == {3: [2]}
    assert transins.tag_id_map == {1: "<h>", 2: "<p></p>"}


def test_transins_extract_markup_dnt():
    """Test for handling of self defined Do Not Translate tags."""
    transins.reset()
    transins.dnt.append("dnt")
    test_input = "<h>this is a <dnt>My Super Special Text that should not be translated!</dnt> test</h>"

    transins.extract_markup(test_input)

    assert transins.tokens == ["this", "is", "a", "test"]
    assert transins.tag_map == {0: [1, 0], 1: [1, 0], 2: [1, 0], 3: [1, 0]}
    assert transins.no_token_tags == {3: [2]}
    assert transins.tag_id_map == {
        1: "<h>",
        2: "<dnt>My Super Special Text that should not be translated!</dnt>",
    }

    transins.dnt.remove("dnt")


def test_transins_migrate_simple():
    """Test for simple tag migration."""
    transins.reset()

    # 4 tokens, 2 tags with token maps, 1 tag without token map
    transins.tag_map = {0: [1], 1: [1], 2: [1, 2], 3: [1]}
    transins.no_token_tags = {1: [3]}

    alignment = [(0, 0, 1), (1, 2, 1), (2, 1, 1), (3, 3, 1)]

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {0: [1], 1: [1, 2], 2: [1], 3: [1]}
    assert transins.tgt_no_token_tags == {2: [3]}


def test_transins_migrate_threshold():
    """Test for migration thresholds."""
    transins.reset()

    # 4 tokens, 2 tags with token maps, 1 tag without token map
    transins.tag_map = {0: [1], 1: [1], 2: [1, 2], 3: [1]}
    transins.no_token_tags = {1: [3]}

    alignment = [(0, 0, 0), (1, 2, 0), (2, 1, 0), (3, 3, 0)]

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {}
    assert transins.tgt_no_token_tags == {
        1: [3]
    }  # Will still migrate due to how algorithm handles tags with no tokens

    transins.tgt_tag_map = {}
    transins.tgt_no_token_tags = {}

    alignment = [(0, 0, 1), (1, 2, 1), (2, 1, 0.6), (3, 3, 0.4)]

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {0: [1], 1: [1, 2], 2: [1]}
    assert transins.tgt_no_token_tags == {2: [3]}

    transins.tgt_tag_map = {}
    transins.tgt_no_token_tags = {}

    transins.migrate_tags(alignment, threshold=0.7)

    assert transins.tgt_tag_map == {0: [1], 2: [1]}
    assert transins.tgt_no_token_tags == {2: [3]}


def test_transins_migrate_dropped_no_tokens():
    """Test for migrating dropped no token tags."""
    transins.reset()

    # 4 tokens, 2 tags with token maps, 1 tag without token map
    transins.tag_map = {0: [1], 1: [1], 2: [1, 2], 3: [1]}
    transins.no_token_tags = {1: [3]}

    # Alignment for indexes more than dropped available
    alignment = [(0, 0, 1), (2, 1, 1), (3, 3, 1)]

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {0: [1], 1: [1, 2], 3: [1]}
    assert transins.tgt_no_token_tags == {2: [3]}

    transins.reset()

    transins.tag_map = {0: [1], 1: [1], 2: [1, 2], 3: [1]}
    transins.no_token_tags = {3: [3]}

    # Alignment for indexes less than dropped available
    alignment = [(0, 0, 1), (1, 1, 1), (2, 1, 1)]

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {0: [1], 1: [1, 2]}
    assert transins.tgt_no_token_tags == {2: [3]}

    transins.reset()

    transins.tag_map = {0: [1], 1: [1], 2: [1, 2], 3: [1]}
    transins.no_token_tags = {1: [3]}

    # No alignments available
    alignment = []

    transins.migrate_tags(alignment)

    assert transins.tgt_tag_map == {}
    assert transins.tgt_no_token_tags == {0: [3]}


def test_transins_tag_interpolate():
    """Test for simple tag interpolation."""
    transins.reset()

    transins.tgt_tag_map = {
        0: [1],
        1: [],
        2: [],
        3: [1],
    }  # Has to contain empty list. The tag_interpolate method has no information about the number of tokens, so it cannot check if empty

    transins.tag_interpolate()

    assert transins.tgt_tag_map == {0: [1], 1: [1], 2: [1], 3: [1]}

    transins.tgt_tag_map = {0: [1], 1: [], 2: [], 3: [1]}

    transins.tag_interpolate(interpolation_gap=1)

    assert transins.tgt_tag_map == {0: [1], 1: [], 2: [], 3: [1]}

    # Only assign tags that both left of gap and right of gap have.
    transins.tgt_tag_map = {0: [1], 1: [], 2: [], 3: [1, 2]}

    transins.tag_interpolate()

    assert transins.tgt_tag_map == {0: [1], 1: [1], 2: [1], 3: [1, 2]}


def test_transins_tag_interpolate_start_end():
    """Test for interpolating gaps at start and end."""
    # If gap at the start, assign tags at the end
    transins.tgt_tag_map = {0: [], 1: [], 2: [3], 3: [1, 2]}

    transins.tag_interpolate()

    assert transins.tgt_tag_map == {0: [3], 1: [3], 2: [3], 3: [1, 2]}

    # If gap at the end, assign tags at the start
    transins.tgt_tag_map = {0: [1], 1: [], 2: [], 3: [1, 2]}

    transins.tag_interpolate()

    assert transins.tgt_tag_map == {0: [1], 1: [1], 2: [1], 3: [1, 2]}


def test_transins_tag_interpolate_multi():
    """Test for interpolating multiple gaps."""
    transins.tgt_tag_map = {
        0: [1],
        1: [],
        2: [],
        3: [1, 2],
        4: [],
        5: [],
        6: [2],
        7: [1, 2, 3],
        8: [],
        9: [3],
    }

    transins.tag_interpolate()

    assert transins.tgt_tag_map == {
        0: [1],
        1: [1],
        2: [1],
        3: [1, 2],
        4: [2],
        5: [2],
        6: [2],
        7: [1, 2, 3],
        8: [3],
        9: [3],
    }


def test_transins_tag_interpolate_source():
    """Test for interpolating source tags instead of target."""
    transins.reset()

    transins.tag_map = {1: [1], 2: [], 3: [], 4: [1, 2]}

    transins.tag_interpolate(target=False)

    assert transins.tag_map == {1: [1], 2: [1], 3: [1], 4: [1, 2]}
    assert transins.tgt_tag_map == {}


def test_transins_reinsert():
    """Test for reinserting markup."""
    transins.reset()

    transins.tag_interpolate = (
        lambda target, interpolation_gap: None
    )  # Do not test tag interpolation during reinsertion test.

    transins.tag_id_map = {
        1: "<h>",
        2: "<b>",
        3: '<script>console.log("DO NOT TRANSLATE ME")</script>',
        4: "<br />",
    }
    transins.tgt_tag_map = {0: [1], 1: [1, 2], 2: [1], 3: [1]}
    transins.tgt_no_token_tags = {0: [3]}
    tgt_tokens = ["token0", "token1_bolded", "token2", "token3"]

    output = transins.reinsert_markup(tgt_tokens)

    assert (
        output
        == '<h><script>console.log("DO NOT TRANSLATE ME")</script>token0 <b>token1_bolded</b> token2 token3</h>'
    )

    # Test leftover tgt_no_token_tag
    transins.tgt_tag_map = {0: [1], 1: [1, 2], 2: [1], 3: [1]}
    transins.tgt_no_token_tags = {4: [3]}

    output = transins.reinsert_markup(tgt_tokens)

    assert (
        output
        == '<h>token0 <b>token1_bolded</b> token2 token3</h><script>console.log("DO NOT TRANSLATE ME")</script>'
    )


def test_transins_reinsert_source():
    """Test for reinserting markup into source instead of target."""
    transins.reset()

    transins.tag_interpolate = (
        lambda target, interpolation_gap: None
    )  # Do not test tag interpolation during reinsertion test.

    transins.tag_id_map = {
        1: "<h>",
        2: "<b>",
        3: '<script>console.log("DO NOT TRANSLATE ME")</script>',
        4: "<br />",
    }
    transins.tag_map = {0: [1], 1: [1, 2], 2: [1], 3: [1]}
    transins.no_token_tags = {0: [3]}
    tokens = ["token0", "token1_bolded", "token2", "token3"]

    output = transins.reinsert_markup(tokens, target=False)

    assert (
        output
        == '<h><script>console.log("DO NOT TRANSLATE ME")</script>token0 <b>token1_bolded</b> token2 token3</h>'
    )

    # Test leftover no_token_tag
    transins.tag_map = {0: [1], 1: [1, 2], 2: [1], 3: [1]}
    transins.no_token_tags = {4: [3]}

    output = transins.reinsert_markup(tokens, target=False)

    assert (
        output
        == '<h>token0 <b>token1_bolded</b> token2 token3</h><script>console.log("DO NOT TRANSLATE ME")</script>'
    )


def test_transins_test(mocker):
    """Test for transins test extract then reinsert."""
    mocker.patch("pytransins.transins.compare_markup", return_value=1)

    transins.reset()

    output = transins.test("<h>this is a test</h>")

    assert output == "<h>this is a test</h>"
