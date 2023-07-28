"""Test file for tokenizers."""
from pytransins.tokenizer import Tokenizer, TokenizerGroup, check_lang


class MockTokenizer(Tokenizer):
    """Mock tokenizer for testing."""

    def __init__(self):
        super(Tokenizer, self).__init__()
        self.languages = ["en"]

    @check_lang
    def tokenize(self, text: str, lang: str):
        """Tokenize for testing."""
        return text.split(" ")

    def detokenize(self, tokens):
        """Detokenize for testing."""
        return " ".join(tokens)


def test_tokenize_not_implemented():
    """Test for NotImplementedError. Should raise if user did not overwrite the base tokenizer methods."""
    mock_tokenizer = Tokenizer()
    mock_tokenizer.languages = ["en"]

    try:
        mock_tokenizer.tokenize("this is a test", "en")
        raise AssertionError

    except NotImplementedError:
        assert True

    try:
        mock_tokenizer.detokenize("this is a test")
        raise AssertionError

    except NotImplementedError:
        assert True


def test_tokenize_language_not_supported():
    """Test for TypeError. Should raise if user sends tokenize job for unsupported language."""
    mock_tokenizer = MockTokenizer()
    try:
        mock_tokenizer.tokenize("this is a test", "id")

    except TypeError as exc:
        assert str(exc) == "Language id not supported!"


def test_tokenize():
    """Test for correct tokenizing."""
    mock_tokenizer = MockTokenizer()

    res = mock_tokenizer.tokenize("this is a test", "en")
    assert res == ["this", "is", "a", "test"]

    res = mock_tokenizer.detokenize(res)
    assert res == "this is a test"


tokenizer_group = TokenizerGroup()


def test_tokenizer_group():
    """Test for TokenizerGroup methods."""
    mock_tokenizer_en = MockTokenizer()
    tokenizer_group.add_tokenizer(mock_tokenizer_en)

    mock_tokenizer_zh = MockTokenizer()
    mock_tokenizer_zh.languages = ["zh"]

    tokenizer_group.add_tokenizer(mock_tokenizer_zh)

    assert tokenizer_group.languages == ["en", "zh"]


def test_tokenizer_group_faulty():
    """Test for TokenizerGroup faulty add_tokenizer input."""
    mock_tokenizer_faulty = Tokenizer()
    mock_tokenizer_faulty.languages = ["some other language"]

    tokenizer_group.add_tokenizer(mock_tokenizer_faulty)

    assert "some other language" not in tokenizer_group.languages

    mock_tokenizer_faulty.tokenize = lambda x, y: x.split(" ")
    tokenizer_group.add_tokenizer(mock_tokenizer_faulty)

    assert "some other language" in tokenizer_group.languages
