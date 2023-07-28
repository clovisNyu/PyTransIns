"""Tokenizer modules to be used by transins. Moses is added as an example by default."""
import logging
from typing import List


def check_lang(func):
    """Check if language is supported. Used as a decorator."""

    def tokenize(self, text, lang):
        if lang not in self.languages:
            raise TypeError(f"Language {lang} not supported!")

        return func(self, text, lang)

    return tokenize


class Tokenizer:
    """Base Tokenizer class to inherit from when writing tokenizers."""

    def __init__(self) -> None:
        self.languages = []  # List of supported languages

    @check_lang
    def tokenize(self, text: str, lang: str) -> List[str]:
        """
        Tokenize text given a language.

        Parameters
        ----------
        text : str
            Text to tokenize
        lang : str
            Language of text to tokenize. Necesary as tokenizers are generally not language agnostic.


        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        raise NotImplementedError

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens. Necessary as not all languages use space separators.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to detokenize.


        Returns
        -------
        output : str
            Detokenized text.
        """
        raise NotImplementedError


class TokenizerGroup(Tokenizer):
    """Tokenizer class for grouping mutliple tokenizers."""

    def __init__(self):
        """
        Run initialization of base Tokenizer class and language_tokenizer_map attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(TokenizerGroup, self).__init__()
        self.language_tokenizer_map = {}

    def add_tokenizer(self, tokenizer: Tokenizer) -> None:
        """
        Add tokenizer to tokenizer group, overriding tokenizer for language if already exists.

        Parameters
        ----------
        tokenizer : Tokenizer
            Object instance of a class that inherits from the Tokenizer base class.

        Returns
        -------
        None
        """
        if not tokenizer.languages:
            logging.warning("No languages supported by tokenizer provided. Not adding.")
            return

        try:
            tokenizer.tokenize("test string", tokenizer.languages[0])

        except NotImplementedError:
            logging.warning("tokenize not implemented. Not adding.")
            return

        try:
            tokenizer.detokenize(["test", "string"])

        except NotImplementedError:
            logging.warning(
                "detokenize not implemented. Defaulting to Moses for languages added"
            )
            tokenizer.detokenize = MosesTokenizer().detokenize

        for lang in tokenizer.languages:
            if lang not in self.languages:
                self.languages.append(lang)

            self.language_tokenizer_map[
                lang
            ] = tokenizer  # Override original if it was there.

    @check_lang
    def tokenize(self, text: str, lang: str) -> List[str]:
        """
        Tokenize text given a language. Routes to tokenizer object based on language.

        Parameters
        ----------
        text : str
            Text to tokenize
        lang : str
            Language of text to tokenize. Necesary as tokenizers are generally not language agnostic.


        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        tokenizer_to_use = self.language_tokenizer_map[lang]

        output = tokenizer_to_use.tokenize(text, lang)

        return output

    def detokenize(self, tokens: List[str], lang: str = "en"):
        """
        Detokenize a list of tokens. Necessary as not all languages use space separators. Routes to tokenizer object based on language.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to detokenize.


        Returns
        -------
        output : str
            Detokenized text.
        """
        tokenizer_to_use = self.language_tokenizer_map[lang]

        output = tokenizer_to_use.detokenize(tokens)

        return output

    def tokenizer_info(self) -> str:
        """
        Collate language to tokenizer summary.

        Parameters
        ----------
        None

        Returns
        -------
        output : str
            Language to tokenizer summary
        """
        output = "\n".join(
            f"{lang}: {type(tokenizer)}"
            for lang, tokenizer in self.language_tokenizer_map.items()
        )

        logging.info(output)

        return output


class MosesTokenizer(Tokenizer):
    """Basic tokenizer based on Moses."""

    def __init__(self):
        """
        Populate required attributes: tokenizer, detokenizer, languages.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(MosesTokenizer, self).__init__()

        import sacremoses  # Import sacremoses to use moses tokenizer

        self.tokenizer = sacremoses.MosesTokenizer()
        self.detokenizer = sacremoses.MosesDetokenizer()
        self.languages = ["en", "es"]

    def tokenize(self, text: str, lang: str):
        """
        Implement basic moses tokenizer.

        Parameters
        ----------
        test : str
            Text to tokenize
        lang : str
            Language of text to tokenize. Necesary as tokenizers are generally not language agnostic.


        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        tokens = self.tokenizer.tokenize(text, return_str=False)
        return tokens

    def detokenize(self, tokens: List[str]):
        """
        Implement basic moses detokenizer.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to detokenize.


        Returns
        -------
        output : str
            Detokenized text.
        """
        output = self.detokenizer.detokenize(tokens)
        return output
