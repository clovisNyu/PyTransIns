"""Module to handle markup extraction and reinsertion."""
import logging
from collections import OrderedDict
from typing import List, Union

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from pytransins.tokenizer import MosesTokenizer, Tokenizer, TokenizerGroup
from pytransins.utils import compare_markup, get_opening_tag, is_tag


class TransIns:
    """Class to orchestrate extraction and reinsertion of markup via the Complete Mapping Strategy as described by the 2019 EMNLP paper TransIns."""

    def __init__(self, tokenizer: Union[None, Tokenizer] = None) -> None:
        """
        Initialize TransIns object to handle markup extraction and reinsertion.

        Parameters
        ----------
        tokenizer : Union[None, Tokenizer]
            Tokenizer to use. If set to None, will initialize MosesTokenizer. (default None)

        Returns
        -------
        None
        """

        self.logger = logging.getLogger("pytransins")
        if tokenizer is None:
            tokenizer = MosesTokenizer()

        # Check if tokenizer works
        try:
            if not tokenizer.languages:
                raise TypeError("Tokenizer does not support any languages")

            res = tokenizer.tokenize("this is a test", tokenizer.languages[0])
            tokenizer.detokenize(res)

        except Exception as exc:
            self.logger.error(exc)
            raise

        self.tokenizer = TokenizerGroup()

        self.tokenizer.add_tokenizer(tokenizer)

        self.self_closing = [
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
            "command",
            "keygen",
            "menuitem",
        ]  # List of tags to treat as self closing

        self.dnt = ["script", "style"]  # Do not translate tags

        self.tag_id_map = {}  # Maps tag IDs to actual tags

        # Source version
        self.tag_map = {}
        self.tokens = []
        self.no_token_tags = {}

        # Target version
        self.tgt_tag_map = {}
        self.tgt_tokens = []
        self.tgt_no_token_tags = {}

    def _extract_markup(
        self, element: Tag, offset: int = 0, lang: str = "en"
    ) -> List[str]:
        """
        Traverses down the document tree (depth first) from a given element to populate the tag_map and no_token_tags dictionaries.

        Parameters
        ----------
        element : bs4.element.Tag
            The root element to start recursing down
        offset : int (default 0)
            An offset to keep track of how many tokens have been generated


        Returns
        -------
        token_buffer : List[str]
            A list containing the plaintext tokens extracted
        """
        # Assign a tag ID to the tag
        if not self.tag_id_map:
            tag_id = 0

        else:
            tag_id = max(self.tag_id_map.keys()) + 1

        if (
            type(element) == NavigableString
        ):  # Text Node, without any child tags. No more markup to process.
            token_buffer = self.tokenizer.tokenize(element.text, lang=lang)
            return token_buffer

        elif type(element) == Comment:
            self.tag_id_map[tag_id] = f"<!-- {str(element)} -->"
            if offset not in self.no_token_tags:
                self.no_token_tags[offset] = []

            self.no_token_tags[offset].append(tag_id)
            return []

        try:
            children = element.contents

        except Exception as exc:
            self.logger.warning(
                f"Dropping tag as could not get children of non text node and non comment: {exc}"
            )
            return []

        self.tag_id_map[tag_id] = get_opening_tag(element)

        # Has children, but not any text nodes. Wrap entire element as 1, without further processing. Also include do not translate tags
        if (children and not element.text) or element.name in self.dnt:
            self.tag_id_map[tag_id] += (
                "".join(str(child) for child in children) + f"</{element.name}>"
            )
            if offset not in self.no_token_tags:
                self.no_token_tags[offset] = []

            self.no_token_tags[offset].append(tag_id)
            return []

        token_buffer = []

        for child in children:
            child_tokens = self._extract_markup(
                child, offset=offset + len(token_buffer)
            )  # Recurse over each child

            # Assign each token a token id and map the tags
            for i in range(
                offset + len(token_buffer),
                offset + len(token_buffer) + len(child_tokens),
            ):
                if i not in self.tag_map:
                    self.tag_map[i] = []

                self.tag_map[i].append(tag_id)

            token_buffer.extend(child_tokens)

        if (
            not token_buffer
        ):  # All children did not return any tokens to tag to. Often occurs for self closing tags
            if element.name in self.self_closing:
                self.tag_id_map[tag_id] = self.tag_id_map[tag_id][:-1] + " />"

            else:
                self.tag_id_map[tag_id] += f"</{element.name}>"

            if offset not in self.no_token_tags:
                self.no_token_tags[offset] = []

            self.no_token_tags[offset].append(tag_id)

        return token_buffer

    def extract_markup(self, raw: str, lang="en") -> None:
        """
        Populate the tag_map, no_token_tags and tokens attributes. (see _extract_markup method).

        Parameters
        ----------
        raw : str
            Raw markup string
        lang : str (default en)
            Language to use for tokenizer

        Returns
        -------
        None
        """
        self.logger.debug("BEGIN EXTRACTION PROCESS")
        if lang not in self.tokenizer.languages:
            raise TypeError(f"Tokenizer not able to handle language: {lang}")

        soup = BeautifulSoup(raw, "html.parser")
        self.tokens = self._extract_markup(soup, lang=lang)
        del self.tag_id_map[0]  # Remove <[document]> tag inserted by BeautifulSoup

    def migrate_tags(self, alignments: List[tuple], threshold: float = 0.5, force_migrate: bool = False) -> None:
        """
        Migrates the tag map to target tokens based on provided alignments. Target alignments stored in tgt_tag_map and tgt_no_token_tags attributes.

        Paramters
        ---------
        alignments : List[tuple[int,int,float]]
            List of tuples where each tuple contains, in order, the source token index, the target token index, and the score for that alignment
        threshold : float (default 0.5)
            Threshold for algorithm to accept the alignment.
        force_migrate: bool (default False)
            Forces tags in source to migrate to target regardless of threshold set.


        Returns
        -------
        None
        """
        self.logger.debug("BEGIN TAG MIGRATION")
        if not self.tag_map:
            self.logger.warning("No tag map found! Did you run extract_markup already?")

        untagged_ids = list(
            self.no_token_tags.keys()
        )  # Keep track of which tokens have no_token_tags mapped but were not migrated. Only done for tags without tokens as tag interpolation will not be able to deal with them.

        secondary_alignment = []
        unmigrated_tag_ids = self.tag_id_map.keys()

        for src_idx, tgt_idx, score in alignments:
            if score < threshold:
                self.logger.debug(f"{src_idx} TO {tgt_idx} SKIPPED, BELOW THRESHOLD")
                secondary_alignment.append((src_idx, tgt_idx, score))
                continue

            self.logger.debug(f"{src_idx} MAPPED TO {tgt_idx}")
            if src_idx in self.tag_map:
                if tgt_idx not in self.tgt_tag_map:
                    self.tgt_tag_map[tgt_idx] = []

                self.tgt_tag_map[tgt_idx].extend(self.tag_map[src_idx])
                unmigrated_tag_ids = [tag for tag in unmigrated_tag_ids if tag not in self.tag_map[src_idx]]

            if src_idx in self.no_token_tags:
                self.logger.debug(f"NO-TOKEN-TAGS FOUND FOR SOURCE INDEX {src_idx}")
                if tgt_idx not in self.tgt_no_token_tags:
                    self.tgt_no_token_tags[tgt_idx] = []

                self.tgt_no_token_tags[tgt_idx].extend(self.no_token_tags[src_idx])
                unmigrated_tag_ids = [tag for tag in unmigrated_tag_ids if tag not in self.no_token_tags[src_idx]]
                if src_idx in untagged_ids:
                    untagged_ids.remove(src_idx)
        

        available_sources = {src_idx: tgt_idx for src_idx, tgt_idx, _ in alignments} # Dropping scores and restructuring as dict.

        # Migrate no_token_tags to nearest token
        if untagged_ids:
            self.logger.debug("NO-TOKEN-TAGS FAILED TO MIGRATE")
            self.logger.debug("APPROXIMATING TO NEAREST TOKEN")

        for untagged_id in untagged_ids:
            if untagged_id in unmigrated_tag_ids:
                unmigrated_tag_ids.remove(untagged_id)

            available_candidates = [
                src_idx for src_idx in available_sources if src_idx >= untagged_id
            ]
            if (
                not available_candidates
            ):  # Check for any src indexes in alignment that are larger than or equal to untagged
                available_candidates = [
                    src_idx for src_idx in available_sources if src_idx < untagged_id
                ]  # If there aren't any, check if any src indexes in alignment that are less than untagged

            if (
                not available_candidates
            ):  # If there are no alignments for any tags more than equal to, or less than untagged (empty alignment)
                self.tgt_no_token_tags[0] = self.no_token_tags[untagged_id]
                self.logger.warning(
                    f"UNABLE TO MAP TAG ID WITH NO TOKENS FROM SOURCE {untagged_id}. ASSIGNING TAGS TO FIRST TOKEN TO AVOID DROPPING. CHECK IF ALIGNMENT IS CORRECT!"
                )

                continue

            closest = (
                min(available_candidates)
                if min(available_candidates) >= untagged_id
                else max(available_candidates)
            )
            tgt_idx = available_sources[closest]
            self.logger.debug(f"ASSIGNING NO-TOKEN-TAG ID {untagged_id} TO TOKEN INDEX {tgt_idx}")
            if closest not in self.tgt_no_token_tags:
                self.tgt_no_token_tags[closest] = []

            self.tgt_no_token_tags[closest].extend(self.no_token_tags[untagged_id])
        
        if unmigrated_tag_ids:
            if not force_migrate:
                self.logger.warning("UNMIGRATED TAGS FOUND! IS YOUR THRESHOLD TOO HIGH?")
            
            else:
                # NOTE: UNTESTED. UNSURE OF HOW WELL THIS PERFORMS.
                self.logger.debug("FORCE MIGRATING TAGS")
                for src_idx, tgt_idx, _ in secondary_alignment:
                    tags = self.tag_map[src_idx]
                    shared = set(tags).intersection(set(unmigrated_tag_ids))
                    if not shared:
                        continue

                    if tgt_idx not in self.tgt_tag_map:
                        self.tgt_tag_map[tgt_idx] = []

                    self.tgt_tag_map[tgt_idx].extend(shared)

        # Remove duplicates
        for key in self.tgt_tag_map:
            self.tgt_tag_map[key] = list(set(self.tgt_tag_map[key]))

        for key in self.tgt_no_token_tags:
            self.tgt_no_token_tags[key] = list(set(self.tgt_no_token_tags[key]))

    def tag_interpolate(self, target: bool = True, interpolation_gap: int = 2) -> None:
        """
        Perform tag interpolation to alleviate word alignment errors.

        Parameters
        ----------
        target : bool (default True)
            Interpolate the target tokens stored in tgt_tag_map attribute. Will interpolate source tags if set to False.
        interpolation_gap : int (default 2)
            Maximum number of consecutive untagged tokens before the

        Returns
        -------
        None
        """
        if target:
            to_interpolate = self.tgt_tag_map

        else:
            to_interpolate = self.tag_map

        if not to_interpolate:
            self.logger.warn("NO ENTRIES IN TAG MAP! NOTHING TO INTERPOLATE!")
            return

        to_interpolate = dict(OrderedDict(sorted(to_interpolate.items())))

        gap = 0
        gap_start = 0
        token_idx_checker = -1

        for token_idx, tags in to_interpolate.items():
            if (
                token_idx != token_idx_checker + 1
            ):  # Case should be handled when calling from reinsert_markup method. Not guaranteed when calling directly.
                self.logger.warning(
                    f"MISSING TOKEN INDEX {token_idx_checker + 1}! ASSIGN AN EMPTY LIST TO INTERPOLATE!"
                )

            token_idx_checker = token_idx

            if not tags:
                if not gap:
                    gap_start = token_idx

                gap += 1
                continue

            if not gap:
                continue

            if gap <= interpolation_gap:
                if gap_start == 0:
                    interpolated_tags = to_interpolate[token_idx]

                else:
                    interpolated_tags = list(
                        set(to_interpolate[gap_start - 1]).intersection(
                            set(to_interpolate[token_idx])
                        )
                    )

                for i in range(gap_start, gap_start + gap):
                    to_interpolate[i] = interpolated_tags

                gap = 0

        if gap and gap <= interpolation_gap and gap_start != 0:
            for i in range(gap_start, gap_start + gap):
                to_interpolate[i] = to_interpolate[gap_start - 1]

        if target:
            self.tgt_tag_map = to_interpolate

        else:
            self.tag_map = to_interpolate

    def reinsert_markup(
        self,
        tokens: List[str],
        target: bool = True,
        interpolation_gap: int = 2,
        lang: str = "en",
    ) -> str:
        """
        Reinserts markup tags into a provided list of tokens.

        Parameters
        ----------
        token : List[str]
            List of tokens to reinsert markup tags into
        target : bool (default True)
            Reinsert based on target token map. Set to false to use source token map.
        interpolation_gap : int (default 2)
            Interpolation gap to use
        lang : str
            Language to use for detokenizer


        Returns
        -------
        output : str
            Output markup after tags have been reinserted
        """
        if not tokens:
            raise TypeError("No tokens to migrate to!")

        if (not self.tgt_tag_map) and (target):
            self.logger.warning(
                "TAG_MAP EMPTY FOR TARGET. DID YOU RUN MIGRATE_TAGS?"
            )

        if target:
            for i in range(len(tokens)):
                if i not in self.tgt_tag_map:
                    self.tgt_tag_map[i] = []

        self.tag_interpolate(target=target, interpolation_gap=interpolation_gap)

        if target:
            to_reinsert = self.tgt_tag_map
            to_reinsert_no_tokens = self.tgt_no_token_tags

        else:
            to_reinsert = self.tag_map
            to_reinsert_no_tokens = self.no_token_tags

        active_tags = []
        output_buffer = []
        _output_buffer = []  # To hold list of tokens for detokenizing

        for token_idx, token in enumerate(tokens):
            new_tags = to_reinsert[token_idx]

            opening = [tag for tag in new_tags if tag not in active_tags]
            closing = list(
                reversed([tag for tag in active_tags if tag not in new_tags])
            )

            if token_idx in to_reinsert_no_tokens:
                no_tokens = to_reinsert_no_tokens[token_idx]

            else:
                no_tokens = []

            if opening or closing or no_tokens: # There exist some change in tags, clear the string buffer.
                if _output_buffer:
                    output_buffer.append(
                        self.tokenizer.detokenize(_output_buffer, lang=lang)
                    )
                    _output_buffer = []

            # Information on ordering of no-token-tags and closing tags are not captured. 
            # Hard coded to insert no-token-tags after closing.
            for closing_tag_id in closing: # First In Last Out structure of XML dictates that closing tags must be inserted first.
                active_tags.remove(closing_tag_id)
                if closing_tag_id == 0: # BeautifulSoup inserted tag
                    continue

                if closing_tag_id not in self.tag_id_map:
                    self.logger.warning(
                        f"Tag ID {closing_tag_id} NOT FOUND WHEN CLOSING! SKIPPING!"
                    )
                    continue

                new_tag = "</" + self.tag_id_map[closing_tag_id].split(" ", 1)[0][1:]
                if not new_tag.endswith(">"):
                    new_tag += ">"

                output_buffer.append(new_tag)
                # TODO: Handle whitespace insertion around tags
                # Added to avoid tokens occuring immediately after tag from sticking to previous token.
                # Not necessarily true for all languages, but much more damaging to readability of outcome if its missing when it needs to be there.
                output_buffer.append(" ")

            for opening_tag_id in opening:
                if opening_tag_id == 0:
                    continue

                if opening_tag_id not in self.tag_id_map:
                    self.logger.warning(
                        f"Tag ID {opening_tag_id} NOT FOUND WHEN OPENING! SKIPPING!"
                    )
                    continue
                active_tags.append(opening_tag_id)

                new_tag = self.tag_id_map[opening_tag_id]
                if (
                    output_buffer
                    and output_buffer[-1] != " "
                    and not is_tag(output_buffer[-1])
                ):
                    output_buffer.append(" ")

                # Handle order of insertion of no-token-tags by tag ID.
                _no_tokens = [tag_id for tag_id in no_tokens if tag_id < opening_tag_id]
                for tag_id in _no_tokens:
                    _new_tag = self.tag_id_map[tag_id]
                    output_buffer.append(_new_tag)
                
                no_tokens = [tag_id for tag_id in no_tokens if tag_id not in _no_tokens]

                output_buffer.append(new_tag)

            for tag_id in no_tokens:
                if tag_id not in self.tag_id_map:
                    self.logger.warning(
                        f"Tag ID {tag_id} NOT FOUND WHEN SELF CLOSING! SKIPPING!"
                    )
                    continue

                new_tag = self.tag_id_map[tag_id]

                if (
                    output_buffer
                    and output_buffer[-1] != " "
                    and not is_tag(output_buffer[-1])
                ):
                    output_buffer.append(" ")

                output_buffer.append(new_tag)

            _output_buffer.append(token)

            active_tags = [tag for tag in active_tags if tag not in closing]
            active_tags.extend(opening)
            active_tags = list(set(active_tags))

        if _output_buffer:
            output_buffer.append(self.tokenizer.detokenize(_output_buffer, lang=lang))

        # Close buffers.
        if active_tags:
            for tag_id in reversed(active_tags):
                if tag_id == 0:
                    continue

                if tag_id not in self.tag_id_map:
                    self.logger.warning(
                        f"Tag ID {tag_id} NOT FOUND WHEN CLOSING! SKIPPING!"
                    )
                    continue

                new_tag = "</" + self.tag_id_map[tag_id].split(" ", 1)[0][1:]
                if not new_tag.endswith(">"):
                    new_tag += ">"

                output_buffer.append(new_tag)

        if to_reinsert_no_tokens and max(to_reinsert_no_tokens.keys()) == len(tokens):
            output_buffer.extend(
                [
                    self.tag_id_map[tag_id]
                    for tag_id in to_reinsert_no_tokens[len(tokens)]
                ]
            )

        output = "".join(output_buffer)
        return output

    def test(
        self,
        text: str,
        to_log: List[str] = ("tokens", "tag_map", "no_token_tags", "tag_id_map"),
    ) -> str:
        """
        Extract and reinsert immediately to test the extraction and reinsertion algorithm. Logs info to stdout.

        Parameters
        ----------
        text : str
            Raw text with markup to test with
        to_log : List[str]
            List of parameters to log to stdout


        Returns
        -------
        output : str
            Output after extracting and reinserting markup.
        """
        logging.basicConfig(level=logging.DEBUG)
        self.extract_markup(text)
        target_tokens = self.tokens.copy()
        alignments = [(i, i, 1) for i in range(len(target_tokens))]
        self.migrate_tags(alignments)
        output = self.reinsert_markup(target_tokens)
        tree_edit_dist = compare_markup(text, output)

        log_string_source = ["Source info"]
        log_string_target = ["Target info"]

        if "tokens" in to_log:
            log_string_source.append(f"Tokens: {self.tokens}")
            log_string_target.append(f"Target Tokens: {target_tokens}")

        if "tag_map" in to_log:
            log_string_source.append(f"Tag Map: {self.tag_map}")
            log_string_target.append(f"Target Tag Map: {self.tgt_tag_map}")

        if "no_token_tags" in to_log:
            log_string_source.append(f"No Token Tags: {self.no_token_tags}")
            log_string_target.append(f"Target No Token Tags: {self.tgt_no_token_tags}")

        log_string = []

        if "tag_id_map" in to_log:
            log_string.append(f"Tag ID Map: {self.tag_id_map}")

        log_string.append("\n\n".join(log_string_source))
        log_string.append("\n\n".join(log_string_target))

        log_string = (
            "\n-----------------------------------\n"
            + "\n\n-----------------------------------\n\n".join(log_string)
            + "\n\n-----------------------------------\n\n"
            + f"Output: {output}\n"
            + f"Tree Edit Distance: {tree_edit_dist}"
            + "\n\n-----------------------------------\n\n"
        )

        self.logger.info(log_string)
        logging.basicConfig(level=logging.WARNING)
        self.reset()
        return output

    def reset(self) -> None:
        """
        Reset the instance to handle new task.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tag_id_map = {}

        # Source version
        self.tag_map = {}
        self.tokens = []
        self.no_token_tags = {}

        # Target version
        self.tgt_tag_map = {}
        self.tgt_tokens = []
        self.tgt_no_token_tags = {}


if __name__ == "__main__":
    test_transins = TransIns()
    test_text = r"<h><p>this is a</p> <p><b><!-- DO NOT TRANSLATE -->test</b></p></h>"
    output = test_transins.test(test_text)
