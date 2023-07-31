# PyTransIns

## Introduction
PyTransIns is a Python implementation of the Complete Mapping Strategy (CMS) described in the 2021 EMNLP paper [TransIns: Document Translation with Markup Reinsertion](https://aclanthology.org/2021.emnlp-demo.4.pdf). This library is meant to handle markup extraction and reinsertion for the purposes of Machine Translation. It requires you to provide your own Machine Translation and Word Alignment capabilities, but abstracts away the need to deal with the markup so that you can use black box MT systems meant for plaintext.

## Installation

Clone this git repo. To get the wheel, CD into the directory and run

```bash
pip install poetry
poetry build
```

Or simply do 
```bash
pip install git+https://github.com/clovisNyu/PyTransIns.git
```

## Usage

### Basic Flow
Create a TransIns object. This class is used to manage markup.

```python
from pytransins.transins import TransIns

transins = TransIns()

markup = "<h>this is <b>markup</b></h>"

transins.extract_markup(markup, lang = "en")

assert transins.tokens == ["this", "is", "markup"]
assert transins.tag_map == {0: [1, 0], 1: [1, 0], 2: [2, 1, 0]}

# tag_map is interpreted as token 0 has tag IDs 1, token 1 has tag IDs 1, token 2 has tag IDs 1 and 2.
# You can view which tags the tag IDs represent via the tag_id_map attribute.

print(f"Tag IDs map: {transins.tag_id_map}") # Map of tag ID to the opening tag

from typing import List
def translate(src: str, src_lang: str, tgt_lang: str):
    # REPLACE WITH TRANSLATION FUNCTION
    return src

def align(src_tokens: List[str], tgt_tokens: List[str]):
    # REPLACE WITH ALIGNMENT FUNCTION
    alignments = []
    for i, _ in enumerate(src_tokens):
        alignments.append((i, min(i, len(tgt_tokens)-1), 1))
    
    return alignments

# Get translation and alignments
plaintext = transins.tokenizer.detokenize(transins.tokens, lang = "en")
translation = translate(plaintext, "en", "es")
target_tokens = transins.tokenizer.tokenize(translation)
alignments = align(transins.tokens, translation)

# Migrate the tags based on the alignment to generate the target token to tag ID map
transins.migrate_tags(alignments)

# Tag ID 0 is removed. This is the <[document]> tag that bs4 inserts. We remove it as we do not want to migrate it to the target
print(f"Target token to tag ID map: {transins.tgt_tag_map}") 

# Reinsert the markup
output = transins.reinsert_markup(target_tokens)
print(output)
```

`output` should contain the translated text with markup reinserted.

### Self Closing Tags
In HTML, self closing tags refer to tags that do not require an opening and closing tag. For example `<img />` or `<br />`. Note that when not using XHTML, self closing tags need not end with "/>". (`<br>` instead of `<br />`) The `TransIns` module keeps a list of tags to treat as self closing and convert them to XHTML compliant format. This will only apply if BeautifulSoup cannot find any child tags, which should always be the case for self closing tags, but on the odd event that a tag is in the `self_closing`

```python
assert "br" in transins.self_closing

transins.reset()

markup = "<p>this <br> is a test</p>"
transins.extract_markup(markup, lang = "en")

assert transins.tag_id_map == {1: "<p>", 2: "<br />"}
assert transins.no_token_tags == {1: [2]}

# no_token_tags interpreted as token 1 has tag id 2
```
**Note**
Self closing tags are not handled by the CMS described by the TransIns paper. Going purely by that strategy described in the paper, all such self closing tags will be dropped as there are no tokens to assign these tags to. The TransIns module tags the self closing tags to the token that occurs immediately after the tag. This will inevitably introduce errors with the reinsertion as word order often changes when translating between languages. Could be improved by a more sophisticated algorithm, but leaving it alone for now. 

### Do Not Translate Tags
The `TransIns` class instantiates an attribute `dnt` which stores a list of tags that are not meant to be translated. By default, `<script>` and `<style>` tags are included. Any content that is wrapped in these tags will not be treated as text, while the tag itself will be treated the same way as a self closing tag.

```python
transins.reset()

markup = "<p>this is <script>console.log('DO NOT TRANSLATE');</script>markup</p>"
transins.extract_markup(markup, lang = "en")

assert transins.tag_id_map == {1: "<p>", 2: "<script>console.log('DO NOT TRANSLATE');</script>"}
assert transins.tokens == ["this", "is", "markup"]
assert transins.no_token_tags == {2: [2]}

# You can also add your own tags to dnt

transins.reset()

transins.dnt.append("dnt")

markup = "<p>this is <dnt>DO NOT TRANSLATE</dnt>markup</p>"

transins.extract_markup(markup)

assert transins.tag_id_map == {1: "<p>", 2: "<dnt>DO NOT TRANSLATE</dnt>"}
assert transins.tokens == ["this", "is", "markup"]
assert transins.no_token_tags == {2: [2]}
```

### Untokened Tags
Sometimes, tags that are neither self closing nor are in the `dnt` list still have no children. For instance, `<span>` tags are often used simply to contain an `<img>` tag. During the traversal, if all of a tag's children do not return any text content, the algorithm will treat the tag and all of its children as an untokened tag. The algorithm then handles it the same way it handles self closing tags, assigning it to the next occuring token.

### Tokenizers and Language
The CMS algorithm relies on the concept of tokens to map tags from source to target. As such we need to have tokenize the text that is present in the markup. Unfortunately, tokenizers are not language agnostic, so users need to ensure that the tokenizers loaded support the languages they are trying to translate. If not specified, MosesTokenizer is used via the sacremoses package. Supported languages are set to `en` (English) and `es` (Spanish) by default. You can add more if you trust Moses to deal with the language you want. Moses seems to support many languages but in practice there seems to be some issues with [certain languages](https://github.com/hplt-project/sacremoses/issues/42).

```python
# Add italian
transins.tokenizer.language_tokenizer_map["en"].languages.append("it") 
transins.tokenizer.language_tokenizer_map["it"] = transins.tokenizer.language_tokenizer_map["en"]
transins.tokenizer.languages.append("it")

# Or instantiate the MosesTokenizer yourself

from pytransins.tokenizer import MosesTokenizer
from pytransins.transins import TransIns

tokenizer = MosesTokenizer()
tokenizer.languages.append("it")

transins = TransIns(tokenizer)
```

You can also add your own tokenizer class by inheriting the base Tokenizer class. The `TransIns` class instantiates a `TokenizerGroup` class instance. This allows you to add new tokenizers, and hence more language support, without having to create a new `TransIns` class instance for every language. 
```python
from pytransins.tokenizer import Tokenizer

class MyTokenizer(Tokenizer):
    def __init__(self):
        super(MyTokenizer, self).__init__()
        self.languages = ["it"]
    
    def tokenize(self, text: str, lang: str):
        # Insert your code to tokenize here. Will raise NotImplementedError otherwise
        pass

    def detokenize(self, text: str, lang: str):
        # Insert your code to detokenize here. Will raise NotImplementedError if calling from this class directly. 
        # Will default to Moses if added to a TokenizerGroup
        pass

my_tokenizer = MyTokenizer()

from pytransins.transins import TransIns

transins = TransIns()
transins.tokenizer.add_tokenizer(tokenizer)
```

## Metrics
This library comes with an evaluation method based on the [Zhang-Shasha tree edit distance algorithm](https://github.com/timtadh/zhang-shasha).

```python
input_markup = "<h>this is <b>markup</b></h>"
output_markup = " <h> this is <b> markup </b></h> "

from pytransins.utils import compare_markup

tree_edit_distance = compare_markup(input_markup, output_markup)

assert tree_edit_distance == 2
```

You can also use the `test` method that comes with the `TransIns` class. This will extract and immediately reinsert the markup, returning the result. It will also log information like the tokens found, the tag ID to tag map, and the tree edit distance score. 

```python
from pytransins.transins import TransIns

transins = TransIns()

input_markup = "<h>this is <b>markup</b></h>"

result = transins.test(input_markup)

assert result == "<h>this is <b>markup</b></h>"
```

## Known Issues
### Untokened tags
As mentioned above, tokens with no tags are assigned to the token that occurs immediately after. This can cause poor migration of such tags as word ordering is not preserved on translation. One way to circumvent the problem is to not rely on the algorithm to handle such tags. This can be done simply by splitting up the document such that you only use the markup reinsertion on chunks that do not have tags that aren't meant to be translated. Having said that, even with the suboptimal approach adopted in this library, with a reasonable word aligner, the untokened tags are never too far away from where they should be.

### Detokenizing around tags
The tokenizer class comes with a detokenizer, but this only works for plaintext. To avoid having tokens clump together around tags, the reinsertion algorithm adds whitespaces after closing tags. The algorithm also adds a whitespace before opening tags and untokened tags if it detects that there is no whitespace before it. This is problematic for languages that do not use whitespace separators like Chinese. However, adding whitespaces to such languages does not hurt the readability of the output as much as not having whitespaces for languages that should.

### Tag ordering
Tag IDs are assigned such that tags always have IDs smaller than that of their child tags. Additionally, valid markup necessitates that tags are opened and closed in a stack. Tags that were opened first, must be closed last. Using these 2 facts, we can guarantee that tags are opened and closed in the correct order. This however does not guarantee that untokened tags are assigned correctly. The reinsertion algorithm hard codes the order to insert untokened tags after closing tags and opening tags.

### Unhandled tags
Some tags are not handled. These are just dropped altogether during the extraction process. Anything `bs4` element that does not have a `contents` attribute will be dropped. The exceptions are `NavigableString` and `Comment` tags. `NavigableString` is a text node, so these are the source of tokens for tagging. Comments are handle as untokened tags. For other elements that lack a `contents` attribute, please handle these in a preprocessing step separately.
