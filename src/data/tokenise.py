import itertools
from typing import Optional

from data.protein import Element, Protein


def tokenize(
    proteins: list[Protein], tokenizer: Optional[dict[Element, int]] = None
) -> dict[Element, int]:
    """
    Tokenize every item[key] in data.
    Modifies item[key] and copies original value to item[key_raw].
    @param (list) data
    @param (str) key  item[key] is iterable
    """
    if len(proteins) == 0:  # sometimes no val split, etc.
        return {}
    # if tokenizer is not provided, create index
    all_values = set(
        itertools.chain(*[protein.filtered_elements for protein in proteins])
    )
    if tokenizer is None:
        tokenizer = {}
    for item in all_values:
        if item not in tokenizer:
            tokenizer[item] = len(tokenizer) + 1  # 1-index

    for protein in proteins:
        protein.tokenize_sequence(tokenizer)
        # if not torch.is_tensor(item[key]):
        #     item[key] = torch.tensor(item[key])
    return tokenizer
