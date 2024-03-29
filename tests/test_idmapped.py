from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from strawman import dummy_triples

from sylloge.id_mapped import (
    IdMappedEADataset,
    PandasTrainTestValSplit,
    enhance_mapping,
)


@dataclass
class Example:
    rel_triples: Tuple[pd.DataFrame, pd.DataFrame]
    attr_triples: Tuple[pd.DataFrame, pd.DataFrame]
    ent_links: pd.DataFrame
    folds: Sequence[PandasTrainTestValSplit]
    dataset_names: Tuple[str, str]
    entity_mapping: Dict[str, int]
    rel_mapping: Dict[str, int]
    attr_rel_mapping: Dict[str, int]


def _create_simple_mapping(
    left_prefix: str, right_prefix: str, left_num: int, right_num: int
) -> Dict[str, int]:
    my_mapping = {f"{left_prefix}{idx}": idx for idx in range(left_num)}
    my_mapping.update(
        {f"{right_prefix}{idx}": left_num + idx for idx in range(right_num)}
    )
    return my_mapping


@pytest.fixture()
def example() -> Example:
    seed = 42
    left_ent_num = 8
    left_rel_num = 6
    left_attr_rel_num = 7
    right_ent_num = 9
    right_rel_num = 4
    right_attr_rel_num = 5
    left_entity_prefix = "l"
    right_entity_prefix = "r"
    left_rel_prefix = "leftrel"
    right_rel_prefix = "rightrel"
    left_attr_rel_prefix = "leftattrrel"
    right_attr_rel_prefix = "rightattrrel"

    left_rel = dummy_triples(
        length=13,
        num_entities=left_ent_num,
        num_rel=left_rel_num,
        entity_prefix=left_entity_prefix,
        relation_prefix=left_rel_prefix,
        seed=seed,
    )
    right_rel = dummy_triples(
        length=18,
        num_entities=right_ent_num,
        num_rel=right_rel_num,
        entity_prefix=right_entity_prefix,
        relation_prefix=right_rel_prefix,
        seed=seed,
    )
    left_attr = dummy_triples(
        length=9,
        num_entities=left_ent_num - 2,
        num_rel=left_attr_rel_num,
        relation_triples=False,
        entity_prefix=left_entity_prefix,
        relation_prefix=left_attr_rel_prefix,
        seed=seed,
    )
    right_attr_len = 14
    right_attr = dummy_triples(
        length=right_attr_len,
        num_entities=right_ent_num,
        num_rel=right_attr_rel_num,
        relation_triples=False,
        entity_prefix=right_entity_prefix,
        relation_prefix=right_attr_rel_prefix,
        seed=seed,
    )
    # add numerical attribute value to test non-string attributes
    right_attr.loc[right_attr_len] = [
        f"{right_entity_prefix}2",
        f"{right_attr_rel_prefix}4",
        123,
    ]
    # add entity only occuring in tail
    left_rel.loc[left_rel_num] = [
        f"{left_entity_prefix}1",
        f"{left_rel_prefix}1",
        f"{left_entity_prefix}8",
    ]
    left_ent_num += 1
    entity_links = pd.DataFrame(
        zip(set(left_rel["head"]), set(right_rel["head"])), columns=["left", "right"]
    )
    folds = [
        PandasTrainTestValSplit(
            train=entity_links[:3],
            test=entity_links[3:5],
            val=entity_links[5:],
        )
    ]

    entity_mapping = _create_simple_mapping(
        left_prefix=left_entity_prefix,
        right_prefix=right_entity_prefix,
        left_num=left_ent_num,
        right_num=right_ent_num,
    )
    rel_mapping = _create_simple_mapping(
        left_prefix=left_rel_prefix,
        right_prefix=right_rel_prefix,
        left_num=left_rel_num,
        right_num=right_rel_num,
    )
    attr_rel_mapping = _create_simple_mapping(
        left_prefix=left_attr_rel_prefix,
        right_prefix=right_attr_rel_prefix,
        left_num=left_attr_rel_num,
        right_num=right_attr_rel_num,
    )

    return Example(
        rel_triples=(left_rel, right_rel),
        attr_triples=(left_attr, right_attr),
        ent_links=entity_links,
        folds=folds,
        dataset_names=("A", "B"),
        entity_mapping=entity_mapping,
        rel_mapping=rel_mapping,
        attr_rel_mapping=attr_rel_mapping,
    )


def test_enhance_mapping():
    full_range = 12
    given_mapping = {f"e{idx}": idx for idx in range(10)}
    labels = sorted({f"e{idx}" for idx in range(full_range)})
    enhanced_mapping = enhance_mapping(labels, given_mapping)
    assert enhanced_mapping == {f"e{idx}": idx for idx in range(full_range)}


def _assert_links(
    unmapped_links: pd.DataFrame,
    mapped_links: np.ndarray,
    entity_mapping: Dict[str, int],
):
    assert len(unmapped_links) == len(mapped_links)
    for unmapped, mapped in zip(unmapped_links.values, mapped_links):
        assert entity_mapping[unmapped[0]] == mapped[0]
        assert entity_mapping[unmapped[1]] == mapped[1]


def test_idmapped(example):
    id_mapped_ds = IdMappedEADataset.from_frames(
        rel_triples_left=example.rel_triples[0],
        rel_triples_right=example.rel_triples[1],
        attr_triples_left=example.attr_triples[0],
        attr_triples_right=example.attr_triples[1],
        ent_links=example.ent_links,
        folds=example.folds,
    )

    assert example.entity_mapping == id_mapped_ds.entity_mapping
    assert example.rel_mapping == id_mapped_ds.rel_mapping
    assert example.attr_rel_mapping == id_mapped_ds.attr_rel_mapping

    _assert_links(example.ent_links, id_mapped_ds.ent_links, example.entity_mapping)

    assert len(example.folds) == len(id_mapped_ds.folds)
    for unmapped_fold, mapped_fold in zip(example.folds, id_mapped_ds.folds):
        _assert_links(unmapped_fold.train, mapped_fold.train, example.entity_mapping)
        _assert_links(unmapped_fold.test, mapped_fold.test, example.entity_mapping)
        _assert_links(unmapped_fold.val, mapped_fold.val, example.entity_mapping)
