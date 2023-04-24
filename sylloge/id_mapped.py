from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import EADataset
from .utils import fix_dataclass_init_docs


def _enhance_mapping(
    labels: Iterable, mapping: Mapping[str, int] = None
) -> Dict[str, int]:
    """Map labels with given mapping and enhance mapping if unseen labels are encountered.

    :param labels: Labels to map
    :param mapping: Know mappings
    :return: Enhanced mapping
    """
    mapping = {} if mapping is None else mapping
    new_id = len(mapping)
    enhanced_mapping = {**mapping}
    for label in labels:
        label = str(label)
        if label in mapping:
            enhanced_mapping[label] = mapping[label]
        else:
            enhanced_mapping[label] = new_id
            new_id += 1
    return enhanced_mapping


def _perform_map(
    triples: np.ndarray,
    head_map: Dict[str, int],
    rel_map: Dict[str, int],
    tail_map: Dict[str, int],
) -> np.ndarray:
    head_getter = np.vectorize(head_map.get)
    rel_getter = np.vectorize(rel_map.get)
    tail_getter = np.vectorize(tail_map.get)
    # have to use triples with start:end instead of heads/rels/tails variable
    # because this way we get array of shape (n,1) instead of (n,)
    head_column = head_getter(triples[:, 0:1])
    rel_column = rel_getter(triples[:, 1:2])
    tail_column = tail_getter(triples[:, 2:3])
    return np.concatenate([head_column, rel_column, tail_column], axis=1)


def _id_map_rel_triples(
    df: pd.DataFrame,
    entity_mapping: Dict[str, int] = None,
    rel_mapping: Dict[str, int] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Map entity and relation labels to ids and create numpy array.

    :param df: labeled triples
    :param entity_mapping: already mapped entities
    :param rel_mapping: already mapped relations
    :return: id-based numpy array triples, (updated) entity label to id mapping, (updated) relation label to id mapping
    """
    triples = df.astype(str).values
    heads, rels, tails = triples[:, 0], triples[:, 1], triples[:, 2]
    # sorting  ensures consistent results
    entity_labels = sorted(set(heads).union(tails))
    relation_labels = sorted(set(rels))
    entity_mapping = _enhance_mapping(entity_labels, entity_mapping)
    rel_mapping = _enhance_mapping(relation_labels, rel_mapping)
    return (
        _perform_map(triples, entity_mapping, rel_mapping, entity_mapping),
        entity_mapping,
        rel_mapping,
    )


def _id_map_attr_triples(
    df: pd.DataFrame,
    entity_mapping: Dict[str, int],
    attr_rel_mapping: Dict[str, int] = None,
    attr_mapping: Dict[str, int] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Map entity, relation labels and attributes to ids and create numpy array.

    :param df: labeled triples
    :param entity_mapping: already mapped entities
    :param attr_rel_mapping: already mapped attribute relations
    :param attr_mapping: already mapped attributes
    :return: id-based numpy array triples, (updated) entity label to id mapping, (updated) relation label to id mapping, (updated) attribute to id mapping
    """
    triples = df.astype(str).values
    heads, rels, tails = triples[:, 0], triples[:, 1], triples[:, 2]
    # sorting  ensures consistent results
    entity_labels = sorted(set(heads))
    relation_labels = sorted(set(rels))
    attributes = sorted(set(tails))
    entity_mapping = _enhance_mapping(entity_labels, entity_mapping)
    rel_mapping = _enhance_mapping(relation_labels, attr_rel_mapping)
    attr_mapping = _enhance_mapping(attributes, attr_mapping)
    return (
        _perform_map(triples, entity_mapping, rel_mapping, attr_mapping),
        entity_mapping,
        rel_mapping,
        attr_mapping,
    )


def _map_links(links: pd.DataFrame, entity_mapping: Dict[str, int]) -> np.ndarray:
    """Map links via given mapping.

    :param links: entity links
    :param entity_mapping: label to id mapping
    :return: numpy array with ids
    """
    tuples = links.values
    entity_getter = np.vectorize(entity_mapping.get)
    return np.concatenate(
        [entity_getter(tuples[:, 0:1]), entity_getter(tuples[:, 1:2])], axis=1
    )


@fix_dataclass_init_docs
@dataclass
class IdMappedTrainTestValSplit:
    """Dataclass holding split of gold standard entity links."""

    #: entity links for training
    train: np.ndarray
    #: entity links for testing
    test: np.ndarray
    #: entity links for validation
    val: np.ndarray


@fix_dataclass_init_docs
@dataclass
class IdMappedEADataset:
    """Dataclass holding information of the alignment class with mapping of string to numerical id."""

    #: relation triples of left knowledge graph
    rel_triples_left: np.ndarray
    #: relation triples of right knowledge graph
    rel_triples_right: np.ndarray
    #: attribute triples of left knowledge graph
    attr_triples_left: np.ndarray
    #: attribute triples of right knowledge graph
    attr_triples_right: np.ndarray
    #: gold standard entity links of alignment
    ent_links: np.ndarray
    #: label to id mapping for all entities
    entity_mapping: Dict[str, int]
    #: label to id mapping for all relations
    rel_mapping: Dict[str, int]
    #: label to id mapping for all attribute relations
    attr_rel_mapping: Dict[str, int]
    #: attribute to id mapping for all attributes
    attr_mapping: Dict[str, int]
    #: optional pre-split folds of the gold standard
    folds: Optional[Sequence[IdMappedTrainTestValSplit]] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)}, ent_links={len(self.ent_links)}, entity_mapping={len(self.entity_mapping)}, rel_mapping={len(self.rel_mapping)}, attr_rel_mapping={len(self.attr_rel_mapping)}, attr_mapping={len(self.attr_mapping)}, folds={len(self.folds) if self.folds else None})"

    @classmethod
    def from_ea_dataset(cls, dataset: EADataset) -> "IdMappedEADataset":
        rel_triples_left, entity_mapping, rel_mapping = _id_map_rel_triples(
            dataset.rel_triples_left
        )
        (rel_triples_right, entity_mapping, rel_mapping,) = _id_map_rel_triples(
            dataset.rel_triples_right,
            entity_mapping=entity_mapping,
            rel_mapping=rel_mapping,
        )
        (
            attr_triples_left,
            entity_mapping,
            attr_rel_mapping,
            attr_mapping,
        ) = _id_map_attr_triples(
            dataset.attr_triples_left, entity_mapping=entity_mapping
        )
        (
            attr_triples_right,
            entity_mapping,
            attr_rel_mapping,
            attr_mapping,
        ) = _id_map_attr_triples(
            dataset.attr_triples_right,
            entity_mapping=entity_mapping,
            attr_rel_mapping=attr_rel_mapping,
            attr_mapping=attr_mapping,
        )

        ent_links = _map_links(dataset.ent_links, entity_mapping)
        new_folds = None
        if dataset.folds:
            new_folds = []
            for fold in dataset.folds:
                train = _map_links(fold.train, entity_mapping)
                test = _map_links(fold.test, entity_mapping)
                val = _map_links(fold.val, entity_mapping)
                new_folds.append(
                    IdMappedTrainTestValSplit(train=train, test=test, val=val)
                )
        return cls(
            rel_triples_left=rel_triples_left,
            rel_triples_right=rel_triples_right,
            attr_triples_left=attr_triples_left,
            attr_triples_right=attr_triples_right,
            ent_links=ent_links,
            entity_mapping=entity_mapping,
            rel_mapping=rel_mapping,
            attr_rel_mapping=attr_rel_mapping,
            attr_mapping=attr_mapping,
            folds=new_folds,
        )
