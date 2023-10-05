<p align="center">
<img src="https://github.com/dobraczka/sylloge/raw/main/docs/logo.png" alt="sylloge logo", width=200/>
</p>

<h2 align="center">sylloge</h2>

<p align="center">
<a href="https://github.com/dobraczka/sylloge/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/sylloge/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a href='https://sylloge.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/sylloge/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://pypi.org/project/sylloge"/><img alt="Stable python versions" src="https://img.shields.io/pypi/pyversions/sylloge"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This simple library aims to collect entity-alignment benchmark datasets and make them easily available.

Usage
=====
Load benchmark datasets:
```
>>> from sylloge import OpenEA
>>> ds = OpenEA()
>>> ds
OpenEA(backend=pandas, graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)
>>> ds.rel_triples_right.head()
                                       head                             relation                                    tail
0   http://www.wikidata.org/entity/Q6176218   http://www.wikidata.org/entity/P27     http://www.wikidata.org/entity/Q145
1   http://www.wikidata.org/entity/Q212675  http://www.wikidata.org/entity/P161  http://www.wikidata.org/entity/Q446064
2   http://www.wikidata.org/entity/Q13512243  http://www.wikidata.org/entity/P840      http://www.wikidata.org/entity/Q84
3   http://www.wikidata.org/entity/Q2268591   http://www.wikidata.org/entity/P31   http://www.wikidata.org/entity/Q11424
4   http://www.wikidata.org/entity/Q11300470  http://www.wikidata.org/entity/P178  http://www.wikidata.org/entity/Q170420
>>> ds.attr_triples_left.head()
                                  head                                          relation                                               tail
0  http://dbpedia.org/resource/E534644                http://dbpedia.org/ontology/imdbId                                            0044475
1  http://dbpedia.org/resource/E340590               http://dbpedia.org/ontology/runtime  6480.0^^<http://www.w3.org/2001/XMLSchema#double>
2  http://dbpedia.org/resource/E840454  http://dbpedia.org/ontology/activeYearsStartYear     1948^^<http://www.w3.org/2001/XMLSchema#gYear>
3  http://dbpedia.org/resource/E971710       http://purl.org/dc/elements/1.1/description                          English singer-songwriter
4  http://dbpedia.org/resource/E022831       http://dbpedia.org/ontology/militaryCommand                     Commandant of the Marine Corps
>>> ds.ent_links.head()
                                  left                                    right
0  http://dbpedia.org/resource/E123186    http://www.wikidata.org/entity/Q21197
1  http://dbpedia.org/resource/E228902  http://www.wikidata.org/entity/Q5909974
2  http://dbpedia.org/resource/E718575   http://www.wikidata.org/entity/Q707008
3  http://dbpedia.org/resource/E469216  http://www.wikidata.org/entity/Q1471945
4  http://dbpedia.org/resource/E649433  http://www.wikidata.org/entity/Q1198381
```

You can get a canonical name for a dataset instance to use e.g. to create folders to store experiment results:

```
   >>> ds.canonical_name
   'openea_d_w_15k_v1'
```

Create id-mapped dataset for embedding-based methods:

```
>>> from sylloge import IdMappedEADataset
>>> id_mapped_ds = IdMappedEADataset.from_ea_dataset(ds)
>>> id_mapped_ds
IdMappedEADataset(rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, entity_mapping=30000, rel_mapping=417, attr_rel_mapping=990, attr_mapping=138836, folds=5)
>>> id_mapped_ds.rel_triples_right
[[26048   330 16880]
 [19094   293 23348]
 [16554   407 29192]
 ...
 [16480   330 15109]
 [18465   254 19956]
 [26040   290 28560]]
```

You can use [dask](https://www.dask.org/) as backend for larger datasets:
```
>>> ds = OpenEA(backend="dask")
>>> ds
OpenEA(backend=dask, graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)
```
Which replaces pandas DataFrames with dask DataFrames.

Datasets can be written/read as parquet via `to_parquet` or `read_parquet`.
After the initial read datasets are cached using this format. The `cache_path` can be explicitly set and caching behaviour can be disable via `use_cache=False`, when initalizing a dataset.

Some datasets come with pre-determined splits:

```bash
tree ~/.data/sylloge/open_ea/cached/D_W_15K_V1 
├── attr_triples_left_parquet
├── attr_triples_right_parquet
├── dataset_names.txt
├── ent_links_parquet
├── folds
│   ├── 1
│   │   ├── test_parquet
│   │   ├── train_parquet
│   │   └── val_parquet
│   ├── 2
│   │   ├── test_parquet
│   │   ├── train_parquet
│   │   └── val_parquet
│   ├── 3
│   │   ├── test_parquet
│   │   ├── train_parquet
│   │   └── val_parquet
│   ├── 4
│   │   ├── test_parquet
│   │   ├── train_parquet
│   │   └── val_parquet
│   └── 5
│       ├── test_parquet
│       ├── train_parquet
│       └── val_parquet
├── rel_triples_left_parquet
└── rel_triples_right_parquet
```
some don't:
```bash
tree ~/.data/sylloge/oaei/cached/starwars_swg
├── attr_triples_left_parquet
│   └── part.0.parquet
├── attr_triples_right_parquet
│   └── part.0.parquet
├── dataset_names.txt
├── ent_links_parquet
│   └── part.0.parquet
├── rel_triples_left_parquet
│   └── part.0.parquet
└── rel_triples_right_parquet
    └── part.0.parquet
```


Installation
============
```bash
pip install sylloge 
```

Datasets
========
| Dataset family name | Year | # of Datasets | Sources | References |
|:--------------------|:----:|:-------------:|:-------:|:----------|
| [OpenEA](https://sylloge.readthedocs.io/en/latest/source/datasets.html#sylloge.OpenEA) | 2020 | 16 | DBpedia, Yago, Wikidata |  [Paper](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf), [Repo](https://github.com/nju-websoft/OpenEA#dataset-overview) |
| [MED-BBK](https://sylloge.readthedocs.io/en/latest/source/datasets.html#sylloge.MED_BBK) | 2020 | 1 | Baidu Baike |  [Paper](https://aclanthology.org/2020.coling-industry.17.pdf), [Repo](https://github.com/ZihengZZH/industry-eval-EA/tree/main#benchmark) |
| [MovieGraphBenchmark](https://sylloge.readthedocs.io/en/latest/source/datasets.html#sylloge.MovieGraphBenchmark) | 2022 | 3 | IMDB, TMDB, TheTVDB | [Paper](http://ceur-ws.org/Vol-2873/paper8.pdf), [Repo](https://github.com/ScaDS/MovieGraphBenchmark) |
| [OAEI](https://sylloge.readthedocs.io/en/latest/source/datasets.html#sylloge.OAEI) | 2022 | 5 | Fandom wikis | [Paper](https://ceur-ws.org/Vol-3324/oaei22_paper0.pdf), [Website](http://oaei.ontologymatching.org/2022/knowledgegraph/index.html) |
