Overview
========

This simple library aims to collect entity-alignment benchmark datasets and make them easily available.

.. code-block:: python

    from sylloge import OpenEA
    ds = OpenEA()
    print(ds)
    # OpenEA(graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)
    print(ds.rel_triples_right.head())
    #                                        head                             relation                                    tail
    # 0   http://www.wikidata.org/entity/Q6176218   http://www.wikidata.org/entity/P27     http://www.wikidata.org/entity/Q145
    # 1   http://www.wikidata.org/entity/Q212675  http://www.wikidata.org/entity/P161  http://www.wikidata.org/entity/Q446064
    # 2   http://www.wikidata.org/entity/Q13512243  http://www.wikidata.org/entity/P840      http://www.wikidata.org/entity/Q84
    # 3   http://www.wikidata.org/entity/Q2268591   http://www.wikidata.org/entity/P31   http://www.wikidata.org/entity/Q11424
    # 4   http://www.wikidata.org/entity/Q11300470  http://www.wikidata.org/entity/P178  http://www.wikidata.org/entity/Q170420
    print(ds.attr_triples_left.head())
    #                                   head                                          relation                                               tail
    # 0  http://dbpedia.org/resource/E534644                http://dbpedia.org/ontology/imdbId                                            0044475
    # 1  http://dbpedia.org/resource/E340590               http://dbpedia.org/ontology/runtime  6480.0^^<http://www.w3.org/2001/XMLSchema#double>
    # 2  http://dbpedia.org/resource/E840454  http://dbpedia.org/ontology/activeYearsStartYear     1948^^<http://www.w3.org/2001/XMLSchema#gYear>
    # 3  http://dbpedia.org/resource/E971710       http://purl.org/dc/elements/1.1/description                          English singer-songwriter
    # 4  http://dbpedia.org/resource/E022831       http://dbpedia.org/ontology/militaryCommand                     Commandant of the Marine Corps

The gold standard entity links are stored as [eche](https://github.com/dobraczka/eche) ClusterHelper, which provides convenient functionalities:

.. code-block:: python

    print(ds.ent_links.clusters[0])
    # {'http://www.wikidata.org/entity/Q21197', 'http://dbpedia.org/resource/E123186'}
    print(('http://www.wikidata.org/entity/Q21197', 'http://dbpedia.org/resource/E123186') in ds.ent_links)
    # True
    print(('http://dbpedia.org/resource/E123186', 'http://www.wikidata.org/entity/Q21197') in ds.ent_links)
    # True
    print(ds.ent_links.links('http://www.wikidata.org/entity/Q21197'))
    # 'http://dbpedia.org/resource/E123186'
    print(ds.ent_links.all_pairs())
    # <itertools.chain object at 0x7f92c6287c10>

Most datasets are binary matching tasks, but for example the `MovieGraphBenchmark` provides a multi-source setting:

.. code-block:: python

    ds = MovieGraphBenchmark(graph_pair="multi")
    print(ds)
    MovieGraphBenchmark(backend=pandas,graph_pair=multi, rel_triples_0=17507, attr_triples_0=20800 rel_triples_1=27903, attr_triples_1=23761 rel_triples_2=15455, attr_triples_2=20902, ent_links=3598, folds=5)
    print(ds.dataset_names)
    ('imdb', 'tmdb', 'tvdb')

Here the [`PrefixedClusterHelper`](https://eche.readthedocs.io/en/latest/reference/eche/#eche.PrefixedClusterHelper) various convenience functions:

.. code-block:: python
    # Get pairs between specific dataset pairs
    print(list(ds.ent_links.pairs_in_ds_tuple(("imdb","tmdb")))[0])
    # ('https://www.scads.de/movieBenchmark/resource/IMDB/nm0641721', 'https://www.scads.de/movieBenchmark/resource/TMDB/person1236714')

    # Get number of intra-dataset pairs
    print(ds.ent_links.number_of_intra_links)
    # (1, 64, 22663)

You can get a canonical name for a dataset instance to use e.g. to create folders to store experiment results:

.. code-block:: python

   print(ds.canonical_name)
   # 'openea_d_w_15k_v1'


You can use `dask <https://www.dask.org/>`_ as backend for larger datasets:

.. code-block:: python

    ds = OpenEA(backend="dask")
    print(ds)
    # OpenEA(backend=dask, graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)

Which replaces pandas DataFrames with dask DataFrames.

Datasets can be written/read as parquet via `to_parquet` or `read_parquet`.
After the initial read datasets are cached using this format. The `cache_path` can be explicitly set and caching behaviour can be disable via `use_cache=False`, when initalizing a dataset.

Some datasets come with pre-determined splits:

.. code-block:: bash

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


some don't:

.. code-block:: bash

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


You can install sylloge via pip:

.. code-block:: bash

  pip install sylloge



.. toctree::
   :maxdepth: 2
   :caption: Contents


.. toctree::
   :maxdepth: 2
   :caption: Documentation

   sylloge API <source/apidoc>
