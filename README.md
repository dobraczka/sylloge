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
```
>>> from sylloge import OpenEA
>>> ds = OpenEA()
>>> ds
OpenEA(graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)
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

Installation
============
```bash
pip install sylloge 
```

Datasets
========
| Dataset family name | Year | # of Datasets | Sources | Authors | Reference |
|:--------------------|:----:|:-------------:|:-------:|:--------|:----------|
| OpenEA | 2020 | 16 | DBpedia, Yago, Wikidata | Zun, S. et. al. | [Paper](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf) |
| MovieGraphBenchmark | 2022 | 3 | IMDB, TMDB, TheTVDB | Obraczka, D. et. al. | [Paper](http://ceur-ws.org/Vol-2873/paper8.pdf) |
