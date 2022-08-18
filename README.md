<h2 align="center">sylloge</h2>

<p align="center">
<a href="https://github.com/dobraczka/sylloge/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/sylloge/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This simple library aims to collect entity-alignment benchmark datasets and make them easily available.

Usage
=====
```
>>> from sylloge import OpenEA
>>> OpenEA()
OpenEA(graph_pair=D_W, size=15K, version=V1, rel_triples_left=38265, rel_triples_right=42746, attr_triples_left=52134, attr_triples_right=138246, ent_links=15000, folds=5)
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
