Overview
========

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
    print(ds.ent_links.head())
    #                                   left                                    right
    # 0  http://dbpedia.org/resource/E123186    http://www.wikidata.org/entity/Q21197
    # 1  http://dbpedia.org/resource/E228902  http://www.wikidata.org/entity/Q5909974
    # 2  http://dbpedia.org/resource/E718575   http://www.wikidata.org/entity/Q707008
    # 3  http://dbpedia.org/resource/E469216  http://www.wikidata.org/entity/Q1471945
    # 4  http://dbpedia.org/resource/E649433  http://www.wikidata.org/entity/Q1198381


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
