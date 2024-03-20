=================
API Documentation
=================

This is the API documentation for ``sylloge``.

Datasets

.. autosummary::
   :nosignatures:

    sylloge.MovieGraphBenchmark
    sylloge.OpenEA
    sylloge.OAEI
    sylloge.MED_BBK


Base

.. autosummary::
   :nosignatures:

    sylloge.base.TrainTestValSplit
    sylloge.base.MultiSourceEADataset
    sylloge.base.ParquetEADataset
    sylloge.base.CacheableEADataset
    sylloge.base.ZipEADataset
    sylloge.base.ZipEADatasetWithPreSplitFolds
    sylloge.base.BinaryEADataset
    sylloge.base.BinaryParquetEADataset
    sylloge.base.BinaryCacheableEADataset
    sylloge.base.BinaryZipEADataset
    sylloge.base.BinaryZipEADatasetWithPreSplitFolds


IdMapped

.. autosummary::
   :nosignatures:

    sylloge.id_mapped.IdMappedTrainTestValSplit
    sylloge.id_mapped.IdMappedEADataset
