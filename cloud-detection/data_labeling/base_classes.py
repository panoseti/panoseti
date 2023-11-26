from abc import ABC, abstractmethod

# TODO: figure out common interfaces for batch and datasources. Don't use these yet.

class GenericDataBatchBuilder(ABC):
    def __init__(self, task, batch_id):
        raise NotImplementedError()

    @abstractmethod
    def build_batch(self, batch_def, ):
        raise NotImplementedError()


    def get_batch_dir(task, batch_id):
        return "task_{0}.batch-id_{1}".format(task, batch_id)

class DataSourceBase(ABC):

    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def init_file_tree(self):
        pass
    @abstractmethod
    def get_uid(self):
        pass
    @abstractmethod
    def create_features(self):
        pass
    @abstractmethod
    def build_batch(self):
        pass

class FeatureBuilderBase(DataSourceBase):

    def __init__(self):
        pass







