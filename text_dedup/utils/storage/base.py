from abc import abstractmethod


class StorageDict:  # pragma: no cover

    @abstractmethod
    def add(self, key, value):
        raise NotImplementedError("Storage dict must implement add method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Storage dict must implement __len__ method")

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError("Storage dict must implement __getitem__ method")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Storage dict must implement __iter__ method")
