import gc


class DisableReferenceCount:
    """
    A context manager to disable reference counting during the execution of a block.
    """

    def __enter__(self):
        # gc manipulations to ensure that uf object is not unneccessarily copied across processes
        gc.freeze()
        gc.disable()

    def __exit__(self, *args):
        gc.enable()
        gc.collect()
