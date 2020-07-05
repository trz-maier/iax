from iax.inferfaces import Engine, Input


class Space:
    def __init__(self, x: Input, engine: Engine, classifier):
        self._input = x
        self._engine = engine
        self._classifier = classifier
        self._engine.initialize(self.intput, self.classifier)

    @property
    def intput(self):
        return self._input

    @property
    def engine(self):
        return self._engine

    @property
    def classifier(self):
        return self._classifier

    @property
    def size(self) -> int:
        """
        :return: int search space size
        """
        return len(self._input.mask.flatten())

    def search(self, **kwargs):
        """
        Start search using given Engine
        :param kwargs: arguments to be passed to Engine
        :return: None
        """
        self._engine.search(**kwargs)
