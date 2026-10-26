from abc import ABC, abstractmethod
import polars as pl


class BaseFactor(ABC):

    @abstractmethod
    def compute(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Build a lazy factor query without trading or evaluating future returns."""
        pass
