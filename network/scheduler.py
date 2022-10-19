from typing import Any


class Scheduler(object):
    def get_func(self, **kwargs) -> Any:
        pass


class Poly(Scheduler):
    @staticmethod
    def func(n: int, epochs: int, power: float) -> float:
        return (1 - n / epochs) ** power

    @staticmethod
    def get_func(epochs: int = 100, power: float = 0.9, **kwargs) -> Any:
        return lambda n: Poly.func(n, epochs, power)
