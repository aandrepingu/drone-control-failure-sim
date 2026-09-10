from abc import ABC, abstractmethod

from .periodic import Periodic, milliseconds_t


class SimModule(ABC):

    def __init__(self, period:int, offset:int):
        self.period=period
        self.offset=offset
        self.periodic = Periodic(self.period, self.offset)

    def Dispatch(self, current_time: milliseconds_t) -> None:
        if self.periodic.advance(current_time):
            self.HandleDispatch(current_time)

    @abstractmethod
    def HandleDispatch(self, current_time: int) -> None:
        """ Execute module logic. To be overridden by subclasses. """

    def reset(self):
        self.periodic.reset()