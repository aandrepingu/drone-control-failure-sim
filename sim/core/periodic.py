type milliseconds_t = int  # noqa: PYI042
class Periodic:
    """
    Manages period and offset for sim modules.
    """
    def __init__(self, period: milliseconds_t, offset: milliseconds_t):
        self.period = period
        self.offset = offset
        self.current_time = self.offset

    def advance(self, time:milliseconds_t) -> bool:
        """
        Advance the periodic forward based on its frequency and offset.
        """
        if time - self.current_time >= self.period:
            self.current_time += self.period
            return True
        return False

    def reset(self):
        self.current_time = self.offset
