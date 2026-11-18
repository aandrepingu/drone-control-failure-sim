import time

from .sim_module import SimModule


class SimLoop:
    def __init__(self, viewer):
        self.modules: list[SimModule] = []
        self.viewer = viewer

    def run_until(self, condition_func):
        current_time = 0

        while condition_func():
            self.step(current_time)

    def run_forever(self):
        self.run_until(lambda: self.viewer.is_running())

    def step(self, current_time: int):
        for module in self.modules:
            module.Dispatch(current_time)

        # only sleep and sync if viewer is active and running
        if self.viewer and self.viewer.is_running():
            time.sleep(2 / 1000)
            # sync viewer every 16ms
            if current_time % 16 == 0:
                try:
                    self.viewer.sync()
                except Exception:
                    pass

    def close(self):
        if self.viewer is not None:
            viewer = self.viewer
            self.viewer = None
            try:
                viewer.close()
            except Exception:
                pass
            time.sleep(0.1)

    def step_n(self, n: int):
        count = 0

        def condition():
            nonlocal count
            if count < n:
                count += 1
                return True
            return False

        self.run_until(condition)

    def add_module(self, module: SimModule):
        self.modules.append(module)
