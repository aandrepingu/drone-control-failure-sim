from .sim_module import SimModule
import time


class SimLoop:
    def __init__(self, viewer):
        self.modules: list[SimModule] = []
        self.viewer = viewer

    def run_until(self, condition_func):
        current_time = 0

        while condition_func():
            self.step(current_time)
            current_time += 2
            time.sleep(2 / 1000)

            # sync viewer every 16ms
            if current_time % 16 == 0:
                self.viewer.sync()

    def run_forever(self):
        self.run_until(lambda: self.viewer.is_running())

    def step(self, current_time: int):
        for module in self.modules:
            module.Dispatch(current_time)

    def add_module(self, module: SimModule):
        self.modules.append(module)
