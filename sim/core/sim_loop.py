from .sim_module import SimModule


class SimLoop:
    def __init__(self, viewer):
        self.modules = []
        self.viewer = viewer

    def run_until(self, condition_func):
        time = 0
        
        while(condition_func):
            self.step(time)
            time += 2
            time.sleep(2/1000)
            self.viewer.sync()

            
    def run_forever(self):
        self.run_until(lambda: True)

    def step(self, time):
        for module in self.modules:
            module.Dispatch(time)

    def add_module(self, module: SimModule):
        self.modules.append(module)