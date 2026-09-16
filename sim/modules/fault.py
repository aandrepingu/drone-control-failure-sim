import numpy as np

from sim.core.sim_module import SimModule
from sim.core.state import ActuatorState, FaultStatus


class FaultModule(SimModule):
    """
    Applies health masks and failure profiles to motor commands.
    """

    def __init__(
        self,
        actuator_state: ActuatorState,
        fault_status: FaultStatus,
    ):
        super().__init__(period=2, offset=fault_status.fault_start_time or 0)
        self.actuator_state = actuator_state
        self.fault_status = fault_status

    def HandleDispatch(self, current_time: int) -> None:
        if not self.fault_status.fault_start_time or not self.fault_status.fault_active:
            return

        if current_time > self.fault_status.fault_start_time:
            # Multiply command by efficiency vector [1.0 = normal, 0.0 = complete failure]
            # e.g., actuator_effectiveness = [1.0, 0.5, 1.0, 0.0] -> Motor 1 @ 50%, Motor 3 Dead
            self.actuator_state.motor_commands *= (
                self.fault_status.actuator_effectiveness
            )
