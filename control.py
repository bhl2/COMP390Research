class Control(object):
    
    def __init__(self) -> None:
        pass

    def execute(self, sim):
        pass

class LiftControl(Control):
    def __init__(self, x, y):
        super(Control, self).__init__()
        self.x = x
        self.y = y
    
    def execute(self, sim, sleep_time=0.0):
        _ ,valid = sim.execute_alt(self.x, self.y, sleep_time=sleep_time)
        return valid 
class NormalControl(Control):
    def __init__(self, ctrl):
        super(Control, self).__init__()
        self.ctrl = ctrl
    
    def execute(self, sim, sleep_time=0.0):
        _, valid = sim.execute(self.ctrl, sleep_time=sleep_time)
        return valid 

class CompoundControl(Control):
    def __init__(self, c1 : Control, c2 : Control) -> None:
        super(Control).__init__()
        self.c1 = c1
        self.c2 = c2
    
    def execute(self, sim, sleep_time=0.0):
        valid1 = self.c1.execute(sim)
        if not valid1:
            return False
        valid2 = self.c2.execute(sim)
        return valid2 