class Control(object):
    
    def __init__(self) -> None:
        pass

    def execute(self, sim):
        pass

class LiftControl(Control):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def execute(self, sim):
        sim.execute_alt(self.x, self.y)

class NormalControl(Control):
    def __init__(self, ctrl):
        super(self).__init__()
        self.ctrl = ctrl
    
    def execute(self, sim):
        sim.execute(self.ctrl)