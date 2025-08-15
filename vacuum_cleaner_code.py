class Vacuumcleaner:
    def __init__(self, shape):
        self.shape = shape

    def command(self, action):
        if action=="start":
            self.start()
        elif action=="stop":
            self.stop()
        elif action=="left":
            self.left()
        elif action=="right":
            self.right()
        elif action=="dock":
            self.dock()
        else:
            print("Not Correct Command")

    def start(self):
        pass

    def stop(self):
        print(f"{self.shape} vacuum is stopped.")

    def left(self):
        print(f"{self.shape} vacuum is turned left.")

    def right(self):
        print(f"{self.shape} vacuum is turned right.")

    def dock(self):
        print(f"{self.shape} vacuum is returning to dock")


#for cirlce shape
class CircleVacuum(Vacuumcleaner):
    def __init__(self):
        super().__init__("circle")

    def start(self):
        print("circle vacuum starts clean")


#for square shape
class SquareVacuum(Vacuumcleaner):
    def __init__(self): 
        super().__init__("square")

    def start(self):
        print("square vacuum starts clean")


#for triangle shape
class TriangleVacuum(Vacuumcleaner):
    def __init__(self):
        super().__init__("triangle")

    def start(self):
        print("triangle vacuum starts clean")


#for rectangle shape
class RectangleVacuum(Vacuumcleaner):
    def __init__(self):
        super().__init__("rectangle")

    def start(self):
        print("rectangle vacuum starts clean")


"""
comamnds:
circle_robo=CircleVacuum()
circle_robo.command("start")
circle_robo.command("left")
circle_robo.command("stop")


square_robo=SquareVacuum()
square_robo.command("start")
square_robo.command("right")
square_robo.command("dock")................
"""


