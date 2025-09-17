# use a "math" for area calculation
# Defining Vacuumcleaner class and its attributes(name,shape.....)
# a code for calculating areas of circular,square,triangle and d-shape. using if,elif,else condition
# make a code for commands (right,left,start,stop,dock)
# describe the vacuumcleaner details(name,shape.....)
# describe advanatges of shape
# Create a objects of VacuumCleaner 
# make a code for user choice(for choice like shape command)

import math

class VacuumCleaner:
    def __init__(self, name, shape, cleaning_diameter):
        self.name = name
        self.shape = shape
        self.cleaning_diameter = cleaning_diameter # cm
        self.state = "docked"
        self.area = self.calculate_area()

    def calculate_area(self):
        d = self.cleaning_diameter
        if self.shape.lower() == "circular":
            return math.pi * (d/2)**2
        elif self.shape.lower() == "square":
            return d * d
        elif self.shape.lower() == "d-shape":
            r = d / 2
            return (math.pi * r**2) / 2 + (d * r)
        elif self.shape.lower() == "triangle":
            return (math.sqrt(3)/4) * d**2
        else:
            return 0

    def process_command(self, command):
        commands = {
            "start": f"{self.shape} vacuum started cleaning.",
            "stop": f"{self.shape} vacuum stopped.",
            "left": f"{self.shape} vacuum turned left.",
            "right": f"{self.shape} vacuum turned right.",
            "dock": f"{self.shape} vacuum returned to dock.",
        }
        print(commands.get(command, f"{self.shape} received unknown command."))

    def describe(self):
        print(f"\nName: {self.name}\nShape: {self.shape}\nCleaning Diameter: {self.cleaning_diameter}cm")
        print(f"Covered Area: {self.area:.2f} sq.cm")
        print(f"Best Use: {self.get_advantage()}\n")

    def get_advantage(self):
        advantages = {
            "circular": "Best for open spaces.",
            "square": "best for rectangular rooms.",
            "d-shape": "Best for edge cleaning and open areas.",
            "triangle": "best for sharp cleaners.",
        }
        return advantages.get(self.shape.lower(), "General use.")


vacuums = [
    VacuumCleaner("Circ", "Circular", int(input("enter area for circular"))),
    VacuumCleaner("squa", "Square",int(input("enter arae for square"))),
    VacuumCleaner("Del", "D-shape", int(input("enter area for d-shape"))),
    VacuumCleaner("Tri", "Triangle",int(input("enter area for triangle"))),
]


print("Select a vacuum by number:")
for i, v in enumerate(vacuums):
    print(f"{i+1}: {v.name} ({v.shape})")
choice = int(input("Enter 1-4: ")) - 1
vacuum = vacuums[choice]
vacuum.describe()

print("Enter commands (start/stop/left/right/dock), 'exit' to quit.")
while True:
    cmd = input("Command: ").strip().lower()
    if cmd == "exit":
        print("Exiting manual control.")
        break
    vacuum.process_command(cmd)

 

    









        
