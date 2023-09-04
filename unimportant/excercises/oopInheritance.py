class Pet:
    def __init__(self, name):
        self.name = name

    def check_collar(self):
        print(self.name)

class Cat(Pet):
    def meow(self):
        print('Meow!')  

james = Cat('James')
james.check_collar()
james.meow()