"""
so langweilig, wie es klingt
"""
from random import randint

min_nr = int(input("gebe das minimum an:\n"))
max_nr = int(input("gebe das maximum an:\n"))

the_nr = randint(min_nr, max_nr)

a = 1

while True:
    inp_nr = int(input("tell me a number:\n"))
    if inp_nr == the_nr:
        print("you got it\n you needet ", a," tries to figure the number out")
        break
    elif inp_nr < the_nr:
        print("you are too low")
    else:
        print("you are too high")
    a += 1