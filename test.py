import sys
import random as r

fin = open("input.txt", "r")
f = open("scor.txt", "w")

number = r.randint(0, 3)
if number == 0 :
    f.write("false")
elif number == 1 :
    f.write("true")
elif number == 2 :
    f.write("partially false")
else:
    f.write("other")