#Question check in Hackerrank webpage


import math

AB = float(input())
BC = float(input())

hyp = math.sqrt(AB**2 + BC**2)
theta = math.acos(BC/hyp)
degree=chr(176)
print(str(round(math.degrees(theta)))+degree)

