import math
import sys
import re

number = "([0-9]+.[0-9]+e[+-][0-9]+)"
text = open("MMS_output.txt", "r").read()

match = re.findall("dx     " + number + "\nL2 norm \(ux-uxmms\) "
                         + number + "\nL2 norm \(uy-uymms\) "
                         + number + "\nL2 norm \(p-pmms\)   " + number, text)
print(match)

#scheme = []
dx = []
L2_norm_u0 = []
L2_norm_u1 = []
L2_norm_p = []

for i in range(len(match)):
    #scheme.append(float(match[i][0]))
    dx.append(float(match[i][0]))
    L2_norm_u0.append(float(match[i][1]))
    L2_norm_u1.append(float(match[i][2]))
    L2_norm_p.append(float(match[i][3]))

for name, err in [("u0: ", L2_norm_u0), ("u1: ", L2_norm_u1), ("p: ", L2_norm_p)]:
    for i in range(len(dx)-1):
        tmp = math.log(err[i]/err[i+1]) / math.log(dx[i]/dx[i+1])
        print(name, tmp)

    print()
