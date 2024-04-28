#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'nearlySimilarRectangles' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts 2D_LONG_INTEGER_ARRAY sides as parameter.
#

def nearlySimilarRectangles(sides):
    # Write your code here
    LONG_INTEGER = 0
    for i in range(0, len(sides)-1):
        x = sides[i]
        for j in range(i+1, len(sides)):
            y = sides[j]
            
            if x[0]/y[0] == x[1]/y[1]:
                LONG_INTEGER = LONG_INTEGER +1
    return LONG_INTEGER
            
if __name__ == '__main__':
