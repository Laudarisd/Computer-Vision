#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'filledOrders' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY order
#  2. INTEGER k
#
import os, sys
def filledOrders(order, k):
    if len(order)==0 or k==0:
        return 0
    order.sort()
    max_orders=0
    for item in order:
        if k<=0:
            return max_orders
        if item<=k:
            max_orders+=1
            k-=item
    return max_orders
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    order_count = int(input().strip())

    order = []

    for _ in range(order_count):
        order_item = int(input().strip())
        order.append(order_item)

    k = int(input().strip())

    result = filledOrders(order, k)

    fptr.write(str(result) + '\n')

    fptr.close()
