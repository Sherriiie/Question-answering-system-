#!/usr/bin python3.4
# -*- coding: UTF-8 -*-

import numpy as np
import random
import os

# 取前4000个问题答案对作 validation/test datasets.
f1=open('qaSet.txt','r')
f2=file('val','a+')
f3=open('train', 'a+')
j=int(0)
for line in f1:
    if j < 40:
	f2.write(line)
	j += 1
    	# f3.write(line)
f2.flush()
f3.flush()
f1.close()	
f2.close()	
f3.close()	
	
