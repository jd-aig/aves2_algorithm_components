#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  
from utils import pickTopLeft

line1=[1,0,30,1,5,55,5,20,40,3,6,40,10,4]
p=np.array(line1).reshape(7, 2) 

points = Polygon(p).convex_hull
print(points)
points = np.array(points.exterior.coords)
print(points)
points = points[::-1]
print(points)

points = pickTopLeft(points)
print(points)
points = np.array(points).reshape([4, 2])
print(points)

poly = points
x_min = int(np.min(poly[:, 0]))
x_max = int(np.max(poly[:, 0]))

k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
b1 = poly[0][1] - k1 * poly[0][0]

k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
b2 = poly[3][1] - k2 * poly[3][0]
print(k1,b1,k2,b2)
res = []
r = 16
start = int((x_min // 16 + 1) * 16)
end = int((x_max // 16) * 16)
print(start)
print(end)
p = x_min
res.append([p, int(k1 * p + b1),
            start - 1, int(k1 * (p + 15) + b1),
            start - 1, int(k2 * (p + 15) + b2),
            p, int(k2 * p + b2)])
print(res)
for p in range(start, end + 1, r):
  res.append([p, int(k1 * p + b1),
              (p + 15), int(k1 * (p + 15) + b1),
              (p + 15), int(k2 * (p + 15) + b2),
               p, int(k2 * p + b2)])
print(res)
