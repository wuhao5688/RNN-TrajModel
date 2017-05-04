# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:01:05 2016
Modified on 2017/1/16

@author: wuhao
"""
from __future__ import print_function
import math
import os
import re
import numpy as np
import matplotlib.pyplot as plt

"""
#Attention
For all the parameters which are named as 'pos', the type must be Point(or its subclass), list, array, tuple, ndarray
or any type compatible (support indexing, e.g., pos[0] is x, pos[1] is y).
For rectangular coordinates, x is the first, y is the second.
For geo coordinates, lon is the first, lat is the second.
In the following docs, I'll use 'Pos type' to indicate the above stuff.
"""


class Point(object):
  """
  #Attributes
  x: float
  y: float

  #Description
  Class for 2D point, indexing (both getter and setter) supported. pt[0] is x and pt[1] is y
  """

  def __init__(self, pos):
    """
    #Arguments
    pos: Pos, the coordinate of the point
    """
    self.x = float(pos[0])
    self.y = float(pos[1])

  def dist(p1, p2):
    """
    #Arguments
    p1, p2: Pos type

    #Description
    No checking for flag, no coodinate transfer. It is no other than the Euclidean distance formula.
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

  def printPt(self):
    print("(%f, %f)" % (self.x, self.y))

  def __getitem__(self, key):
    if key == 0:
      return self.x
    elif key == 1:
      return self.y
    else:
      raise Exception("[ERROR]: The index of Point can only be 0 or 1.")

  def __setitem__(self, key, value):
    if key == 0:
      self.x = value
    elif key == 1:
      self.y = value
    else:
      raise Exception("[ERROR]: The index of Point can only be 0 or 1.")


class GeoPoint(Point):
  """
  #Attributes
  x: float (derived)
  y: float (derived)
  lat: float
  lon: float
  time: int or any other type

  #Description
  class for points supporting geo coordinate.
  """

  AREA_LAT = None
  R = 6371004  # radius of the earth

  def __init__(self, pos, flag='rect', time=None, omit_area_lat_setting_check=False):
    """
    #Arguments
    pos: Pos type, the coordinate of the point
    flag: string,
          'rect' indicates the pos is represented in rectangular coordinate.
          'geo' indicates the pos is represented in geo coordinate.
    time: int or any other type
    omit_area_lat_setting_check: bool, set True to compulsively omit checking the ‘AREA_LAT’. Please do not touch this argument if you do not know what you are doing.

    #Description
    You can use either geo or rect coord to initialize the GeoPoint.
    If you convey the geo coord, the rect coord will be automatically computed and stored, and vise versa.

    #Attention
    Before initializing any GeoPoint, the static attribute AREA_LAT should be first set
    (the distance of 1 degree in lon always changes with different lats).
    """
    if (omit_area_lat_setting_check is False) and (self.AREA_LAT is None):
      raise Exception(
        "[ERROR]: You should first set 'GeoPoint.AREA_LAT' or explicitly set the param 'omit_area_lat_setting_check' to be True.")
    if flag is 'rect':
      self.x = pos[0]
      self.y = pos[1]
      self.lat, self.lon = self.rect2geo(pos)
    elif flag is 'geo':
      self.lon = pos[0]
      self.lat = pos[1]
      self.x, self.y = self.geo2rect(pos)
    else:
      raise Exception("[ERROR]: pos.flag should only be 'rect' or 'geo'.")
    self.time = time

  @staticmethod
  def setAreaLat(lat):
    GeoPoint.AREA_LAT = lat

  @staticmethod
  def geo2rect(pos):
    x = math.pi * GeoPoint.R * math.cos(GeoPoint.AREA_LAT * math.pi / 180) / 180 * pos[0]
    y = math.pi * GeoPoint.R / 180 * pos[1]
    return x, y

  @staticmethod
  def rect2geo(pos):
    lon = pos[0] / (math.pi * GeoPoint.R * math.cos(GeoPoint.AREA_LAT * math.pi / 180) / 180)
    lat = pos[1] / (math.pi * GeoPoint.R / 180)
    return lon, lat

  def printPt(self):
    print("geo pos: (%f, %f)" % (self.lon, self.lat))
    print("rect pos: ", end='')
    super(GeoPoint, self).printPt()


class Line(object):
  """
  #Attributes
  p1: pos
  p2: pos
  """

  def __init__(self, pos1, pos2):
    self.p1 = Point(pos1)
    self.p2 = Point(pos2)


class Rectangle(object):
  """
  #Attributes
  minPt: Point, storing minX, minY
  maxPt: Point, storing maxX, maxY
  """

  def __init__(self, minPos, maxPos):
    self.minPt = Point(minPos)
    self.maxPt = Point(maxPos)

  def inRect(self, pos):
    return (pos[0] > self.minPt.x) and (pos[0] < self.maxPt.x) and (pos[1] > self.minPt.y) and (pos[1] < self.maxPt.y)

  def printRect(self):
    print("minPt: ",end='')
    self.minPt.printPt()
    print("maxPt: ", end='')
    self.maxPt.printPt()


class Area(Rectangle):
  """
  #Attributes
  minPt: GeoPoint, storing minX, minY
  maxPt: GeoPoint, storing maxX, maxY
  """

  def __init__(self, minPos=None, maxPos=None, flag='rect'):
    if minPos is None:
      return
    else:
      self.setArea(minPos, maxPos, flag)
    return

  def setArea(self, minPos, maxPos, flag='rect'):
    self.minPt = GeoPoint(minPos, flag)
    self.maxPt = GeoPoint(maxPos, flag)

  def inArea(self, pos):
    return self.inRect(pos)


class Polyline(object):
  """
  #Attributes
  shape: list<Point> or list<Pos> or None
  length: float or None   tips:getLen() will also return the length (no repeated computing)

  #Description
  class for list of Point/Pos or any compatible type
  """

  def __init__(self, shape=None, trans_to_Point=True):
    self.shape = shape
    self.length = None
    if shape is not None:
      if trans_to_Point and (isinstance(shape[0], Point) is False):
        self.shape = [Point(pos) for pos in shape]
      self.length = self.getLen()
    else:
      self.length = None;
    return

  def getLen(self):
    if self.length is not None:
      return self.length
    else:
      totLen = 0
      for i in range(len(self.shape) - 1):
        totLen += Point.dist(self.shape[i], self.shape[i + 1])
      return totLen


class Edge(Polyline):
  """
  #Attributes
  shape: list<GeoPoint> or None
  length: float or None   tips:getLen() will also return the length (no repeated computing) (derived)
  startNodeId: int
  endNodeId: int
  id: int
  adjList: list<Edge>, stores all the edges starting from map.nodes[self.endNodeId]
  adjList_ids: list<int>, stores all edge_ids starting from map.nodes[self.endNodeId]

  #Description
  class for edges used for class 'Map'
  """

  def __init__(self, shape=None, startNodeId=None, endNodeId=None, edgeId=None, adjList=None):
    if isinstance(shape[0], GeoPoint) is False:
      raise Exception("[ERROR]: type(shape[i]) should be 'GeoPoint'.")
    super(Edge, self).__init__(shape)
    self.startNodeId = startNodeId
    self.endNodeId = endNodeId
    self.id = edgeId
    self.adjList = adjList
    if self.adjList is not None:
      self.adjList_ids = [adjEdge.edgeId for adjEdge in self.adjList]


class Node(GeoPoint):
  """
  #Attributes
  x: float (derived)
  y: float (derived)
  lat: float (derived)
  lon: float (derived)
  id: int
  adjList: list<Edge>, stores all the edges starting from this node

  #Description
  class for vertexes used for class 'Map'
  """

  def __init__(self, Pos, flag, nodeId):
    super(Node, self).__init__(Pos, flag)
    self.id = nodeId
    self.adjList = []
    return


class Map(object):
  nodes = []
  edges = []

  def __init__(self, area=None):
    self.area = area
    return

  def open(self, mapFolderPath, gridSizeM=None, gridCount_in_width=None):
    """
    #Arguments
    mapFolderPath: string, the path of the folder containing 'edgeOSM.txt' and 'nodeOSM.txt'

    #Description
    Load OSM map file, standard format (old format not supported yet).
    """
    # load nodeOSM.txt
    # format: nodeId \t lat \t lon \n
    nodeFile = open((mapFolderPath + 'nodeOSM.txt'))# open(unicode(mapFolderPath + 'nodeOSM.txt', 'utf8'))
    nodeText = nodeFile.read()
    lines = nodeText.splitlines()
    count = 0
    for line in lines:
      values = re.split('[\n\t ]', line)
      nodeId, lat, lon = int(values[0]), float(values[1]), float(values[2])
      node = Node((lon, lat), 'geo', nodeId)
      if self.area is None or self.area.inArea(node):
        count += 1
      self.nodes.append(node)
    print("nodes count = %d" % len(self.nodes))
    print("nodes not in area = %d" % count)

    # load edgeOSM.txt
    # format: edgeId \t startNodeId \t endNodeId \t shape_len \t pt1.lat \t pt1.lon \t pt2.lat \t pt2.lon ...
    edgeFile = open(mapFolderPath + 'edgeOSM.txt')# open(unicode(mapFolderPath + 'edgeOSM.txt', 'utf8'))
    edgeText = edgeFile.read()
    lines = edgeText.splitlines()
    for line in lines:
      values = re.split('[\n\t ]', line)
      edgeId, startNodeId, endNodeId, shapeCount = int(values[0]), int(values[1]), int(values[2]), int(values[3])
      shape = [GeoPoint((float(values[i + 1]), float(values[i])), 'geo') for i in range(4, 4 + 2 * shapeCount, 2)]
      edge = Edge(shape, startNodeId, endNodeId, edgeId)
      self.edges.append(edge)
      # add topology info
      self.nodes[startNodeId].adjList.append(edge)

    # add topology info for each edge
    for edge in self.edges:
      edge.adjList = self.nodes[edge.endNodeId].adjList
      edge.adjList_ids = [adjEdge.id for adjEdge in edge.adjList]

    print("edges count = %d" % len(self.edges))

    nodeFile.close()
    edgeFile.close()
    return

  def drawMap(self):
    all_x = []
    all_y = []
    for edge in self.edges:
      xs = [pt.x for pt in edge.shape]
      ys = [pt.y for pt in edge.shape]
      # all_x.extend(xs)
      # all_y.extend(ys)
      plt.plot(xs, ys)
      # plt.plot(all_x, all_y)

  def hasEdge(self, startNodeId, endNodeId):
    """
    #Return
    Edge, if there exists the edge
    or None, if no such edge exists.
    """
    for edge in self.nodes[startNodeId].adjList:
      if edge.endNodeId == endNodeId:
        return edge
    return None

# GeoPoint.AREA_LAT = 2
# roadnet = Map()
# roadnet.open("F:/FTP/Data/OSM Maps/Singapore/new/")
# roadnet.open("D:/工作区/华为/MM_for_Huawei/Small_for_Huawei/")
# roadnet.drawMap()

# plt.plot([0,1,2,3,4], [0,2,1,6,8])
# plt.show()
# plt.savefig("D:/project/Python/Map/1.png", dpi=500)
# print('done')
