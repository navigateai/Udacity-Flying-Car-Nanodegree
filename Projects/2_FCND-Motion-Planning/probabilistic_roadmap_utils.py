from enum import Enum
from queue import PriorityQueue
import numpy as np
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString
import networkx as nx
from scipy.spatial import Voronoi
from bresenham import bresenham
from tqdm import tqdm


class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]
    
    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)

    def disjoint(self, other):
        return self._polygon.disjoint(other)



def extract_polygons(data, dist):

    polygons = []
    centers = []
    
    for i in tqdm(range(data.shape[0])):

        # print('obstacle {}/{}'.format(i,data.shape[0]))
        
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        
        obstacle = [north - d_north - dist, 
                    north + d_north + dist, 
                    east - d_east - dist, 
                    east + d_east + dist]
        
        corners = [(obstacle[0], obstacle[2]), 
                   (obstacle[0], obstacle[3]), 
                   (obstacle[1], obstacle[3]), 
                   (obstacle[1], obstacle[2])]
        
        height = alt + d_alt

        p = Poly(corners, height)
        
        polygons.append(p)
        centers.append((north,east))

    return (polygons,centers)




class Sampler:

    def __init__(self, data, alt, dist):
  
        # print('....getting polygons and centers')
        self._polygons, centers = extract_polygons(data, dist)

        # print('....setting limits')
        self._xmin = np.min(data[:, 0] - data[:, 3] - dist)
        self._xmax = np.max(data[:, 0] + data[:, 3] + dist)

        self._ymin = np.min(data[:, 1] - data[:, 4] - dist)
        self._ymax = np.max(data[:, 1] + data[:, 4] + dist)

        if isinstance(alt,int):
            self._zmin = alt
            self._zmax = alt
        else:
            self._zmin = alt[0]
            self._zmax = alt[1]


        # print('....computing max_poly_xy')
        # Record maximum polygon dimension in the xy plane
        # multiply by 2 since given sizes are half widths
        # This is still rather clunky but will allow us to 
        # cut down the number of polygons we compare with by a lot.
        self._max_poly_xy = 2 * np.max((data[:, 3], data[:, 4])) 
  
        # print('..getting centers')
        # centers = np.array([p.center for p in self._polygons])
 
        # print('....making center tree')
        self._tree = KDTree(centers, metric='euclidean')

        
    def sample(self, num_samples):
        """Implemented with a k-d tree for efficiency."""

        pts = []

        while len(pts)<num_samples:

            xvals = np.random.uniform(self._xmin, self._xmax, num_samples*2)
            yvals = np.random.uniform(self._ymin, self._ymax, num_samples*2)
            zvals = np.random.uniform(self._zmin, self._zmax, num_samples*2)
            samples = list(zip(xvals, yvals, zvals))

            for s in samples:

                in_collision = False
                
                #query for neighbors within a given radius
                idxs = list(self._tree.query_radius(np.array([s[0], s[1]]).reshape(1, -1), 
                                                    r=self._max_poly_xy)[0])
                if len(idxs) > 0:
                    for ind in idxs: 
                        p = self._polygons[int(ind)]
                        if p.contains(s) and p.height >= s[2]:
                            in_collision = True
                if not in_collision:
                    pts.append(s)

                if len(pts)>=num_samples:
                    return pts
                
        return pts

    @property
    def polygons(self):
        return self._polygons



#method to find the closest node in the graph
def closest_point(g, p):
    """
    Compute the closest point in the graph `g`
    to the current point `p`.
    """

    nodes = [n for n in g.nodes]

    tree = KDTree(nodes)
    idx = tree.query([p], k=1, return_distance=False)[0][0]
    return nodes[idx]



#method for creating the graph
def create_graph(nodes, k, sampler):

    g = nx.Graph()
    
    #tree of nodes
    node_tree = KDTree(nodes)

    #loop over nodes
    for n1 in tqdm(nodes):

        #find k nearest nodes
        node_idx = node_tree.query([n1], k, return_distance=False)[0]
        
        #loop over nearest nodes
        for i in tqdm(node_idx):

            #current nearest node
            n2 = nodes[i]

            #check if nodes are identical
            if not n1==n2:

                #check if edge already exists
                if not g.has_edge(n1, n2):

                    #assume nodes are connectable by default
                    is_connectable = True

                    #discrete points connecting the points
                    pts = list(bresenham(int(n1[0]), int(n1[1]), int(n2[0]), int(n2[1])))

                    #loop over a selection of points (every 10 meters) and check for collision with nearest polygon
                    for ind in range(0, len(pts), 20 ):

                        #current point on the line
                        p = pts[ind]

                        #find closest polygon to the current point
                        j = sampler._tree.query([p], 1, return_distance=False)[0][0]

                        #check if closest polygon and point of the line overlap
                        if sampler._polygons[j].disjoint(Point(p))==False:

                            #if they do, set connectable to false
                            is_connectable = False
                            break

                    #if connectable, add edge
                    if is_connectable:
                        g.add_edge(n1, n2, weight=1)

    return g



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))



#method for running A* on a graph
def a_star_graph(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""
    
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
            
    return path[::-1], path_cost



def path_pruning(path, epsilon=1e-3):
    
    i=0
    pruned_path=[p for p in path]
    while i < len(pruned_path) - 2:

        det = np.linalg.det( np.concatenate((
            np.array([pruned_path[i  ][0], pruned_path[i  ][1], 1.]).reshape(1, -1),
            np.array([pruned_path[i+1][0], pruned_path[i+1][1], 1.]).reshape(1, -1),
            np.array([pruned_path[i+2][0], pruned_path[i+2][1], 1.]).reshape(1, -1)
            ), 0))

        if abs(det) < epsilon:
            pruned_path.remove(pruned_path[i+1])
        else:
            i +=1
    pruned_path = [tuple(p) for p in pruned_path]

    return pruned_path



def waypoints_from_path(path, local_position, heading=True):
    '''
    Generates a list of waypoints [ [int(x), int(y), int(z), angle], ... ] from
    path:  3d grid coordinates in a list [ (x, y, z), ...]
    local_position:   current local position 
    heading:   whether or not to orient the quadrotor towards the next waypoint
    '''

    waypoints = []

    #loop over waypoints
    for i in range(len(path)):
        
        #previous location
        if i==0:  
            p0 = np.array([ local_position[0], local_position[1] ]) 
        else:
            p0 = np.array([ path[i-1][0], path[i-1][1] ]) 

        #current location
        p1 = np.array([ path[i][0], path[i][1]])  

        #unit vector from translation from previous to current location
        v01 =  p1 - p0               
        v01 = v01 / np.linalg.norm(v01)

        #unit vector pointing into north direction
        vref = np.array([1.,0.])     
        vref = vref / np.linalg.norm(vref)

        #angle between translation vector and north (note: it's the smallest angle, between 0-pi radians or 0-180 deg)
        angle = np.arccos(np.dot(vref, v01))

        #check rotation direction: clockwise/positive (default) for translations to the east and
        #counterclockwise/negative for translations to the wests
        if v01[1]<0:
            angle = -angle

        #append waypoint locations (integers) and rotation angle (in radians, relative to north)
        waypoints.append( tuple([int(path[i][0]), int(path[i][1]), int(path[i][2]), angle]) )


    return waypoints


