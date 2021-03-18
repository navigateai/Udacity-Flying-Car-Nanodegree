## Project: 3D Motion Planning
![Quad Image](./misc/enroute.png)

---


# Required Steps for a Passing Submission:
1. Load the 2.5D map in the colliders.csv file describing the environment.
2. Discretize the environment into a grid or graph representation.
3. Define the start and goal locations.
4. Perform a search using A* or other search algorithm.
5. Use a collinearity test or ray tracing method (like Bresenham) to remove unnecessary waypoints.
6. Return waypoints in local ECEF coordinates (format for `self.all_waypoints` is [N, E, altitude, heading], where the drone’s start location corresponds to [0, 0, 0, 0].
7. Write it up.
8. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1534/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

### Explain the Starter Code

#### 1. Explain the functionality of what's provided in `motion_planning.py` and `planning_utils.py`

I will describe the functionality of `plan_path()` and `planning_utils.py`, which are called after the quadrotor has been armed.
In `plan_path()`, two local variables specify the flight altitude and the minimum safety distance to obstacles in meters:

    TARGET_ALTITUDE = 5
    SAFETY_DISTANCE = 5

Obstacle information is read from the file `colliders.csv`:

    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

Next, a 2D grid representation of configuration space for the specified target altitude is created based on obstacle information from `data` and by adding the specified safety distance to obstacle borders. Cell size in the grid is 1x1 meters.

    grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

The variable `grid` represents the 2D configuration space, where free space is marked as False and occupied space as True. The variables `north_offset` and `east_offset` represent grid offset in local ECEF coordinates.

Next, the start and goal locations for the mission are specified by setting the start to grid center and goal 10 meters north and east of that location:

    grid_start = (-north_offset, -east_offset)
    grid_goal = (-north_offset + 10, -east_offset + 10)

After that, a path from start to goal is planned using A* :

    path, _ = a_star(grid, heuristic, grid_start, grid_goal)

whereby euclidian distance serves as a heuristic:

    def heuristic(position, goal_position):
        return np.linalg.norm(np.array(position) - np.array(goal_position))

and there are four possible actions: 1-m horizontal translations (to adjacent cells) in either west, east, north, or south direction:

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

No diagonal actions are allowed, and no path pruning is applied, thus the planned path follows a zig-zag trajectory with many waypoints leading from start to  goal.

After that, the planned path in 2D configuration space is been converted to 3D local coordinates, by adding `north_offset` and `east_offset` to x an y coordinates repectively, and using the `TARGET_ALTITUDE` for z. A heading angle of 0 (north direction) is used for all waypoints:

    waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]

Finally, the waypoints are send to the simulator and the drone transitions to takeoff:

    self.send_waypoints()





### Implementing Your Path Planning Algorithm

I implemented three solutions for the planning problem

(1) __Simple grid-based solution:__ 

This is the minimum requirement for passing the assignment, i.e. adding diagonal actions, setting start location to current location, setting goal location in geodetic coordinates, and adding path pruning using collinearity check. To run the code type in commandline:

    `python grid_astar.py`


(2) __Probabilistic roadmap:__ 

This is a graph-based solution using a 2.5D configuration space representation, random sampling, A* modified for graphs, and path pruning. To run the code type in commandline:

    `probabilistic_roadmap.py`


(3) __Receding horizon:__

This is probably the most interesting solution, which uses graph-based search for finding a coarse path (same as for probabilistic roadmap), and while executing the path, performs local obstacle detection and re-planning if needed. I will answer the rubric points below regarding this solution. To run the code type in commandline:

    `python receding_horizon.py`



#### 1. Set your global home position
Here students should read the first line of the csv file, extract lat0 and lon0 as floating point values and use the self.set_home_position() method to set global home. Explain briefly how you accomplished this in your code.

I read the first line from `colliders.csv` as a string, split it and converted the latitude and longitude values to float as follows:

    (lat0, lon0) = [ np.float64(s.split(' ')[1]) for s in 
    np.loadtxt('colliders.csv', delimiter='\n', dtype='str')[0].split(', ') ]
        
After that, I used these values and a zero altitude (ground floor) to set the global home position as follows:

    self.set_home_position(lon0, lat0, 0.)



#### 2. Set your current local position
Here as long as you successfully determine your local position relative to global home you'll be all set. Explain briefly how you accomplished this in your code.

This can be done by typing `self.local_position` or by using the `global_to_local()` function :

    local_position = global_to_local((self._longitude, self._latitude, self._altitude), self.global_home)

    

#### 3. Set grid start position from local position
This is another step in adding flexibility to the start location. As long as it works you're good to go!

Since for the receding horizon solution I do not use grid-based configuration space, I initially set the start location to the current local position, and later on when the random sampling-based graph is defined, i search for the closest node in the graph to update the start location as follows:

    start = self.local_position      # setting start location to current location position in NED coordinates
    start = closest_point(g, start)  # searching for the closest node to start location in graph `g`

The function `closest_point()` is implemented in `receding_horizon_utils.py` and uses a KDTree of nodes for rapid query:

    def closest_point(g, p):
        """
        Compute the closest point in the graph `g`
        to the current point `p`.
        """
        
        nodes = [n for n in g.nodes]
           
        tree = KDTree(nodes)
        idx = tree.query([p], k=1, return_distance=False)[0][0]
        return nodes[idx]



#### 4. Set grid goal position from geodetic coords
This step is to add flexibility to the desired goal location. Should be able to choose any (lat, lon) within the map and have it rendered to a goal location on the grid.

Using geodetic longitude and latitude of some location, I convert these coordinates along with the negative (!) target altitude (because of the downward direction of the z-axis in NED coordinates) to local coordinates and subsequently search for the closest node in the graph to update the goal location as follows: 

    (lat, lon) = (37.793373, -122.398809)  #some location on california street
    goal = global_to_local((lon, lat, -TARGET_ALTITUDE), self.global_home)  #convert geodetic to local coordinates
    goal = closest_point(g, goal)  #find closest node to goal loaction the graph `g`



#### 5. Modify A* to include diagonal motion (or replace A* altogether)
Minimal requirement here is to modify the code in planning_utils() to update the A* implementation to include diagonal motions on the grid that have a cost of sqrt(2), but more creative solutions are welcome. Explain the code you used to accomplish this step.

This is not necessary for the graph-based solution (however see `grid_astar_utils.py` for a modification of grid-based astar). For my solution, i modified A* to work with graphs, and giving a weight of 1 for every transition to a connected node. The algorithm is implemented in `receding_horizon_utils.py`:

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

I am using euclidian distance as a heuristic:

    def heuristic(position, goal_position):
        return np.linalg.norm(np.array(position) - np.array(goal_position))



#### 6. Cull waypoints 
For this step you can use a collinearity test or ray tracing method like Bresenham. The idea is simply to prune your path of unnecessary waypoints. Explain the code you used to accomplish this step.


I wrote a function for removing waypoints using collinearity checks:

     path = path_pruning(path, epsilon=0.1)

And here the function implemented in `receding_horizon_utils.py`. I iteratively check three subsequent points for collinearity and given a certain threshold (epsilon) remove the middle point:

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

Of note: In my tests this has not been neither necessary nor effective, since graph-based paths using a sparse distribution of nodes are used for reducing computational cost.



### Execute the flight
#### 1. Does it work?

It works, well most of the time, depending on the chosen start and goal locations and the number of nodes chosen.


### Double check that you've met specifications for each of the [rubric](https://review.udacity.com/#!/rubrics/1534/view) points.
  
# Extra Challenges: Real World Planning

For an extra challenge, consider implementing some of the techniques described in the "Real World Planning" lesson. You could try implementing a vehicle model to take dynamic constraints into account, or implement a replanning method to invoke if you get off course or encounter unexpected obstacles.


I decided to implement a receding horizon solution. The key component is a novel local planning algorithm, called at regular intervals given the following sampling rate:

    self.sr_local_planner = 2 # local planner sampling rate in Hz

The function is called in `local_position_callback`, thus when position changes, and when time is due given the sampling rate as follows:

    def local_position_callback(self):
            
        [...]
             
        if self.flight_state == States.WAYPOINT:
              
          [...]
                 
          #this is the current time
          self.t1 = time.time()   
           
          #run local planning if time is due
          if self.t1 - self.t0 > 1 / self.sr_local_planner:   
                   
              #set past time to current time
              self.t0 = self.t1
                 
              #run local planning
              self.local_path_planning()


In the following i describe what happens in `local_path_planning()`.

Initially, a 2D grid representation of local space around the quadrotor is created, aclled `self.voxmap`, but note it is currently not 3D as the name may suggest. The variable `self.horizon`, specifies the x and y length in meters of the 2D grid, the object `self.sampler` contains the 2.5D global representation of obstacles on the map and `self.local_position` is the map center:

    #update local map
    self.voxmap = update_local_map(self.horizon, self.horizon,, self.local_position, self.sampler)

The function `update_local_map()` is implemented in `receding_horizon_utils.py` and creates a grid representing 2D space around the quadrotor, where free space is False and occupied space is True.

Next, a `local_target` is defined, which is a 2D vector from the current location to the next target location.

Afte that the function `local_path_obstacle_detection()` (implemented in `receding_horizon_utils.py`) is called, which uses the 2D local grid and the bresenham algorithm for checking if in the current trajectory toward the next target_position there are any obstacles. It returns `in_collision`=True if there are any obstacles, and also returns the local path towards the next target position:

    (in_collision, local_path, _) = local_path_obstacle_detection(self.voxmap, local_target)

If no collisions are detected, nothing will be done, and the local planner is finished.

However, if there is a collision with the local path, an interative search is started for chaning the local path. Thus the linear path in collision with an obstacle is iteratively rotated around the center point until it is no longer in collision, and then this new waypoint is added to the list of waypoints

For development purposes, i made a command-line visualization of the current 2D voxmap, which shows obstacles as `o`, free space as `.`, the current position as `X` and the current local path as `P`. The visualization is called with the function `print_local_map(self.voxmap, local_path)` implemented in `receding_horizon_utils.py`. And here an example of such a local map (pretty nice!):

    ....oooooooooooooooooooooooooooooooooooo
    ....oooooooooooooooooooooooooooooooooooo
    ....oooooooooooooooooooooooooooooooooooo
    ....oooooooooooooooooooooooooooooooooooo
    ....oooooooooooooooooooooooooooooooooooo
    ..............oooooooooooooooooooooooooo
    ..............oooooooooooooooooooooooooo
    P.............oooooooooooooooooooooooooo
    .P............oooooooooooooooooooooooooo
    ..PP..........oooooooooooooooooooooooooo
    ....P.........oooooooooooooooooooooooooo
    .....PP.......oooooooooooooooooooooooooo
    .......PP.....oooooooooooooooooooooooooo
    .........P....oooooooooooooooooooooooooo
    ..........PP..oooooooooooooooooooooooooo
    ............P...........................
    .............PP.........................
    ...............PP.......................
    .................P......................
    ..................PP....................
    ....................X...................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................
    ........................................

By the way, i also added a heading command that orients the drone toward the next waypoint in the function `waypoints_from_path()` implemented in `receding_horizon_utils.py` and called as follows:

        self.waypoints = waypoints_from_path(path, self.local_position[:2], heading=True)


I have tested the alogrithm in many different scenarios. It seems that for graph-based planning in a satic environment there are very few collisions, thus the local planning may be beneficial for different (more dynamic) environments or using a more basice (e.g. breadth-first search) initial coarse path definition.

Hav fun trying it :)







