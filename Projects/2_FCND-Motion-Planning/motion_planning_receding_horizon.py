import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import math

from receding_horizon_utils import *
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        #global obstacle information (2.5D polygons)
        self.sampler = None 
        self.TARGET_ALTITUDE = 5  # target altitude in meters

        #local planner variables
        self.horizon = 40
        self.voxmap = np.zeros((self.horizon, self.horizon), dtype=np.bool) #2D configuration space (1m cell size)
        self.t1 = time.time() # current time in sec
        self.t0 = self.t1 # past time in sec
        self.sr_local_planner = 2 # local planner sampling rate in Hz
        self.deadband_waypoint = 10.0 #waypoint deadband for proceding to next waypoint
        self.deadband_local_position = 5 #local position deadband in meters, ingnorin obstacles in close proximity
        self.angle_increment = np.pi/9 #increment of the angle 

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):

        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()

        elif self.flight_state == States.WAYPOINT:

            ## check if local waypoint has been reached
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < self.deadband_waypoint:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

            ## local planning: 
            #update current time
            self.t1 = time.time() 
            #if local planning update is due
            if self.t1 - self.t0 > 1 / self.sr_local_planner:
                #set past time to current time
                self.t0 = self.t1
                #perform local planning
                self.local_path_planning()




    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(
            self.target_position[0], 
            self.target_position[1], 
            self.target_position[2], 
            self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING

        # Use more than 5m safety distance to work well with the simulator
        SAFETY_DISTANCE = 7

        # Set target altitude
        TARGET_ALTITUDE = self.TARGET_ALTITUDE

        # Set target position (really needed?)
        self.target_position[2] = TARGET_ALTITUDE

        # Read home latitude and longitude from colliders.csv
        (lat0, lon0) = [ np.float64(s.split(' ')[1]) for s in 
            np.loadtxt('colliders.csv', delimiter='\n', dtype='str')[0].split(', ') ]
       
        # Setting home position
        print('..setting gobal home to {} latitude, {} longitude, {} altitude'.format(lat0, lon0, 0.))
        self.set_home_position(lon0, lat0, 0.)

        #determine local position
        local_position = global_to_local((self._longitude, self._latitude, self._altitude), self.global_home)
        
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        print('..extract polygons')
        sampler = Sampler(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        self.sampler = sampler

        num_samples = 150
        print('..sample {} points and check for collisions'.format(num_samples))
        nodes = sampler.sample(num_samples)
        print('....sampled {} points in free space'.format(len(nodes)))

        num_connections = 15
        print('..create graph with up to {} edges per node'.format(num_connections))
        g = create_graph(nodes, num_connections, sampler)
        print('....graph has {} nodes and {} edges'.format(len(g.nodes), len(g.edges)))

        # Use local position as start location
        print('..setting start and goal location')
        start = self.local_position
        print('....start: {}'.format(start))

        # Set some goal location by longitude and latitude
        # (lat, lon) =(lat0, lon0) #home
        # (lat, lon) =(37.794821, -122.397882) #FEDEX
        # (lat, lon) =(37.794598, -122.396599) #Hyatt
        (lat, lon) =(37.793373, -122.398809) #California Street
        # (lat, lon) =(37.793685, -122.396311) #Embarcadero Station
        # (lat, lon) =(37.793171, -122.396678) #Peet's Coffee

        # Convert goal location from geodetic to grid frame
        goal = global_to_local((lon, lat, -TARGET_ALTITUDE), self.global_home)
        print('....goal:  {}'.format(goal))

        #find closest nodes for start and goal
        start = closest_point(g, start)
        goal = closest_point(g, goal)
        print('....start node: {}'.format(start))
        print('....goal node:  {}'.format(goal))

        #run a*
        print('..run a* on graph')
        path, _ = a_star_graph(g, heuristic, start, goal)
        if len(path)>0:
            path.append(goal)
        print('....path found by a*: {}\n'.format(path))

        # Prune path to reduce the number of waypoints
        path = path_pruning(path, epsilon=0.1)
        print('....path after pruning:\n{}'.format(path))

        # Convert path to waypoints
        waypoints = waypoints_from_path(path, self.local_position, heading=True)
        print('....waypoints:\n{}'.format(waypoints))

        # Send waypoints to sim (this is just for visualization of waypoints)
        self.waypoints = waypoints

        self.send_waypoints()


    def local_path_planning(self):

        #update local map
        self.voxmap = update_local_map(self.horizon, self.horizon, self.local_position, self.sampler)

        #set lcoal target as the local vector to target
        local_target = self.target_position[:2] - self.local_position[:2]

        #check for obstacles in local path
        (in_collision, local_path, _) = local_path_obstacle_detection(self.voxmap, local_target)

        #if no collision
        if in_collision:

            #current angle deviation is zero
            angle = 0

            #assume possible collision
            while in_collision:

                #update the angle
                if angle>=0:
                    angle += self.angle_increment
                    angle = -angle
                else:
                    angle = -angle

                #set lcoal target as the local vector to target
                local_target = self.target_position[:2] - self.local_position[:2]

                print('=============')
                print('collision detected: replanning local path')
                print('angle: ',angle)
                print('local_target (before rotation): ',local_target)

                #make rotation matrix
                (c, s) = np.cos(angle), np.sin(angle)
                R = np.array(((c,-s), (s, c)))

                #update the local target using the angle rotation
                local_target = np.dot(R, local_target)

                print('local_target (after rotation): ',local_target)

                #check for obstacles in local path
                (in_collision, local_path, _) = local_path_obstacle_detection(
                    self.voxmap, 
                    local_target, 
                    voxel_size=1, 
                    deadband=self.deadband_local_position)

                print('still in collision: ', in_collision)


            #print the local voxmap
            print_local_map(self.voxmap, local_path)


            #if the angle is non-zero (thus the target was updated) and new trajectory is not in collision
            if angle != 0 and not in_collision:

                #old waypoints
                path = [ [p[:3]] for p in self.waypoints ] 

                #add current target position to the front
                path[:0] = [[ 
                    self.target_position[0], 
                    self.target_position[1], 
                    self.TARGET_ALTITUDE ]]

                #add new target position to the front
                path[:0] = [[ 
                    self.local_position[0] - self.voxmap.shape[0]/2 + local_path[-1][0],
                    self.local_position[1] - self.voxmap.shape[1]/2 + local_path[-1][1], 
                    self.TARGET_ALTITUDE ]] 

                #compose waypoints and add heading
                self.waypoints = waypoints_from_path(path, self.local_position[:2], heading=True)

                self.waypoint_transition()

        #if in collision
        else:

            #just print the voxmap to prompt
            print_local_map(self.voxmap, local_path)





    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
