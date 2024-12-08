"""
@description Contains functionality for controlling UAVs and
interfacing with Sionna methods.
@date 11-8-2024
@author(s) Everett Tucker
"""

#TODO: Look at AERPAW user manual to cross-reference ideas for this simulation framework

import sionna
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.rt import Antenna, AntennaArray
from sionna.rt import Transmitter, Receiver

# The Earth's gravitational acceleration in m/s^2
GRAVITATIONAL_ACCEL = 9.80665
# Average Air Density Around the Area where the UAV Flies, kg / m^3
AIR_DENSITY = 1.213941
# The number of samples to use when calculating the energy consumption
NUM_INTEGRATION_SAMPLES = 1000


class Environment():
    # Tested!
    def __init__(self, scene_path, position_df_path, time_step=1, ped_height=1.5, ped_rx=True, wind_vector=np.zeros(3)):
        """
        Creates a new environment from a scene path and a position_df_path
        This method may take several minutes to run because of the scene creation

        Args:
            scene_path (str): the file path of the XML scene from Mitsuba
            position_df_pat (str): the file path of the csv position data
            time_step (float): the time step between simulation iterations
            ped_height (float): the assumed height of the pedestrians, in meters
            ped_rx (bool): if the pedestrians are receivers and the UAVs are transmitters, default true
            wind_vector (np.array(3,)): The velocity of the wind in the environment
        """
        print("Loading Scene")
        self.scene = sionna.rt.load_scene(scene_path)
        print("Parsing Positions")
        self.ped_rx = ped_rx
        self.time_step = time_step
        self.ped_height = ped_height
        self.n_rx = 0
        self.n_tx = 0
        self.uavs = {}
        self.gus = self.createGroundUsers(position_df_path)
        self.wind = wind_vector
        # This is used to speed up computation later in the simulation
        self.bezier_matrix = np.linalg.inv(np.array([
            [1, 0, 0, 0], 
            [-3, 3, 0, 0], 
            [(1 - self.time_step) ** 3, 3 * self.time_step * (1 - self.time_step) ** 2, 3 * self.time_step ** 2 * (1 - self.time_step), self.time_step ** 3], 
            [-3 * (1 - self.time_step) ** 2, -6 * self.time_step * (1 - self.time_step) + 3 * (1 - self.time_step) ** 2, -3 * self.time_step ** 2 + 6 * self.time_step * (1 - self.time_step), 3 * self.time_step ** 2]
        ]))

    # Tested
    def createGroundUsers(self, position_df_path):
        """
        Parses the positions data from SUMO and creates a list of ground user objects
        Also initializes the inital positions with receivers or transmitters

        Args:
            position_df_path (str): the file path to the pedestrian position dataframe
        
        Returns:
            np.array(): a list of ground user objects
        """

        df = pd.read_csv(position_df_path)
        i = 0
        res = []
        while (len(df.loc[df["person_id"] == "ped" + str(i)]) > 0):
            res.append(df.loc[df["person_id"] == "ped" + str(i)])
            i += 1
        
        # Creating the ground users
        rtn = []
        for j in range(len(res)):
            if self.ped_rx:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"]]).T, height=self.ped_height, com_type="rx"))
                self.n_rx += 1
            else:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"]]).T, height=self.ped_height, com_type="tx"))
                self.n_tx += 1
            self.scene.add(rtn[j].device)
                
        return np.array(rtn)
    
    # Tested!
    def advancePedestrianPositions(self):
        """
        Advances all the pedestrian positions to the next their next time step
        """

        for id in range(len(self.gus)):
            self.updateGroundUser(id)
    
    # Tested!
    def computeShortestDistance(self):
        """
        Computes the shortest distance between any two UAVs, used for collision avoidance
        I would use the O(nlogn) algorithm, but I think the constant time factor is way too high for our application
        This algorithm is better when len(self.uavs) < 125, which I think is basically always guarenteed

        Returns:
            the shortest squared distance between any two uavs, in m
        """
        rtn = math.inf
        keys = list(self.uavs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                rtn = min(rtn, 
                (self.uavs[keys[i]].pos[0] - self.uavs[keys[j]].pos[0]) ** 2 +
                (self.uavs[keys[i]].pos[1] - self.uavs[keys[j]].pos[1]) ** 2 +
                (self.uavs[keys[i]].pos[2] - self.uavs[keys[j]].pos[2]) ** 2)

        return math.sqrt(rtn)
            
    # Tested!
    def visualize(self):
        """
        Visualizes the current receivers and transmitters in the scene.
        """
        self.scene.preview(show_devices=True)


    def computeLOSLoss(self):
        """
        Computes the Line-of-sight path loss across all pairs of transmitters and receivers

        Returns:
            tf.tensor: A tensor describing the path loss across all pairs, in decibels
        """
        paths = self.scene.compute_paths(max_depth=0, method="exhaustive", num_samples=(self.n_rx * self.n_tx), los=True,
                                         reflection=False, diffraction=False, scattering=False, check_scene=False)

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs.keys()]), np.array([x.vel for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.vel for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs.keys()]))
        
        a, tau = tf.squeeze(paths).cir(los=True, reflection=False, diffraction=False, scattering=False, ris=False)
        return -20 * np.log10(np.abs(a))
    
    # Tested
    def addUAV(self, id, mass=1, efficiency=0.8, pos=np.zeros(3), vel=np.zeros(3), color=np.random.rand(3), rotor_area=None):
        """
        Adds a UAV to the environment and initalizes its quantities and receiver / transmitter

        Args:
            id (int): the unique id of the UAV
            mass (float): the mass of the UAV in kilograms
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV2
            color (np.array(3,)): the color of the UAV used for visualization
        """
        if rotor_area is None:
            self.uavs[id] = UAV(id, mass, efficiency, pos, vel - self.wind, self.time_step, mass * 0.3)
        else:
            self.uavs[id] = UAV(id, mass, efficiency, pos, vel - self.wind, self.time_step, rotor_area)

        if self.ped_rx:
            self.uavs[id].device = Transmitter(name=str(id), position=pos, color=color)
            self.n_tx += 1
        else:
            self.uavs[id].device = Transmitter(name=str(id), position=pos, color=color)
            self.n_rx += 1
        
        self.scene.add(self.uavs[id].device)

    # TODO: Deprecated
    def impulseUAV(self, id, force, wind_vector=np.zeros(3,)):
        """
        Administers the desired impulse to the UAV with the specified id, and updates its receiver / transmitter position

        Args:
            id (int): the unique id number of the UAV
            force (np.array(3,)): a vector of the force, in Newtons
            wind_vector (np.array(3,)): a vector of the wind velocity, in m/s
        """
        self.uavs[id].impulse(force, wind_vector)
        if self.ped_rx:
            self.scene._transmitters[str(id)].position = self.uavs[id].pos
        else:
            self.scene._receivers[str(id)].position = self.uavs[id].pos
    
    # Tested!
    def moveAbsUAV(self, id, abs_pos, abs_vel):
        """
        Moves the uav with the specified id to a a new absolution position and velocity
        Also updates the position of the communication device associated with the UAV

        Args:
            id (int): the unique id of the UAV
            abs_pos (np.array(3,)): the absolute position vector of the UAV after the move
            abs_vel (np.array(3,)): the absolute velocity vector of the UAV after the move
        """
        self.uavs[id].move(abs_pos, abs_vel - self.wind, self.bezier_matrix)
        self.uavs[id].device.position = abs_pos

    # Tested!
    def moveRelUAV(self, id, relative_pos):
        """
        Moves the UAV to a new position relative to its current position, maintains current velocity
        Also updates the position of the communication device associated with the UAV

        Args:
            id (int): the unique id of the UAV
            relative_pos (np.array(3,)): the relative position of the UAV
        """
        self.uavs[id].move(self.uavs[id].pos + relative_pos, self.uavs[id].vel, self.bezier_matrix)
        self.uavs[id].device.position = self.uavs[id].pos + relative_pos

    # Tested!
    def getUAVPos(self, id):
        """
        Gets the UAV position by Id

        Args:
            id (str): the unique id of the UAV

        Returns:
            np.array(3,): the position vector of the UAV
        """
        return self.uavs[id].pos

    # Tested!
    def getUAVVel(self, id):
        """
        Gets the absolute velocity of the UAV by Id

        Args:
            id (str): the unique id of the UAV
        
        Returns:
            np.array(3,): the absolute velocity of the UAV
        """
        return self.uavs[id].vel + self.wind
    

    def getUAVConsumption(self, id):
        """
        Gets the power consumption of the specified UAV, in joules

        Args:
            id (str): the unique id of the UAV

        Returns:
            float: the consumption of the specified UAV, in joules
        """
        return self.uavs[id].getConsumption()

    # Tested!
    def updateGroundUser(self, id):
        """
        Moves a ground user to the next avaliable position and updates velocity
        """
        self.gus[id].update()
        # Just update the x and y positions, the height stays constant
        self.gus[id].device.position = tf.constant([self.gus[id].pos[0], self.gus[id].pos[1], self.gus[id].height])

        """
        if self.ped_rx:
            self.scene._receivers[str(id)].position = self.gus[id].pos
        else:
            self.scene._transmitters[str(id)].position = self.gus[id].pos
        """
    

    def setTransmitterArray(self, arr):
        """
        Sets the scene's transmitter array to arr

        Args:
            arr the antenna array for the transmitters
        """
        self.scene.tx_array = arr

    
    def setReceiverArray(self, arr):
        """
        Sets the scene's receiver array to arr

        Args:
            arr the antenna array for the receivers
        """
        self.scene.rx_array = arr

    # Tested!
    def plotUAVs(self, length=None):
        """
        Plots the positions and velocities of all the current UAVs in 3 Dimensions

        Args:
            length (float): the length of the normalized velocity vectors, None if not normalized
        """
        axis = plt.figure().add_subplot(projection='3d')
        for uav in self.uavs.values():
            c = np.random.rand(3)
            axis.scatter(uav.pos[0], uav.pos[1], uav.pos[2], color=c)
            if not np.array_equal(uav.vel, np.zeros(3)):
                x, y, z = np.meshgrid(uav.pos[0], uav.pos[1], uav.pos[2])
                if length is None:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], color=c)
                else:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], length=length, normalize=True, color=c)
        plt.show()

    # Tested!
    def plotGUs(self):
        """
        Plots the Ground Users in 2D
        """

        for gu in self.gus:
            plt.scatter(gu.pos[0], gu.pos[1])
        plt.show()


    def getBezier(self, x_i, x_f, v_i, v_f, samples):
        """
        Returns a number of sample points along the Bezier curve from the given position-velocity pairs

        Args:
            x_i (np.array(3,)): The initial position vector, in meters
            x_f (np.array(3,)): The final position vector, in meters
            v_i (np.array(3,)): The initial velocity vector, in m/s
            v_f (np.array(3,)): The final velocity vector, in m/s
        
        Returns:
            np.array(samples,3): An array of points along the calculated Bezier curve
        """

        f = np.array([x_i, v_i, x_f, v_f])
        b = np.dot(self.bezier_matrix, f)

        rtn = []
        for i in range(samples):
            t = self.time_step * i / samples
            # Add the value of the parametric curve at time t
            rtn.append(((1 - t) ** 3) * b[0] + 3 * t * ((1 - t) ** 2) * b[1] + 3 * (t ** 2) * (1 - t) * b[2] + (t ** 3) * b[3])


        return np.array(rtn)


class UAV():
    def __init__(self, id, mass=1, efficiency=0.8, pos=np.zeros(3,), vel=np.zeros(3,), bandwidth=50, delta_t=1, rotor_area=0.5):
        """
        Creates a new UAV object with the specified physical and communication parameters

        Args:
            id (int): the unique id of the UAV
            mass (float): the mass of the UAV in kg
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV
            bandwidth (float): the bandwidth of the UAV's communication system, in Mbps
            color (np.array(3,)): the color of the UAV used for visualization
            com_type (str): either "tx" for transmitter or "rx" for receiver, TODO: add functionality for both at the same time?
            rotor_area (float): the total area of the UAV's rotors, in square meters
        """
        self.id = id
        self.mass = mass
        self.efficiency = efficiency
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.consumption = 0  # Cumulative consumption from movement, computation, and communication, in joules
        self.bandwidth = bandwidth
        self.rotor_area = rotor_area
        self.device = None  # Initialized later
        
    
    # TODO: DEPRECATED
    def impulse(self, force, wind_vector=np.zeros(3)):
        """
        Updates the UAV's position and velocity with a constant impulse over a time step
        Also updates the consumption with the new energy change
        This should be the primary method for moving the UAV

        Args:
            time_step (float): the time, in seconds, to send the impulse for. Should match the time_step of the environment
            force (np.array(3,)): the force, in Newtons, to apply to the UAV
            wind_vector (np.array(3,)): the vector of the wind, each component in m/s
        """

        v_f = force * self.delta_t / self.mass + self.vel + wind_vector
        x_f = (self.delta_t * self.delta_t / (2 * self.mass)) * force + self.vel * self.delta_t + wind_vector * self.delta_t
        delta_k_e = 1/2 * self.mass * np.dot(v_f - self.vel, v_f - self.vel)
        delta_p_e = self.mass * GRAVITATIONAL_ACCEL * (x_f[2] - self.pos[2])
        self.consumption += (delta_k_e + delta_p_e) / self.efficiency
        self.vel = v_f
        self.pos = x_f

    # TODO: DEPRECATED
    def addUAVToPlot(self, length):
        """
        Adds the UAV to a 3D plot with its position and a velocity normalized to length

        Args:
            length (float): the length of the normalized velocity vectors
        """
        
        axis = plt.figure().add_subplot(projection='3d')
        x, y, z = np.meshgrid(self.pos[0], self.pos[1], self.pos[2])
        axis.quiver(x, y, z, self.vel[0], self.vel[1], self.vel[2], length=length, normalize=True)


    def v(self, t, bezier):
        """
        Computes the velocity of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the velocity at
            bezier (np.array(3,4)): an array of bezier parameters

        Returns:
            np.array(3,): a velocity vector
        """

        return -3 * (1 - t) * (1 - t) * bezier[0] + (9 * t * t - 12 * t + 3) * bezier[1] + (-9 * t * t + 6 * t) * bezier[2] + 3 * t * t * bezier[3]
    

    def a(self, t, bezier):
        """
        Computes the acceleration of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the acceleration at
            bezier (np.array(3,4)): an array of bezier parameters

        Returns:
            np.array(3,): an acceleration vector
        """

        return 6 * (1 - t) * bezier[0] + (18 * t - 12) * bezier[1] + (6 - 18 * t) * bezier[2] + 6 * t * bezier[3]

    # Sample points along the curve to see that consumption is evely distributed over the curve.
    # TODO: Test with compairisons, straight and curved paths
    def computeConsumption(self, bezier, num_samples):
            """
            Computes the consumption from the array of cubic bezier parameters

            Args:
                bezier (np.array(4, 3)): an array of bezier parameters

            Returns:
                float: The work done by the UAV, in joules
            """

            dt = self.delta_t / num_samples
            moving = 0

            for i in range(num_samples):
                moving += np.abs(np.dot(self.a(dt * i, bezier), self.v(dt * i, bezier))) * dt
            
            static = 0.5 * self.delta_t * ((self.mass * GRAVITATIONAL_ACCEL) ** 1.5) / ((self.rotor_area * AIR_DENSITY) ** 0.5)
            
            return moving * self.mass / self.efficiency + static


    # TODO: Add object checks along the bezier curve trajectory, check for building checks and potentially UAV checks
    def move(self, new_pos, new_vel, bezier_matrix):
        """
        Moves the UAV from its current position and velocity to a new position and velocity over a single time step
        Uses a cubic Bezier curve to interpolate the points to minimize the consumption
        Updates the UAV's consumption with the work done by moving

        Args:
            new_pos (np.array(3,)): the UAV's new position
            new_vel (np.array(3,)): the UAV's new velocity
        """

        # Computing the bezier
        f = np.array([self.pos, self.vel, new_pos, new_vel])
        bezier = np.dot(bezier_matrix, f)

        # Updating consumption
        self.consumption += self.computeConsumption(bezier, NUM_INTEGRATION_SAMPLES)
        
        # Updating position and velocity
        self.pos = new_pos
        self.vel = new_vel


    def getConsumption(self):
        """
        Gets the total power consumption of the UAV up to this point in simulation

        Returns:
            float: the UAV's total power consumption thus far
        """
        return self.consumption
    

class GroundUser():
    def __init__(self, id, positions, intitial_velocity=np.zeros(3,), height=1.5, bandwidth=50, com_type="tx", delta_t=1, color=np.random.rand(3)):
        """
        Creates a new ground user with the specified parameters

        Args:
            id (int): the unique id number of the ground user
            positions (np.array(*, 2)): an array of positions over time on the xy plane
            initial_velocity (float): the velocity of the ground user at time zero
            height (float): the height of the ground user in meters
            bandwidth (float): the bandwidth of the ground user's device, in Mbps
            com_type (str): either "transmitter" or "receiver" denotes the type of the ground user
        """

        self.id = id
        # TODO: Check the time gap for the data from SUMO, and ensure that this time series is accurate
        self.positions = positions
        self.pos = self.positions[0]
        self.vel = intitial_velocity
        self.step = 0
        self.height = height
        self.bandwidth = bandwidth
        self.delta_t = delta_t
        # TODO: Remove device this from the state
        if com_type == "tx":
            self.device = Transmitter(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1], height])
        elif com_type == "rx":
            self.device = Receiver(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1], height])
        else:
            raise ValueError("com_type must be either 'tx' or 'rx'")
    

    def update(self):
        """
        Updates the Ground User's position to the next time step and updates velocity
        """
        self.step += 1
        if self.step >= len(self.positions):
            raise ValueError("No more positions to update for Ground User: " + self.id)
        self.vel = (self.positions[self.step] - self.positions[self.step - 1]) / self.delta_t
        self.pos = self.positions[self.step]
