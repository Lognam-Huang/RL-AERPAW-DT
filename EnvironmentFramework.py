"""
@description Contains functionality for controlling UAVs and
interfacing with Sionna methods.
@date 11-8-2024
@author(s) Everett Tucker
"""

import sionna
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.rt import Antenna, AntennaArray
from sionna.rt import Transmitter, Receiver

GRAVITATIONAL_ACCEL = 9.80665

class Environment():
    def __init__(self, scene_path, position_df_path, time_step, ped_height=1.5, ped_rx=True):
        """
        Creates a new environment from a scene path and a position_df_path
        This method may take several minutes to run because of the scene creation

        Args:
            scene_path (str): the file path of the XML scene from Mitsuba
            position_df_pat (str): the file path of the csv position data
            time_step (float): the time step between simulation iterations
            ped_rx (bool): if the pedestrians are receivers and the UAVs are transmitters, default true
        """
        self.step = 0
        print("Loading Scene")
        self.scene = sionna.rt.load_scene(scene_path)
        print("Parsing Positions")
        self.ped_rx = ped_rx
        self.gus = self.createGroundUsers(position_df_path)
        self.ped_df = self.parsePositions(position_df_path)
        self.time_step = time_step
        self.ped_height = ped_height
        self.n_rx = 0
        self.n_tx = 0
        self.uavs = np.array([])

    
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
        for i in range(len(res)):
            rtn.append(GroundUser(i, res[i], height=1.5, com_type=("receiver" if self.ped_rx else "transmitter")))
            if self.ped_rx:
                self.n_rx += 1
            else:
                self.n_tx += 1
                
        self.step = 1
        return np.array(rtn)        
    

    def advancePedestrianPositions(self):
        """
        Advances the pedestrian positions to the next time step
        """
        self.scene._receivers.clear()
        if self.ped_rx:
            for i in range(len(self.ped_df)):
                self.scene.add(Receiver(name="ped" + str(i), 
                                        position=[self.ped_df[i]["local_person_x"].iloc[0], self.ped_df[i]["local_person_y"].iloc[0], self.ped_height], 
                                        color=np.random.rand(3)))
                self.n_rx += 1
        else:
            self.scene._transmitters.clear()
            for i in range(len(self.ped_df)):
                self.scene.add(Transmitter(name="ped" + str(i), 
                                        position=[self.ped_df[i]["local_person_x"].iloc[0], self.ped_df[i]["local_person_y"].iloc[0], self.ped_height], 
                                        color=np.random.rand(3)))
                self.n_tx += 1
        
        self.step += 1
    

    def computeShortestDistance(self):
        """
        Computes the shortest distance between any two UAVs, used for collision avoidance
        I would use the O(nlogn) algorithm, but I think the constant time factor is way too high for our application
        This algorithm is better when len(self.uavs) < 125, which I think is basically always guarenteed

        Returns:
            the shortest squared distance between any two uavs, in m
        """
        rtn = math.inf
        for i in range(len(self.uavs)):
            for j in range(i + 1, len(self.uavs)):
                rtn = min(rtn, (self.uavs[i].pos[0] - self.uava[j].pos[0]) ** 2 +
                          (self.uavs[i].pos[1] - self.uavs[j].pos[1]) ** 2,
                          (self.uavs[i].pos[2] - self.uavs[j].pos[2]) ** 2)
        return math.sqrt(rtn)
            

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
        
        a, tau = paths.cir(los=True, reflection=False, diffraction=False, scattering=False, ris=False)
        return -20 * np.log10(np.abs(a))
    
    def addUAV(self, name, mass=1, efficiency=0.8, pos=np.zeros(3), vel=np.zeros(3), color=np.random.rand(3)):
        """
        Adds a UAV to the environment and initalizes its quantities and receiver / transmitter

        Args:
            name (str): the unique name of the UAv
            mass (float): the mass of the UAV in kilograms
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV2
            color (np.array(3,)): the color of the UAV used for visualization
        """
        self.uavs.append(UAV(self.scene, name, mass, efficiency, pos, vel, self.time_step, color, self.ped_rx))
        if self.ped_rx:
            self.n_tx += 1
        else:
            self.n_rx += 1
    

    def plotUAVs(self, length=1):
        """
        Plots the positions and velocities of all the current UAVs in 3 Dimensions

        Args:
            length (float): the length of the normalized velocity vectors
        """
        for uav in self.uavs:
            uav.addUAVToPlot(length)
        plt.show()


class UAV():
    def __init__(self, id, mass, efficiency, pos, vel, bandwidth, color=np.random.rand(3), com_type="transmitter"):
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
            com_type (str): either "transmitter" or "receiver"
        """
        self.id = id
        self.mass = mass
        self.efficiency = efficiency
        self.vel = vel
        self.consumption = 0  # Cumulative consumption from movement, computation, and communication, in joules
        self.bandwidth = bandwidth
        if com_type == "transmitter":
            self.scene.add(Transmitter(name=str(id), position=pos, color=color))
        elif com_type == "receiver":
            self.scene.add(Receiver(name=str(id), position=pos, color=color))
        else:
            raise ValueError("Invalid com_type, should be 'transmitter' or 'receiver'")
    

    def impulse(self, time_step, force, wind_vector=np.zeros(3)):
        """
        Updates the UAV's position and velocity with a constant impulse over a time step
        Also updates the consumption with the new energy change
        This should be the primary method for moving the UAV

        Args:
            time_step (float): the time, in seconds, to send the impulse for. Should match the time_step of the environment
            force (np.array(3,)): the force, in Newtons, to apply to the UAV
            wind_vector (np.array(3,)): the vector of the wind, each component in m/s
        """

        v_f = force * time_step / self.mass + self.vel + wind_vector
        x_f = (time_step * time_step / (2 * self.mass)) * force + self.vel * time_step + wind_vector * time_step
        delta_k_e = 1/2 * self.mass * np.dot(v_f - self.vel, v_f - self.vel)
        delta_p_e = self.mass * GRAVITATIONAL_ACCEL * (x_f[2] - self.pos[2])
        self.consumption += (delta_k_e + delta_p_e) / self.efficiency
        self.vel = v_f
        self.pos = x_f

    
    def addUAVToPlot(self, length):
        """
        Adds the UAV to a 3D plot with its position and a velocity normalized to length

        Args:
            length (float): the length of the normalized velocity vectors
        """
        
        axis = plt.figure().add_subplot(projection='3d')
        x, y, z = np.meshgrid(self.pos[0], self.pos[1], self.pos[2])
        axis.quiver(x, y, z, self.vel[0], self.vel[1], self.vel[2], length=length, normalize=True)


    # TODO: Deprecated
    def update(self, new_pos, new_vel):
        """
        Updates the UAV's position and veclocity

        Args:
            new_pos (np.array(3,)): the UAV's new position
            new_vel (np.array(3,)): the UAV's new velocity
        """
        self.pos = new_pos
        self.vel = new_vel

        """
        TODO:
        Update to include power consumption
        """
    
    # TODO: Deprecated
    def move(self, vector):
        """
        Moves the UAV according to a specified vector
        
        Args:
            vector (np.array(3,)): the vector to move the UAV in
        """
        self.pos += vector

        """
        Update to include power consumption
        """

    def getConsumption(self):
        """
        Gets the total power consumption of the UAV up to this point in simulation

        Returns:
            float: the UAV's total power consumption thus far
        """
        return self.consumption
    

class GroundUser():
    def __init__(self, id, positions, intitial_velocity=np.zeros(3,), height=1.5, bandwidth=50, com_type="transmitter", color=np.random.rand(3)):
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
        self.positions = positions
        self.velocity = intitial_velocity
        self.step = 1
        self.height = height
        self.bandwidth
        if com_type == "transmitter":
            self.device = Transmitter(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1]])
        elif com_type == "receiver":
            self.device = Receiver(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1]])
        else:
            raise ValueError("com_type must be one of 'transmitter', 'receiver'")



        

