"""
@description Contains functionality for controlling UAVs, Ground Users, and
interfacing with Sionna methods.
@start-date 11-8-2024
@updated 1-29-2025
@author(s) Everett Tucker
"""

import sionna
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from ortools.sat.python import cp_model
from sionna.rt import Antenna, AntennaArray
from sionna.rt import Transmitter, Receiver

# The Earth's gravitational acceleration in m/s^2
GRAVITATIONAL_ACCEL = 9.80665
# The Boltzmann Constant in Joules/Kelvin
BOLTZMANN_CONSTANT = 1.380649e-23
# Average Air Density Around the Area where the UAV Flies, kg / m^3
AIR_DENSITY = 1.213941
# The number of samples to use when calculating the energy consumption
NUM_INTEGRATION_SAMPLES = 1000

"""
TODO:
Change the path loss functions or introduce another function that
calculates the maximum achievable data rate over the connection.
That requires having the signal-to-noise ratio, the initial signal
power, and the path loss. I already have the path loss through
so I just have to worry about the other stuff.

Perhaps the SNR is not dependent on the path itself though and could
be thought of as constant, as a hyperparameter of the simulation.
"""
class Environment():
    def __init__(self, scene_path, position_df_path, desired_throughputs=None, time_step=1, ped_height=1.5, ped_rx=True, ped_color=np.zeros(3), wind_vector=np.zeros(3), temperature=290):
        """
        Creates a new environment from a scene path and a position_df_path
        This method may take several minutes to run because of the scene creation

        Args:
            scene_path (str): the file path of the XML scene from Mitsuba
            position_df_pat (str): the file path of the csv position data
            time_step (float): the time step between simulation iterations
            ped_height (float): the assumed height of the pedestrians, in meters
            ped_rx (bool): if the pedestrians are receivers and the UAVs are transmitters, default true
            ped_color (np.array(3,)): The RGB Color of the pedestrians, where each entry is in [0, 1]. Defaults to black.
            wind_vector (np.array(3,)): The velocity of the wind in the environment
            temperature (float): the temperature of the environment, in degrees Kelvin
        """
        # print("Loading Scene")
        self.scene = sionna.rt.load_scene(scene_path)
        # print("Parsing Positions")
        self.ped_rx = ped_rx
        self.time_step = time_step
        self.ped_height = ped_height
        self.ped_color = ped_color
        self.n_rx = 0
        self.n_tx = 0
        self.uavs = []
        self.gus = self.createGroundUsers(position_df_path, desired_throughputs)
        self.temperature = temperature
        self.wind = wind_vector
        # This is used to speed up computation later in the simulation
        self.bezier_matrix = np.linalg.inv(np.array([
            [1, 0, 0, 0], 
            [-3, 3, 0, 0], 
            [(1 - self.time_step) ** 3, 3 * self.time_step * (1 - self.time_step) ** 2, 3 * self.time_step ** 2 * (1 - self.time_step), self.time_step ** 3], 
            [-3 * (1 - self.time_step) ** 2, -6 * self.time_step * (1 - self.time_step) + 3 * (1 - self.time_step) ** 2, -3 * self.time_step ** 2 + 6 * self.time_step * (1 - self.time_step), 3 * self.time_step ** 2]
        ]))


    def createGroundUsers(self, position_df_path, desired_throughputs=None):
        """
        Parses the positions data from SUMO and creates a list of ground user objects
        Also initializes the inital positions with receivers or transmitters

        Args:
            position_df_path (str): the file path to the pedestrian position dataframe
            desired_throughputs (np.array(num_gus, num_time_steps)): the desired throughput time series for each ground user, in bits per second
        
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
            if desired_throughputs is None:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"], np.full(len(res[j]), 
                                      self.ped_height)]).T, height=self.ped_height, com_type=("rx" if self.ped_rx else "tx"), 
                                      delta_t=self.time_step, color=self.ped_color))
            else:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"], np.full(len(res[j]), 
                                      self.ped_height)]).T, height=self.ped_height, com_type=("rx" if self.ped_rx else "tx"), 
                                      delta_t=self.time_step, color=self.ped_color, desired_throughputs=desired_throughputs[j]))
            if self.ped_rx:
                self.n_rx += 1
            else:
                self.n_tx += 1
            self.scene.add(rtn[j].device)
                
        return np.array(rtn)
    

    def advancePedestrianPositions(self):
        """
        Advances all the pedestrian positions to the next their next time step
        """

        for id in range(len(self.gus)):
            self.updateGroundUser(id)
    

    def step(self, uav_params):
        """
        Updates the simulation by moving all ground users to the next position,
        updating the UAVs' signal power, and moving all the uavs to the positions
         specified in UAV params

        Args:
            dict(int, (float, np.array(3,), np.array(3,))): A dictionary of signal power, position, velocity tuples
            that describe the new states of all the UAVs. Any uavs without a key in
            uav_positions will remain in place after the step and their signal power will remain constant.
        """

        for x in range(len(self.uavs)):
            if x in uav_params:
                self.moveAbsUAV(x, uav_params[x][1], uav_params[x][2])
            else:
                self.moveAbsUAV(x, self.uavs[x].pos, self.uavs[x].vel)
            self.setUAVSignalPower(x, uav_params[x][0])

        self.advancePedestrianPositions()


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
                rtn = min(rtn, 
                (self.uavs[i].pos[0] - self.uavs[j].pos[0]) ** 2 +
                (self.uavs[i].pos[1] - self.uavs[j].pos[1]) ** 2 +
                (self.uavs[i].pos[2] - self.uavs[j].pos[2]) ** 2)

        return math.sqrt(rtn)
            

    def visualize(self, paths=None, coverage_map=None):
        """
        Visualizes the current receivers and transmitters in the scene.
        Includes the paths and/or coverage map, if provided.

        Args:
            (sionna.rt.Paths): A paths object that you want to display in simulation
            (sionna.rt.CoverageMap): A coverage map object that you want to display
        """

        if paths is None:
            if coverage_map is None:
                self.scene.preview(show_devices=True)
            else:
                self.scene.preview(show_devices=True, coverage_map=coverage_map)
        else:
            if coverage_map is None:
                self.scene.preview(show_devices=True, paths=paths)
            else:
                self.scene.preview(show_devices=True, paths=paths, coverage_map=coverage_map)
    

    def computeCoverageMap(self, max_depth, num_samples):
        """
        Computes a coverage map for every transmitter using the provided
        arguments for maximum reflection depth and the number of samples
        on the fibonacci sphere.

        Args:
            max_depth (int): the maximum reflection depth computed
            num_samples (int): the number of points to sample on the fibonacci sphere
        """

        return self.scene.coverage_map(max_depth=max_depth, cm_cell_size=(1,1), num_samples=num_samples, 
                                        los=True, reflection=True, diffraction=True, 
                                        edge_diffraction=True, ris=False, check_scene=False, num_runs=1)
    

    def computeLOSPaths(self):
        """
        Computes the line-of-sight paths for all potential receivers and transmitters

        Returns:
            (sionna.rt.Paths): All possible line-of-sight paths
        """
        return self.scene.compute_paths(max_depth=0, method="exhaustive", num_samples=(self.n_rx * self.n_tx), los=True,
                                         reflection=False, diffraction=False, scattering=False, check_scene=False)


    def computeLOSDataRate(self):
        """
        Computes the average Line-of-sight theoretical maximum data rate across all pairs of transmitters and receivers.
        A larger value means that the paths can send more data and the UAVs are in more
        optimal positions for communication with the Ground Users.

        Returns:
            np.array(num_rx, num_tx): The total theoretical maximum data rate for each pair of receivers and transmitters across all line-of-sight paths in the simulation
        """

        paths = self.computeLOSPaths()

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs]), np.array([x.getVelocity() for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.getVelocity() for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs]))
        
        a, tau = paths.cir(los=True, reflection=False, diffraction=False, scattering=False, ris=False)
        
        # Computes the sum of the theoetical maximum data rates for each UAV in simulation
        # r_max = Blog2(1 + (Pt * a^2) / kTB); B = bandwidth (Mbps), Pt = transmission power (W), a = path coefficients (unitless), k = Boltzmann Constant (J/K), T = temperature (Kelvin)

        a = tf.squeeze(a)  # TODO: If I only have one User or UAV in the simulation this is going to cause an error
        bandwidth = tf.convert_to_tensor([uav.bandwidth for uav in self.uavs], dtype=tf.float32)
        bandwidth = tf.broadcast_to(tf.reshape(bandwidth, [1, -1]), [self.n_rx, self.n_tx])
        signal_power = tf.convert_to_tensor([uav.signal_power for uav in self.uavs], dtype=tf.float32)
        signal_power = tf.broadcast_to(tf.reshape(signal_power, [1, -1]), [self.n_rx, self.n_tx])

        return bandwidth * np.log2(1 + (signal_power * np.abs(a) ** 2) / (BOLTZMANN_CONSTANT * self.temperature * bandwidth))
    

    def computeGeneralPaths(self, max_depth, num_samples):
        """
        Computes line-of-sight, reflection, diffraction, and scattering
        paths from all transmitters.

        Args:
            max_depth (int): the maximum reflection depth usually 2-3 works well
            num_samples (int): the number of sample points to take from the
            fiboncci sphere, usually about 10^4 or 10^5
        
        Returns:
            (sionna.rt.Paths): All possible paths in the environment
        """

        return self.scene.compute_paths(max_depth=max_depth, method="fibonacci", num_samples=num_samples, los=True,
                                         reflection=True, diffraction=True, scattering=True, check_scene=False)


    def computeGeneralDataRate(self, max_depth, num_samples):
        """
        Computes the average theoretical maximum data rate for all different types of paths, including
        line-of-sight, reflection, diffraction, and scattering for each transmitter. A
        larger value means that a UAV is in a more optimial position and can send
        more data to/from the Ground Users.

        Args:
            max_depth (int): the maximum reflection depth usually 2-3 works well
            num_samples (int): the number of sample points to take from the
            fiboncci sphere, usually about 10^4 or 10^5
        
        Returns:
            np.array(num_rx, num_tx): The theoretical maximum data rate of all possible paths for each transmitter
        """

        paths = self.computeGeneralPaths(max_depth, num_samples)

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs]), np.array([x.getVelocity() for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.getVelocity() for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs]))
        
        a, tau = paths.cir(los=True, reflection=True, diffraction=True, scattering=True, ris=False)
        
        # Computes the sum of the theoetical maximum data rates for each UAV in simulation
        # r_max = Blog2(1 + (Pt * a^2) / kTB); B = bandwidth (Mbps), Pt = transmission power (W), a = path coefficients (unitless), k = Boltzmann Constant (J/K), T = temperature (Kelvin)

        a = tf.squeeze(a)
        bandwidth = tf.convert_to_tensor([uav.bandwidth for uav in self.uavs], dtype=tf.float32)
        bandwidth = tf.broadcast_to(tf.reshape(bandwidth, [1, -1, 1]), [self.n_rx, self.n_tx, tf.shape(a)[2]])
        signal_power = tf.convert_to_tensor([uav.signal_power for uav in self.uavs], dtype=tf.float32)
        signal_power = tf.broadcast_to(tf.reshape(signal_power, [1, -1, 1]), [self.n_rx, self.n_tx, tf.shape(a)[2]])

        return tf.math.reduce_sum(bandwidth * np.log2(1 + (signal_power * np.abs(tf.squeeze(a)) ** 2) / (BOLTZMANN_CONSTANT * self.temperature * bandwidth)), axis=2).numpy().astype(np.float64)
    

    def addUAV(self, id, mass=1, efficiency=0.8, pos=np.zeros(3), vel=np.zeros(3), color=np.random.rand(3), bandwidth=50, rotor_area=None, signal_power=0, throughput_capacity=625000000):
        """
        Adds a UAV to the environment and initalizes its quantities and receiver / transmitter

        Args:
            id (int): the unique id of the UAV
            mass (float): the mass of the UAV in kilograms
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV2
            color (np.array(3,)): the color of the UAV used for visualization
            bandwidth (float): the bandwidth of the UAV's communication channel, in Mbps
            rotor_area (float): the area of the UAV's rotors, in m^2
            signal_power (float): the transmitter/receiver power of the UAV, in watts
            num_channels (int): the number of channels the UAV has for communication, equal to the number of Ground Users it can support
        
        Returns:
            (int) the id of the created UAV
        """

        if rotor_area is None:
            self.uavs.append(UAV(id, mass, efficiency, pos, vel - self.wind, bandwidth, self.time_step, mass * 0.3, signal_power, throughput_capacity))
        else:
            self.uavs.append(UAV(id, mass, efficiency, pos, vel - self.wind, bandwidth, self.time_step, rotor_area, signal_power, throughput_capacity))

        if self.ped_rx:
            self.uavs[id].device = Transmitter(name=str(id), position=pos, color=color)
            self.n_tx += 1
        else:
            self.uavs[id].device = Transmitter(name=str(id), position=pos, color=color)
            self.n_rx += 1
        
        self.scene.add(self.uavs[id].device)
        return len(self.uavs) - 1


    def setUAVSignalPower(self, id, power):
        """
        Updates the signal power of the specified UAV

        Args:
            id (int): the unique id of the UAV
            power (float): the signal power of the UAV, in watts
        """
        if power >= 0:
            self.uavs[id].signal_power = power
        else:
            raise ValueError("Signal power must be non-negative")


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


    def getUAVPos(self, id):
        """
        Gets the UAV position by Id

        Args:
            id (int): the unique id of the UAV

        Returns:
            np.array(3,): the position vector of the UAV
        """
        return self.uavs[id].pos


    def getUAVVel(self, id):
        """
        Gets the absolute velocity of the UAV by Id

        Args:
            id (int): the unique id of the UAV
        
        Returns:
            np.array(3,): the absolute velocity of the UAV
        """
        return self.uavs[id].vel + self.wind
    

    def getUAVConsumption(self, id):
        """
        Gets the power consumption of the specified UAV, in joules

        Args:
            id (int): the unique id of the UAV

        Returns:
            float: the consumption of the specified UAV, in joules
        """
        return self.uavs[id].getConsumption()


    def getConsumptions(self):
        """
        Returns a dictionary of all the id : consumption pairs for each UAV
        currently in the simulation

        Returns:
            dict(int, float): A dictionary of all the id, consumption pairs for all uavs
        """

        rtn = {}
        for x in range(len(self.uavs)):
            rtn[x] = self.getUAVConsumption(x)
        return rtn


    def updateGroundUser(self, id):
        """
        Moves a ground user to the next avaliable position and updates velocity
        """
        self.gus[id].update()
        # Just update the x and y positions, the height stays constant
        self.gus[id].device.position = tf.constant([self.gus[id].getPosition()[0], self.gus[id].getPosition()[1], self.gus[id].height])
    

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


    def plotUAVs(self, length=None):
        """
        Plots the positions and velocities of all the current UAVs in 3 Dimensions

        Args:
            length (float): the length of the normalized velocity vectors, None if not normalized
        """
        axis = plt.figure().add_subplot(projection='3d')
        for uav in self.uavs:
            c = np.random.rand(3)
            axis.scatter(uav.pos[0], uav.pos[1], uav.pos[2], color=c)
            if not np.array_equal(uav.vel, np.zeros(3)):
                x, y, z = np.meshgrid(uav.pos[0], uav.pos[1], uav.pos[2])
                if length is None:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], color=c)
                else:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], length=length, normalize=True, color=c)
        plt.show()


    def plotGUs(self):
        """
        Plots the Ground Users in 2D
        """

        for gu in self.gus:
            plt.scatter(gu.getPosition()[0], gu.getPosition()[1])
        plt.title(f'Ground User Positions for timestep: {self.gus[0].step}')
        plt.xlabel("x-coordinate")
        plt.ylabel('y-coordinate')
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
    

    def assignGUsByCount(self, path_qualities, alpha=1, beta=1):
        """
        Args:
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
            alpha (float): Optimization coefficient for the throughput maximization objective.
            beta (float): Optimization coefficient for the minimum number of GUs per UAV objective.
        
        Returns:
            assignments (list(list(int))): assignments[i] contains the list of GUs assigned to UAV i.
            total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        """

        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        # Ensuring path quality values are integers, necessary for the solver
        try:
            path_qualities = path_qualities.astype(np.int64)
        except ValueError:
            raise ValueError("Input data must be of integer type or broadcastable.")

        model = cp_model.CpModel()
        gus = range(self.n_rx)
        uavs = range(self.n_tx)
        
        x = []
        for i in gus:
            x.append([])
            for j in uavs:
                x[i].append(model.NewBoolVar(""))
        x = np.array(x)

        # Assigning each gu to at most one uav
        for i in gus:
            model.Add(np.sum(x[i, :]) <= 1)

        # Ensuring each uav's assigned gus do not exceed its total capacity
        for j in uavs:
            model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

        # Defining the task counts for each uav
        assignment_counts = np.array([model.NewIntVar(0, self.n_rx, "") for j in uavs])
        for j in uavs:
            model.Add(assignment_counts[j] == np.sum(x[:, j]))

        # Defining the minimum number of gus across all uavs
        min_gus = model.NewIntVar(0, self.n_rx, "")
        for j in uavs:
            model.Add(assignment_counts[j] >= min_gus)

        # Maximizing the combined objective
        model.Maximize(np.sum(path_qualities * x) * alpha + min_gus * beta)

        # Solving
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the model failed to solve the problem
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Input is unsolvable or infeasible.")

        # Getting uav assignments
        assignments = []
        for j in uavs:
            assignments.append([])
            for i in gus:
                if solver.Value(x[i][j]):
                    assignments[j].append(i)

        # Undoing objective function to find the total throughput
        total_throughput = (solver.ObjectiveValue() - solver.Value(min_gus) * beta) / alpha

        return assignments, total_throughput

    
    def assignGUsByThroughputWaste(self, path_qualities, alpha=1, beta=1):
        """
        Args:
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
            alpha (float): Optimization coefficient for the throughput maximization objective, should be positive.
            beta (float): Optimization coefficient for the maximum throughput waste minimization objective, should be positive.
        
        Returns:
            assignments (list(list(int))): assignments[i] contains a list of GUs assigned to UAV i.
            total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        """

        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        # Ensuring path quality values are integers, necessary for the solver
        try:
            path_qualities = path_qualities.astype(np.int64)
        except ValueError:
            raise ValueError("Input data must be of integer type or broadcastable.")

        model = cp_model.CpModel()
        gus = range(self.n_rx)
        uavs = range(self.n_tx)
        
        x = []
        for i in gus:
            x.append([])
            for j in uavs:
                x[i].append(model.NewBoolVar(""))
        x = np.array(x)

        # Assigning each gu to at most one uav
        for i in gus:
            model.Add(np.sum(x[i, :]) <= 1)

        # Ensuring each uav's assigned gus do not exceed its total capacity
        for j in uavs:
            model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

        # Defining the task counts for each uav
        throughput_wastes = np.array([model.NewIntVar(0, capacities[j], "") for j in uavs])
        for j in uavs:
            model.Add(throughput_wastes[j] == capacities[j] - np.sum(x[:, j]))

        # Defining the minimum number of gus across all uavs
        max_throughput_waste = model.NewIntVar(0, np.max(capacities), "")
        for j in uavs:
            model.Add(throughput_wastes[j] <= max_throughput_waste)

        # Maximizing the combined objective
        model.Maximize(np.sum(path_qualities * x) * alpha - max_throughput_waste * beta)

        # Solving
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the model failed to solve the problem
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Input is unsolvable or infeasible.")

        # Getting uav assignments
        assignments = []
        for j in uavs:
            assignments.append([])
            for i in gus:
                if solver.Value(x[i][j]):
                    assignments[j].append(i)

        # Undoing objective function to find the total throughput
        total_throughput = (solver.ObjectiveValue() + solver.Value(max_throughput_waste) * beta) / alpha

        return assignments, total_throughput


class UAV():
    def __init__(self, id, mass=1, efficiency=0.8, pos=np.zeros(3,), vel=np.zeros(3,), bandwidth=50, delta_t=1, rotor_area=0.5, signal_power=0, throughput_capacity=625000000):
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
            signal_power (float): the transmitter/receiver power of the UAV, in watts
            throughput_capacity (int): the throughput capacity of the UAV, in bytes per second, rounded to the nearest integer
            defaults to 625,000,000 bytes per second, or 5 Gbps
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
        self.signal_power = signal_power
        self.throughput_capacity = throughput_capacity
        self.device = None  # Initialized later
        
    
    def p(self, t, bezier):
        """
        Computes the position of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the position at
            bezier (np.array(3,4)): an array of bezier parameters
        """

        return (1 - t) * (1 - t) * (1 - t) * bezier[0] + 3 * t * (1 - 1) * (1 - t) * bezier[1] + 3 * t * t * (1 - t) * bezier[2] + t * t * t * bezier[3]


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


    def computeConsumption(self, bezier, num_samples):
        """
        Computes the consumption from the array of cubic bezier parameters through
        numerical integration with the specified number of samples and
        includes the consumption from the signal power the UAV is currently set at

        Args:
            bezier (np.array(4, 3)): an array of bezier parameters
            num_samples (int): the number of samples used in numerical integration

        Returns:
            float: The energy consumed by the UAV, in joules
        """

        dt = self.delta_t / num_samples
        moving = 0

        for i in range(num_samples):
            moving += np.abs(np.dot(self.a(dt * i, bezier), self.v(dt * i, bezier))) * dt
        
        static = 0.5 * self.delta_t * ((self.mass * GRAVITATIONAL_ACCEL) ** 1.5) / ((self.rotor_area * AIR_DENSITY) ** 0.5)
        signal = self.delta_t * self.signal_power

        return (moving * self.mass + static) / self.efficiency + signal


    # TODO: Add object checks along the bezier curve trajectory, check for building checks and potentially UAV checks
    # The method for this could be with ray tracing, if there are no paths from UAV to outside, then it has collided with a building.
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
    def __init__(self, id, positions, initial_velocity=np.zeros(3,), height=1.5, bandwidth=50, com_type="rx", delta_t=1, color=np.zeros(3), desired_throughputs=np.full(150, 375000)):
        """
        Creates a new ground user with the specified parameters

        Args:
            id (int): the unique id number of the ground user
            positions (np.array(*, 3)): an array of positions over time on the xy plane
            initial_velocity (float): the velocity of the ground user at time zero
            height (float): the height of the ground user in meters
            bandwidth (float): the bandwidth of the ground user's device, in Mbps
            com_type (str): either "transmitter" or "receiver" denotes the type of the ground user
            delta_t (float): the absolute time between each time step, in seconds
            color (np.array(3,)): the color of the UAV displayed in the visualize function, expressed as RGB values in [0, 1]
            desired_throughputs (np.array(int)): the desired throughput of the ground user at each time step, in bytes per second rounded to the nearest integer
            defaults to 375000 bytes per second, or 3 Mbps
        """

        self.id = id
        self.positions = positions
        self.initial_velocity = initial_velocity
        self.step = 0
        self.height = height
        self.bandwidth = bandwidth
        self.delta_t = delta_t
        self.desired_throughputs = desired_throughputs
        if com_type == "tx":
            self.device = Transmitter(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1], height], color=color)
        elif com_type == "rx":
            self.device = Receiver(name="gu" + str(id), position=[self.positions[0][0], self.positions[0][1], height], color=color)
        else:
            raise ValueError("com_type must be either 'tx' or 'rx'")
    

    def update(self):
        """
        Updates the Ground User's position to the next time step and updates velocity
        """
        self.step += 1
        if self.step >= len(self.positions):
            raise ValueError(f'No more positions to update for Ground User: {self.id}')
        if self.step >= len(self.desired_throughputs):
            raise ValueError(f'No more desired throughputs to update for Ground User: {self.id}')


    def getPosition(self):
        """
        Gets the current position of the Ground User

        Returns:
            float: the current position of the Ground User, in m
        """
        return self.positions[self.step]

    
    def getVelocity(self):
        """
        Gets the current velocity of the Ground User

        Returns:
            float: the current velocity of the Ground User, in m/s
        """
        if self.step == 0:
            return self.initial_velocity
        else:
            return (self.positions[self.step] - self.positions[self.step - 1]) / self.delta_t
    

    def getDesiredThroughput(self):
        """
        Gets the desired throughput of the Ground User at the current time step

        Returns:
            float: the desired throughput of the Ground User, in bytes per second rounded to the nearest integer
        """
        return self.desired_throughputs[self.step]
