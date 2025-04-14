# RL-AERPAW-DT
## A UAV Digital Twin for Communication Optimization with Reinforcement Learning

## Research Poster
![A research poster updated on March 18th, 2025](presentations/research_poster_final_img.jpg)
This research poster was presented at the Goodnight Research Symposium at NC State on March 18th, 2025.

## Digital Twin Demonstrations
The three GIFs below show the evolution of the simulated digital twin environment including
* (Top) Coverage Map with Ground Users color-coded based on their UAV assignment
* (Far Left) The number of Ground Users assigned to each UAV
* (Middle Left) The total theoretical maximum throughput for each UAV, a measure of how optimal the UAV's position is with respect to the Users
* (Middle Right) The total actual throughput for each UAV, including contributions from each assigned Ground User
* (Far Right) The power consumption of each UAV including movement, hovering, and signal transmission power

### Case 1: Demonstration of What-If Analysis on Network Performance (Prioritize Actual Data Rate)
![A Demonstration of the simulation showing Data Rate, UAV Energy Consumption, Receiver Assignments, and the Coverage Map.](gifs/0_assignGUs.gif)

Case 1 shows a basic throughput maximization algorithm to assign the Ground Users. It uses a Constraint-Programming Solver from Google OR-Tools to optimize a linear integer constraint problem. Case 1 shows that the actual data rate is high, but the number of assignments is incredibly variable and unstable over time, which Case 1 attempts to fix.

### Case 2: Demonstration of What-If Analysis on Network Performance (Consider UAV GU Count Loads and Actual Data Rate)
![A Demonstration of the simulation showing Data Rate, UAV Energy Consumption, Receiver Assignments, and the Coverage Map using an optimized load-balancing algorithm.](gifs/1_assignGUsWithLoadBalancing.gif)

Case 2 showcases another bi-objective algorithm that seeks to balance the cumulative actual data rate while including a load-balancing compensation. This algorithm attempts to maximize the minimum number of Ground Users assigned to any UAV. Because the total number of GUs is bounded, this results in a more equal distribution of network load. We can see from this demonstration that the total coverage, meaning the percentage of GUs assigned to a UAV, is almost always 100%. Furthermore, the variance in the load for each UAV is stable compared to Case 1. Finally, the acutal data rate is relatively unaffected by the change, meaning that we gain load-balancing without losing communication rate.

### Case 3: Demonstration of What-If Analysis on Network Performance (Consider UAV Throughput Loads and Acutal Data Rate)
![A Demonstration of the simulation showing Data Rate, UAV Energy Consumption, Receiver Assignments, and the Coverage Map using an optimized load-balancing algorithm that focuses on UAV throughput loads](gifs/2_assignGUsWithThroughputWaste.gif)

Case 3 demonstrates a bi-objective algorithm that focuses on the throughput load of the UAVs to balance the loads. Importantly, the distribution of UAV loads is slightly more consistent but lower overall than the pure throughput optimization. Furthermore, the distribution of assigned users in incredibly consistent, likely beyond what would be required of an actual simulation. This shows that taking a throughput-focused route is much more effective at balancing UAV loads than a user-count method.

## Set-Up Instructions
### Basic Initialization - Sample Data
1. To begin using the simulation environment with the sample data, which includes a 118 pedestrian position time series within a building map of downtown Raleigh, simply clone the git repository to your local machine.
2. You can then use methods from the library by importing EnvironmentFramework to your Jupyter Notebook or Python files.
3. A good starting point is running the Demonstration Notebook, which includes an overview of the simulation's features that you can customize and modify to suit your needs.

### Advanced Set-Up - Custom Data
1. Identify your target region by getting the minimum and maximum latitude and longitude coordinates of your bounding box and storing them in the file named bounding_box.txt. This is essential for the data parsing and synchronization of Building and Pedestrian Data.
2. Import building data as an XML file. There is a great [video tutorial](https://www.youtube.com/watch?v=7xHLDxUaQ7c) by a developer from Sionna. I highly suggest watching this before installation.
  * First, you need to download [Blender](https://www.blender.org/download/), which is an open-source 3D rendering and modeling software. I used Blender 4.2 but you could likely use a more recent version.
  * Download the [Mitsuba Blender Add-On](https://github.com/mitsuba-renderer/mitsuba-blender/releases). I used version 0.4.0. Just download the .zip file from their GitHub releases, then follow their installation and update guide [Link to Installation Guide](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide) to include it in Blender. This is used for exporting as an XML.
  * Download [Blosm](https://prochitecture.gumroad.com/l/blender-osm), which is another Blender add-on for importing data from OpenStreetMaps. You can choose to pay for it on the website, or not. Then, follow these [installation instructions](https://github.com/vvoovv/blosm/wiki/Documentation#installation) to set it up inside Blender.
  *  In Blender, open the Blosm tab on the right of the view area. Input your coordinates from bounding_box.txt, and then specify the import options for OpenStreetMaps. This should take a few minutes to pull all the data from the OpenStreetMaps server, but when you're done it should look like this:
![Image of successful Blosm usage.](https://github.com/user-attachments/assets/04cdd374-8967-4fe1-b13f-4dd6d436f588)
  * Go to File -> Export -> mitsuba (.xml), and then select your location and preferred coordinate system. I would suggest using -Z Forward, Y up because that's what I used in my example.
  * Move this .xml file into the data directory of your cloned repository, it should now be accessible within the simulation.
3. Generate Pedestrian Data with Simulation of Urban Mobility (SUMO).
 * Download [SUMO](https://eclipse.dev/sumo/).
 * In your SUMO installation, navigate to Eclipse -> Sumo -> tools -> osmWebWizard.py, and run it as a Python script. This should open a web browser where you can input all the parameters of your simulation in a simple format. There is also a [tutorial](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html) on using osmWebWizard.
 * The osmWebWizard should generate a directory filled with configuration files, which can then be visualized and run with the sumo-gui. Sumo-gui is located in Eclipse -> Sumo -> bin -> sumo-gui.exe. Run this executable. This should open an application where you can go to File -> Open Simulation and select your osm.sumocfg file from the osmWebWizard export. It should look something like this:
 ![image](https://github.com/user-attachments/assets/d688f01b-4563-4d49-a3f5-c623a1023b2c)
 * You can modify the simulation here, or simply run it with the green button in the top left. Once you have run the simulation, go to Simulation -> Save and save the simulation results.
 * Use the xml2csv converter located in Eclipse -> Sumo -> tools -> xml -> xml2csv, to convert the outputted xml files to csvs.
 * Use the sumoSimulationDataParser within the data parsing folder of this Git repository to convert the csv files to a useable format for the simulation environment. This requires the bounding box to account for positions effectively.
 * Move the output of the sumoSimulationDataParser into the data folder of your cloned repository, this should now be usable within your simulation environment.

## Goals
- Construct a pipeline for the simulation of pedestrian data using SUMO
- Import building data from OpenStreetMaps and use Sionna for ray tracing
- Create a realistic simulation for simulating UAV communication with ground users in urban settings.
- Train a reinforcement learning model to control the UAVs to maximize the communication capacity of the UAV network with the Ground Users.

## Current Task
- Working on a minimial control framework to simulation UAV and pedestrian movement and provide a wrapper for machine learning algorithms.

## What's Done Already
- Generating pedestrian data with SUMO and parsing it into a usable format.
- Importing data from OpenStreetMaps into Sionna
- Running Sionna ray tracing algorithms to determine communication metrics like path loss between a UAV and a ground user.


