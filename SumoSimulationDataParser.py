# -*- coding: utf-8 -*-
"""
@description: SUMO Simulation output data parser. Generates timestamped
pedestian position data from a csv generated from a --netstate-dump call
@author: Everett Tucker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse

def latLon2Cartesian(lat, lon, bbox, r):
    centerLat = (bbox[0][0] + bbox[0][1]) / 2
    centerLon = (bbox[1][0] + bbox[1][1]) / 2
    return [2 * r * math.sin((lat - centerLat) * math.pi / 360), 
            2 * r * math.sin((lon - centerLon) * math.pi / 360)]

def parse():
    parser = argparse.ArgumentParser("simulation_data_parser")
    parser.add_argument("parameter_file", help="Input the absolute path of a text file that contains:\n the desired radius (in meters) on the first line, the space-seperated min and max latitude on the second line, and the space-seperated min and max longitude on the third line.")
    parser.add_argument("simulation_data_path", help="Input the path to a CSV file of simulated SUMO data.")
    parser.add_argument("vehicle_save_path", help="Input a file to save the vehicle data in.")
    parser.add_argument("person_save_path", help="Input a file to save the pedestrian data in.")
    args = parser.parse_args()

    # Reading arguments
    df = pd.read_csv(args.simulation_data_path, delimiter=';')
    pd.set_option("display.max_columns", len(df.columns))

    # Building Bounding box from data file
    with open(args.parameter_file, "r") as file:
        R = float(file.readline())
        bbox = []
        bbox.append([float(x) for x in file.readline().strip().split()])
        bbox.append([float(x) for x in file.readline().strip().split()])

    print(f'Using radius: {R} meters')
    print(f'Using min lat {bbox[0][0]} and max lat {bbox[0][1]}')
    print(f'Using min lon {bbox[1][0]} and max lon {bbox[1][1]}')
            
    bad_cols = ["vehicle_angle", 
                "vehicle_lane", 
                "vehicle_pos", 
                "vehicle_slope", 
                "person_angle",
                "person_edge", 
                "person_pos",
                "person_slope"]
    
    # We keep time, id, type, speed, and lat lon position
    for col in bad_cols:
        df = df.drop(col, axis=1)
                
    m = len(df)

    # Extracting dataframes
    person_df = df.loc[[i for i in range(m) if not type(df["vehicle_id"][i]) == str]]
    vehicle_df = df.loc[[i for i in range(m) if not type(df["person_id"][i]) == str]]
    
    # Dropping respective cols
    suffix = ["id", "speed", "x", "y", "type"]
    for s in suffix:
        person_df = person_df.drop("vehicle_" + s, axis=1)
        
    for s in suffix[:4]:
        vehicle_df = vehicle_df.drop("person_" + s, axis=1)
    
    # Resetting Indices
    vehicle_df.reset_index(drop=True, inplace=True)
    person_df.reset_index(drop=True, inplace=True)

    # Converting to local coords, lat->y and lon->x
    m = len(person_df)
    x = np.full(shape=m, fill_value=0.0, dtype=np.float64)
    y = np.full(shape=m, fill_value=0.0, dtype=np.float64)
    for i in range(m):
        res = latLon2Cartesian(person_df["person_y"].iloc[i], person_df["person_x"].iloc[i], bbox, R)
        x[i] = res[1]
        y[i] = res[0]
    person_df.insert(0, "local_person_x", x)
    person_df.insert(0, "local_person_y", y)

    m = len(vehicle_df)
    x.resize(m)
    y.resize(m)
    for i in range(m):
        res = latLon2Cartesian(vehicle_df["vehicle_y"].iloc[i], vehicle_df["vehicle_x"].iloc[i], bbox, R)
        x[i] = res[1]
        y[i] = res[0]
    vehicle_df.insert(0, "local_vehicle_x", x)
    vehicle_df.insert(0, "local_vehicle_y", y)
    
    # Dropping lat/lon columns
    person_df = person_df.drop("person_x", axis=1)
    person_df = person_df.drop("person_y", axis=1)
    vehicle_df = vehicle_df.drop("vehicle_x", axis=1)
    vehicle_df = vehicle_df.drop("vehicle_y", axis=1)

    """
    # Trying to print
    for i in range(100):
        person = person_df.loc[person_df["person_id"] == "ped" + str(i)]
        plt.scatter(person["local_person_x"], person["local_person_y"], color="green")
    plt.show()
    
    person0 = person_df.loc[person_df["person_id"] == "ped0"]
    person1 = person_df.loc[person_df["person_id"] == "ped1"]
    person2 = person_df.loc[person_df["person_id"] == "ped2"]
    person3 = person_df.loc[person_df["person_id"] == "ped3"]
    person4 = person_df.loc[person_df["person_id"] == "ped4"]
    person5 = person_df.loc[person_df["person_id"] == "ped5"]
    person6 = person_df.loc[person_df["person_id"] == "ped6"]
    person7 = person_df.loc[person_df["person_id"] == "ped7"]
    person8 = person_df.loc[person_df["person_id"] == "ped8"]
    person9 = person_df.loc[person_df["person_id"] == "ped9"]
    person10 = person_df.loc[person_df["person_id"] == "ped10"]
    person11 = person_df.loc[person_df["person_id"] == "ped11"]
    plt.scatter(person0["local_person_x"], person0["local_person_y"], color="red")
    plt.scatter(person1["local_person_x"], person1["local_person_y"], color="green")
    plt.scatter(person2["local_person_x"], person2["local_person_y"], color="blue")
    plt.scatter(person3["local_person_x"], person3["local_person_y"], color="purple")
    plt.scatter(person4["local_person_x"], person4["local_person_y"], color="black")
    plt.scatter(person5["local_person_x"], person5["local_person_y"], color="orange")
    plt.scatter(person6["local_person_x"], person6["local_person_y"], color="yellow")
    plt.scatter(person7["local_person_x"], person7["local_person_y"], color="pink")
    plt.scatter(person8["local_person_x"], person8["local_person_y"], color="violet")
    plt.scatter(person9["local_person_x"], person9["local_person_y"], color="lime")
    plt.scatter(person10["local_person_x"], person10["local_person_y"], color="cyan")
    plt.scatter(person11["local_person_x"], person11["local_person_y"], color="magenta")
    plt.show()
    """

    # Saving Dataframes
    vehicle_df.to_csv(args.vehicle_save_path)
    person_df.to_csv(args.person_save_path)


if __name__ == '__main__':
    parse()
