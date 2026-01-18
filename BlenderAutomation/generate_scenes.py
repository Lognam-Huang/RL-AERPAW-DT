"""
@description Parses a csv file of bounding boxes and generates the scenes
in a format that is parsable by Sionna RT
@author Everett Tucker
"""

import os
import sys
import subprocess
import pandas as pd

from xml_parsing import process_xml_file


def exportScene(scene_name, bbox, env):
    """
    Exports a scene with the given name and lat/lon bounding box bbox
    
    :param scene_name: The name of the scene
    :param bbox: The lat/lon bounding box that defines the scene region
    :param env: The OS environment used to maintain a unified terminal session
    """

    # Making the necessary directories
    subprocess.run(["mkdir", "-p", f'./sionna_scene_exports/{scene_name}'], env=env)

    # Exporting environment variables for Blender
    env["BLENDER_ARGS"] = f"{bbox["min_lat"]}, " \
              f"{bbox["max_lat"]}, " \
              f"{bbox["min_lon"]}, " \
              f"{bbox["max_lon"]}, " \
              f'./sionna_scene_exports/{scene_name}/temp-scene.xml, ' \
              '1'

    # Pulling the data from OpenStreetMaps and through Blender
    subprocess.run(["blender",  "--background", "--python", "./blender_automation.py"], env=env)

    # Standardizing the .xml that blender generated
    process_xml_file(f'./sionna_scene_exports/{scene_name}/temp-scene.xml',
                     f'./sionna_scene_exports/{scene_name}/final-scene.xml')
    
    # Removing the invalid scene xml
    subprocess.run(["rm", f"./sionna_scene_exports/{scene_name}/temp-scene.xml"], env=env)


def main():
    # Defining the OS environment for all of the generating
    env = os.environ.copy()
    
    # Making the necessary parent directories
    subprocess.run(["mkdir", "-p", "./sionna_scene_exports"], env=env)

    # Reading the csv with the bounding boxes
    csv_path = "./outer_loop_scenes.csv"
    bounding_boxes = pd.read_csv(csv_path)

    # Converting all of the scenes
    print(f'Extracting data for {len(bounding_boxes)} scenes')
    for i in reversed(range(len(bounding_boxes))):
        name = bounding_boxes["Scene Name"].iloc[i]
        name = name.replace(" ", "-")  # Avoiding spaces in filenames
        lats = [float(x) for x in bounding_boxes["Latitude Range"].iloc[i].split(" - ")]
        lons = [float(x) for x in bounding_boxes["Longitude Range"].iloc[i].split(" - ")]
        bbox = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }

        try:
            exportScene(name, bbox, env)
            print(f"[{i + 1}] - {name} scene exported successfully!")
        except Exception as e:
            print(f"{name} scene failed to export:")
            print(e)
            sys.exit(1)

    print(f"All {len(bounding_boxes)} scenes exported successfully!")


if __name__ == '__main__':
    main()

