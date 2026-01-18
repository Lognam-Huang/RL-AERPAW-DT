"""
Imports scenes into Blender from OpenStreetMaps and exports
them in mitsuba format

Command Line Parameters:
min_lat
max_lat
min_lon
max_lon
export_path: (str) the path of the file you want to export, should be *.xml
--terrain_reduction: (str) ['1', '2', '5', '10', '1200']

Run in Blender background with:
blender --background --python /home/everetttucker471/Documents/RL-AERPAW-DT/BlenderAutomation/blender_automation.py
"""

import bpy
import addon_utils
import os
import sys
# import argparse

# Grabbing arguments from environment variables
blender_args = os.environ.get("BLENDER_ARGS", None)
if blender_args is None:
    raise RuntimeError("No arguments provided")

args = blender_args.split(", ")
min_lat, max_lat, min_lon, max_lon = map(float, args[:4])
export_path = args[4]
terrain_reduction = args[5]


# parser = argparse.ArgumentParser(description="Imports scenes into Blender from OpenStreetMaps " \
# "\n and exports them in mitsuba format")

# parser.add_argument("--min_lat", type=float, help="Minimum latitude of the scene")
# parser.add_argument("--max_lat", type=float, help="Maximum latitude of the scene")
# parser.add_argument("--min_lon", type=float, help="Minimum longitude of the scene")
# parser.add_argument("--max_lon", type=float, help="Maximum longitude of the scene")
# parser.add_argument("--export_path", type=str, help="The path to export the scene to")
# parser.add_argument("--terrain_reduction", type=int, default=1, help="Reduction factor for terrain vertices")

# args = parser.parse_args()

# Grabbing user preferences from the current Blender instance
bpy.ops.wm.read_userpref()

# Enabling addons and confirming
addon_utils.enable("mitsuba-blender")
addon_utils.enable("blosm")

# Enabling Blender in preferences
bpy.ops.preferences.addon_enable(module="mitsuba-blender")
bpy.ops.preferences.addon_enable(module="blosm")

# Adding Blosm to Blender path
addon_path = bpy.utils.user_resource("SCRIPTS")
blosm_path = os.path.join(addon_path, "blosm")
if blosm_path not in sys.path:
    sys.path.append(blosm_path)

if addon_utils.check("mitsuba-blender") == (True, True):
    print("Mitsuba Enabled")
if addon_utils.check("blosm") == (True, True):
    print("Blender OSM Enabled")

# Deleting all existing objects
for o in bpy.context.scene.objects:
    o.select_set(True)
bpy.ops.object.delete()

# Setting bounding box
bpy.context.scene.blosm.minLat = min_lat
bpy.context.scene.blosm.maxLat = max_lat
bpy.context.scene.blosm.minLon = min_lon
bpy.context.scene.blosm.maxLon = max_lon

# Importing terrain first
bpy.context.scene.blosm.dataType = 'terrain'
bpy.context.scene.blosm.relativeToInitialImport = False
terrain_options = ['1', '2', '5', '10', '1200']
bpy.context.scene.blosm.terrainReductionRatio = str(terrain_reduction)
bpy.ops.object.shade_smooth()
bpy.context.preferences.addons["blosm"].preferences.dataDir = "/home/everetttucker471/Documents/RL-AERPAW-DT/BlenderAutomation/temp_data/"
bpy.ops.blosm.import_data()

# Then import buildings
bpy.context.scene.blosm.dataType = 'osm'

# Import Buildings and vegetation
bpy.context.scene.blosm.buildings = True
bpy.context.scene.blosm.forests = True
bpy.context.scene.blosm.vegetation = True

# Don't import streets or water because they aren't saved as meshes
bpy.context.scene.blosm.highways = False
bpy.context.scene.blosm.railways = False
bpy.context.scene.blosm.water = False

# Additional Parameters
bpy.context.scene.blosm.singleObject = True
bpy.context.scene.blosm.relativeToInitialImport = False

# Looping to avoid occasional 504 internal server errors with OpenStreetMaps API
i = 10
while i > 0:
    try:
        bpy.ops.blosm.import_data()
        break
    except Exception as e:
        print(f"Importing failed, retrying: {str(e)}")
        i -= 1

if i > 0:
    print("Data imported successfully!")
else:
    print("Data import failed")
    sys.exit()

# Export with the mitsuba blender plugin
bpy.ops.export_scene.mitsuba(filepath=export_path, check_existing=False, axis_forward='X', axis_up='Z')
