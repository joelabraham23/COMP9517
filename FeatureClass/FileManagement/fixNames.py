import os

folder_path = "FeatureClass/TurtleVPenguins/archive/train/train/"

# Iterate over files in the directory
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        name, ext = os.path.splitext(filename)
        newName = name.replace(".", "")
        newFilename = newName + ext
        os.rename(
            os.path.join(folder_path, filename), os.path.join(folder_path, newFilename)
        )
