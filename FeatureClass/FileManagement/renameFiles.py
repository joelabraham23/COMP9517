import os
import shutil

penguins = os.listdir("FeatureClass/TurtleVPenguins/archive/valid/valid/P/")
turtles = os.listdir("FeatureClass/TurtleVPenguins/archive/valid/valid/T/")
main = "FeatureClass/TurtleVPenguins/archive/valid/valid/"

count = 0
for penguinIMG in penguins:
    filename, extension = os.path.splitext(penguinIMG)
    sourcePath = os.path.join(
        "FeatureClass/TurtleVPenguins/archive/valid/valid/P", penguinIMG
    )
    destPath = os.path.join(main, f"P{count}{extension}")

    os.rename(sourcePath, destPath)
    count += 1

count = 0
for turtleIMG in turtles:
    filename, extension = os.path.splitext(turtleIMG)
    sourcePath = os.path.join(
        "FeatureClass/TurtleVPenguins/archive/valid/valid/T", turtleIMG
    )
    destPath = os.path.join(main, f"T{count}{extension}")

    os.rename(sourcePath, destPath)
    count += 1

print("done")
