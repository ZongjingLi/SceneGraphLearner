import numpy as np
import matplotlib.pyplot as plt

def process_command(command, grid):
    prompt = "Set"
    if prompt == "Set":
        pass
    if prompt == "Rect":
        pass
    if prompt == "Segment":
        return 
    if prompt == "":
        return 

if __name__ == "__main__":
    resolution = tuple(int(w) for w in input("InputResolution:").split(","))
    grid = np.zeros(resolution)
    file_name = input("input_filename:")

    flag = True
    while(flag):
        command = input("next_command:")
        if command in ["exit"]:
            flag = False

        process_command(command, grid)
        plt.cla()
        plt.axis("off")
        plt.imshow(grid)
        plt.pause(0.01)
    
    if file_name != "0":
        np.save("{}.npy".format(file_name), grid)
        print("plot saved as: {}.npy".format(file_name))