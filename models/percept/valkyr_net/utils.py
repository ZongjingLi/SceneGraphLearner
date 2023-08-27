import torch

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            
            for r in range(1):
                random_long_range = torch.randint(128, (1,2) )[0]
                #edges[0].append(random_long_range[0] // size)
                #edges[1].append(random_long_range[1] % size)
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        if (i+dx) * size + (j + dy) != loc:
                            edges[0].append(loc)
                            edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

def grid(width, height):
    x = torch.linspace(0,1,width)
    y = torch.linspace(0,1,height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.cat([grid_x,grid_y], dim = 0)
    