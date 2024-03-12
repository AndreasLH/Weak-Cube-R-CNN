import torch
from cubercnn import util
import numpy as np
import matplotlib.pyplot as plt

'''
coordinate system is assumed to have origin in the upper left
(0,0) _________________(0,N)
|  
|    
| 
|
|
(0,M)
'''


class Box:
    '''
    2D box with the format [c1, c2, w, h] or
    [x1, y1, x2, y2]
    ```
         (x1, y1)______________________   
                |                      |   
                |                      |   
                |                      |   
                |                      |   
                |        (c1,c2)       | h
                |                      |   
                |                      |  
                |                      | 
                |______________________|(x2, y2)
                            w                      
    ```
    '''
    def __init__(self, coords: torch.Tensor, format='c1, c2, w, h') -> None:
        
        self.format = format
        if self.format == 'c1, c2, w, h':    
            self.c1 = coords[0]
            self.c2 = coords[1]
            self.width = coords[2]
            self.height = coords[3]
            if self.width < 0:
                raise ValueError('Width must be greater than 0. Did you make sure that the input is in the correct order? (center, dimensions)')
            if self.height < 0:
                raise ValueError('Height must be greater than 0. Did you make sure that the input is in the correct order? (center, dimensions)')
            
        elif self.format == 'x1, y1, x2, y2':
            self.x1 = coords[0]
            self.y1 = coords[1]
            self.x2 = coords[2]
            self.y2 = coords[3]

            # check that (x1, y1) is indeed the upper left corner
            if self.x1 > self.x2 or self.y1 > self.y2:
                raise ValueError('(x1, y1) must be the upper left corner. Did you make sure that the input is in the correct order? (upper left, bottom right)')

    def convert_boxmode(self, format_to: str) -> None:
        '''Set the format of the box, changes the coordinates inplace
        '''
        if format_to == 'c1, c2, w, h':
            self.c1, self.c2, self.width, self.height = (self.x1+self.x2)/2, (self.y1+self.y2)/2, self.x2-self.x1, self.y2-self.y1
            self.x1, self.y1, self.x2, self.y2 = None, None, None, None
            self.format = 'c1, c2, w, h'
        elif format_to == 'x1, y1, x2, y2':
            # essentially ul and br of get_all_corners
            self.x1, self.y1, self.x2, self.y2 = self.c1-self.width/2, self.c2-self.height/2, self.c1+self.width/2, self.c2+self.height/2
            self.c1, self.c2, self.width, self.height = None, None, None, None
            self.format = 'x1, y1, x2, y2'
        else:
            raise ValueError('Format not supported.')

    def get_all_corners(self) -> torch.Tensor:
        '''
        It returns the 4 corners of the box in the format [x, y]
        '''
        ul = [self.c1-self.width/2, self.c2-self.height/2]
        ur = [self.c1+self.width/2, self.c2-self.height/2]
        br = [self.c1+self.width/2, self.c2+self.height/2]
        bl = [self.c1-self.width/2, self.c2+self.height/2]

        return torch.tensor([ul, ur, br, bl])
    
    def __repr__(self) -> str:
        if self.format == 'c1, c2, w, h':
            return f"Box({self.c1}, {self.c2}, {self.width}, {self.height}), format: '{self.format}'"
        elif self.format == 'x1, y1, x2, y2':
            return f"Box({self.x1}, {self.y1}, {self.x2}, {self.y2}), format: '{self.format}'"






class Cube:
    '''
    3D box in the format [c1, c2, c3, w, h, l, R]

    Args:
        c1: The x coordinate of the center of the box.
        c2: The y coordinate of the center of the box.
        c3: The z coordinate of the center of the box.
        w: The width of the box in meters.
        h: The height of the box in meters.
        l: The length of the box in meters.
        R: The 3D rotation matrix of the box.
    ```

                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2,c3)    |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w             
    ```
    '''
    def __init__(self,tensor: torch.Tensor, R: torch.Tensor) -> None:
        self.tensor = tensor
        self.center = tensor[:3]
        self.dimensions = tensor[3:6]
        self.rotation = R

        if self.dimensions[0] < 0:
            raise ValueError('Width must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        if self.dimensions[1] < 0:
            raise ValueError('Height must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        if self.dimensions[2] < 0:
            raise ValueError('Length must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        
        if self.rotation.shape != (3,3):
            raise ValueError('Rotation must be a 3x3 matrix.')

        color = [c/255.0 for c in util.get_color()]
        self.cube = util.mesh_cuboid(torch.cat((self.center,self.dimensions)), self.rotation, color=color)

    def get_cube(self):
        return self.cube

    def get_all_corners(self) -> torch.Tensor:
        return self.cube.verts_list()[0]

        


class Bube:
    '''
    3D box on the 2D image plane in the format [cube, K]

    Args:
        cube: A cube.
        K: The 3D camera matrix of the box.
    ```
                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2)       |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w    
    ```       
    '''
    def __init__(self,cube: Cube, K: torch.Tensor) -> None:
        self.cube = cube
        self.K = K
        self.center = cube.center[:2]
        self.dimensions = cube.dimensions

        if K.shape != (3,3):
            raise ValueError('K must be a 3x3 matrix.')

    def get_all_corners(self) -> torch.Tensor:
        '''
        It returns the 8 corners of the bube in the format [x, y]
        '''
        corners = self.cube.get_all_corners()
        #corners = torch.cat((corners, torch.ones(8,1)), dim=1) 
        corners = torch.mm(self.K, corners.t()).t() # translation camera missing?
        corners = corners[:,:2]/corners[:,2].unsqueeze(1)

        return corners
    
    def plot_bube(self):
        bube_corners = self.get_all_corners(self)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(torch.cat((bube_corners[:4,0],bube_corners[0,0].reshape(1))),torch.cat((bube_corners[:4,1],bube_corners[0,1].reshape(1))),color='r',linewidth=3)
        ax.plot(torch.cat((bube_corners[4:,0],bube_corners[4,0].reshape(1))),torch.cat((bube_corners[4:,1],bube_corners[4,1].reshape(1))),color='r')
        for i in range(4):
            ax.plot(torch.cat((bube_corners[i,0].reshape(1),bube_corners[4+i,0].reshape(1))),torch.cat((bube_corners[i,1].reshape(1),bube_corners[4+i,1].reshape(1))),color='r')
