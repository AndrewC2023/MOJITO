import numpy as np

# add argument for if build is happening or test
import matplotlib.pyplot as plt

from typing import List, Tuple

from Utils import PointXY, PointXYZ, generate_random_starshaped_polygon, do_two_polygons_intersect

class ConfigurationSpace2D:
    def __init__(self, *args, **kwargs):
        
        # Set defaults
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10
        self.GroundPlaneVector = [0, 0, 1]
        self.Origin2PlaneVector = [0, 0, 0]
        self.maxObstacleVertecies = 6
        self.minObstacleVertecies = 3
        self.minObstacleSize = 0.2
        self.maxObstacleSize = 2.0
        
        # Process positional arguments
        if len(args) >= 1:
            # First argument: dimensions [xMin, xMax, yMin, yMax]
            dimensions = args[0]
            if len(dimensions) == 4:
                self.xMin = dimensions[0]
                self.xMax = dimensions[1]
                self.yMin = dimensions[2]
                self.yMax = dimensions[3]
            else:
                raise ValueError("Dimensions must have 4 values: [xMin, xMax, yMin, yMax]")
        
        if len(args) >= 2:
            # Second argument: ground plane vector
            self.GroundPlaneVector = args[1]
            
        if len(args) >= 3:
            # Third argument: origin to plane vector
            self.Origin2PlaneVector = args[2]
            
        if len(args) > 3:
            raise ValueError("Too many positional arguments. Expected at most 3.")
        
        # Process keyword arguments (override positional if specified)
        if "dimensions" in kwargs:
            dimensions = kwargs["dimensions"]
            if len(dimensions) >= 4:
                self.xMin = dimensions[0]
                self.xMax = dimensions[1]
                self.yMin = dimensions[2]
                self.yMax = dimensions[3]
            else:
                raise ValueError("Dimensions must have at least 4 values: [xMin, xMax, yMin, yMax]")
        
        if "GroundPlaneVector" in kwargs:
            self.GroundPlaneVector = kwargs["GroundPlaneVector"]
            
        if "Origin2PlaneVector" in kwargs:
            self.Origin2PlaneVector = kwargs["Origin2PlaneVector"]
        
        if "maxObstacleVertecies" in kwargs:
            self.maxObstacleVertecies = kwargs["maxObstacleVertecies"]

        if "minObstacleSize" in kwargs:
            self.minObstacleSize = kwargs["minObstacleSize"]
            
        if "maxObstacleSize" in kwargs:
            self.maxObstacleSize = kwargs["maxObstacleSize"]

        # constants and list to be filled
        self.knownClassifications = ["Free", "Positive Obstacle", "Negative Obstacle", "Unknown"]
        self.obstacles: List[Tuple[List[PointXYZ], str]] = []  # List of (obstacle_points, classification)
        self.obstacles2D: List[Tuple[List[PointXY], str]] = []  # List of (obstacle_points, classification)
        self.clearZones: List[PointXYZ] = []

        # check if ground plane is, well, a ground plane
        checkVector = [0, 0, 1]
        # if angle between two is larger than 30 degrees, raise error
        angle = np.arccos(np.dot(self.GroundPlaneVector, checkVector) / (np.linalg.norm(self.GroundPlaneVector) * np.linalg.norm(checkVector)))
        if angle > np.radians(30):
            raise ValueError("Ground plane vector is not within 30 degrees of vertical.") 
        
        # calculate the z values of the corners of the ground plane
        self.cornerPoints: List[PointXYZ] = [
            [self.xMin, self.yMin, 0],
            [self.xMin, self.yMax, 0],
            [self.xMax, self.yMax, 0],
            [self.xMax, self.yMin, 0]
        ]
        # Project points onto the ground plane using the plane equation:
        # ax + by + cz = d, where [a,b,c] is the plane normal vector (GroundPlaneVector)
        # Solving for z: z = (d - ax - by) / c
        # where d = dot(normal_vector, point_on_plane)
        for point in self.cornerPoints:
            d = np.dot(self.GroundPlaneVector, self.Origin2PlaneVector)
            point.z = (d - self.GroundPlaneVector[0] * point.x - self.GroundPlaneVector[1] * point.y) / self.GroundPlaneVector[2]

        self.cornerPointsProjection: List[PointXY] = [
            [self.xMin, self.yMin],
            [self.xMin, self.yMax],
            [self.xMax, self.yMax],
            [self.xMax, self.yMin]
        ]

    def Reconfigure(self, dimensions: List[float]):
        self.xMin = dimensions[0]
        self.xMax = dimensions[1]
        self.yMin = dimensions[2]
        self.yMax = dimensions[3]
        self.obstacles.clear()

    def AddObstacle(self, obstacle, Classification: str):
        obstacleCopy = obstacle.copy()
        if Classification not in self.knownClassifications:
            raise ValueError(f"Classification '{Classification}' is not recognized. Known classifications are: {self.knownClassifications}")
        for i in range(len(obstacleCopy)):
            if not isinstance(obstacleCopy[i], (PointXYZ, PointXY)):
                raise ValueError("Obstacle points must be of type PointXYZ or PointXY")
            elif isinstance(obstacleCopy[i], PointXY):
                # Project it based on ground plane
                d = np.dot(self.GroundPlaneVector, self.Origin2PlaneVector)
                z = (d - self.GroundPlaneVector[0] * obstacleCopy[i].x - self.GroundPlaneVector[1] * obstacleCopy[i].y) / self.GroundPlaneVector[2]
                obstacleCopy[i] = PointXYZ(obstacleCopy[i].x, obstacleCopy[i].y, z)
        self.obstacles.append((obstacleCopy, Classification))

    def AddObstacles(self, obstacles, classifications: List[str]):
        if len(obstacles) != len(classifications):
            raise ValueError("Number of obstacles must match number of classifications.")
        for i in range(len(obstacles)):
            self.AddObstacle(obstacles[i], classifications[i])
    
    def addClearZone(self, zone : List[PointXY]):
        self.clearZones.append(zone)
    
    def resetField(self):
        self.obstacles.clear()
        self.clearZones.clear()
    
    def reset(self):
        self.obstacles.clear()
        self.clearZones.clear()
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10
        self.GroundPlaneVector = [0, 0, 1]
        self.Origin2PlaneVector = [0, 0, 0]

    def GenerateSpace(self, **kwargs):
        # warn user if no obstacles or clear zones
        if not self.obstacles and not self.clearZones:
            print("Warning: No obstacles or clear zones defined in the configuration space.")
        elif not self.obstacles:
            print("Warning: No obstacles defined in the configuration space.")
        elif not self.clearZones:
            print("Warning: No clear zones defined in the configuration space.")
        
        if "numObstacles" in kwargs:
            GoalNumberObstacles = kwargs["numObstacles"] 
        else:
            GoalNumberObstacles = 4
        
        # randomly create polygons that dont cross over themselves to make obstacles
        numObstacles = 0
        while numObstacles < GoalNumberObstacles:
            # randomly choose number of vertices
            numVertices = np.random.randint(3, self.maxObstacleVertecies)

            # sample an obstacle
            obstacle = generate_random_starshaped_polygon(
                self.xMin - self.maxObstacleSize,
                self.xMax + self.maxObstacleSize, 
                self.yMin - self.maxObstacleSize, 
                self.yMax + self.maxObstacleSize,
                self.minObstacleSize, self.maxObstacleSize,
                numVertices
            )

            # check first if the obstacle is entirely out of the config space
            if do_two_polygons_intersect(self.cornerPointsProjection, obstacle):
                # check if it intersects with any existing obstacles
                intersects = False
                for existing_obstacle, _ in self.obstacles:
                    if do_two_polygons_intersect(existing_obstacle, obstacle):
                        intersects = True
                        break
                    
                for clear_zone in self.clearZones:
                    if do_two_polygons_intersect(clear_zone, obstacle):
                        intersects = True
                        break
                
                if not intersects:
                    # add to list of obstacles
                    classification = "Positive Obstacle" if np.random.rand() > 0.5 else "Negative Obstacle"
                    self.obstacles.append((obstacle, classification))
                    numObstacles += 1
            # else ignore and resample


# Test main function
if __name__ == "__main__":
    def test_configuration_space():
        print("Testing ConfigurationSpace class...")
        
        # Create a configuration space
        config_space = ConfigurationSpace(
            [-5, 5, -5, 5],  # dimensions
            [0, 0, 1],       # ground plane vector (normal pointing up)
            [0, 0, 0]        # origin to plane vector
        )
        
        # Add some predefined obstacles
        obstacle1 = [PointXY(1, 1), PointXY(2, 1), PointXY(2, 2), PointXY(1, 2)]
        config_space.AddObstacle(obstacle1, "Positive Obstacle")
        
        obstacle2 = [PointXY(-3, -3), PointXY(-2, -4), PointXY(-1, -3), PointXY(-2, -2)]
        config_space.AddObstacle(obstacle2, "Negative Obstacle")
        
        # Add a clear zone
        clear_zone = [PointXY(3, 3), PointXY(4, 3), PointXY(4, 4), PointXY(3, 4)]
        config_space.addClearZone(clear_zone)
        
        # Generate random obstacles
        config_space.GenerateSpace(numObstacles=3)
        
        # Plot the configuration space
        plt.figure(figsize=(10, 10))
        
        # Plot the boundary
        boundary_x = [config_space.xMin, config_space.xMax, config_space.xMax, config_space.xMin, config_space.xMin]
        boundary_y = [config_space.yMin, config_space.yMin, config_space.yMax, config_space.yMax, config_space.yMin]
        plt.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Boundary')
        
        # Plot the obstacles
        for obstacle, classification in config_space.obstacles:
            x_coords = [point.x for point in obstacle]
            y_coords = [point.y for point in obstacle]
            x_coords.append(x_coords[0])  # Close the polygon
            y_coords.append(y_coords[0])
            
            if classification == "Positive Obstacle":
                plt.plot(x_coords, y_coords, 'r-', linewidth=2)
                plt.fill(x_coords, y_coords, 'r', alpha=0.3)
            elif classification == "Negative Obstacle":
                plt.plot(x_coords, y_coords, 'b-', linewidth=2)
                plt.fill(x_coords, y_coords, 'b', alpha=0.3)
            else:
                plt.plot(x_coords, y_coords, 'g-', linewidth=2)
                plt.fill(x_coords, y_coords, 'g', alpha=0.3)
        
        # Plot the clear zones
        for zone in config_space.clearZones:
            x_coords = [point.x for point in zone]
            y_coords = [point.y for point in zone]
            x_coords.append(x_coords[0])  # Close the polygon
            y_coords.append(y_coords[0])
            plt.plot(x_coords, y_coords, 'g-', linewidth=2)
            plt.fill(x_coords, y_coords, 'g', alpha=0.2)
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('Configuration Space')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='r', alpha=0.3, edgecolor='r', label='Positive Obstacle'),
            Patch(facecolor='b', alpha=0.3, edgecolor='b', label='Negative Obstacle'),
            Patch(facecolor='g', alpha=0.2, edgecolor='g', label='Clear Zone')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.show()
        print("Test completed!")
    
    # Run the test
    test_configuration_space()