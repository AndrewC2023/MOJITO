import numpy as np
import shapely
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Tuple
import fcl


# point type utilities:
class PointXY:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"({self.x}, {self.y})"

class PointXYT:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta
    
    def __str__(self):
        return f"({self.x}, {self.y}) , Orientation: {self.theta}"

class PointXYZ:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class IndexXY:
    def __init__(self, x: int, y: int):
        self.x: int = x 
        self.y: int = y
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

class polygon:
    def __init__(self, points: List[PointXY]):
        self.points: List[PointXY] = points

class SimpleVolume:
    def __init__(self, minPoint: PointXYZ, maxPoint: PointXYZ):
        self.minPoint: PointXYZ = min
        self.maxPoint: PointXYZ = maxPoint

class Box3D:
    """
    3D Box collision object using python-fcl for collision detection.
    
    Attributes:
        center: PointXYZ representing the center of the box
        size: PointXYZ representing the dimensions (x, y, z) of the box
        rotation: 3x3 numpy array representing the rotation matrix (default: identity)
        collision_object: fcl.CollisionObject for collision checking
    """
    def __init__(self, center: PointXYZ, size: PointXYZ, rotation: np.ndarray = None):
        """
        Initialize a 3D box for collision detection.
        
        Args:
            center: Center point of the box (PointXYZ)
            size: Dimensions of the box (PointXYZ where x=width, y=depth, z=height)
            rotation: 3x3 rotation matrix (optional, defaults to identity/no rotation)
        """
        self.center = center
        self.size = size
        
        # Set rotation matrix (identity if not provided)
        if rotation is None:
            self.rotation = np.eye(3)
        else:
            self.rotation = rotation
            
        # Create FCL box geometry (FCL uses side lengths for this)
        self.geometry = fcl.Box(size.x, size.y, size.z)
        
        # Create transform with position and rotation
        self.transform = fcl.Transform(self.rotation, [center.x, center.y, center.z])
        
        # Create collision object
        self.collision_object = fcl.CollisionObject(self.geometry, self.transform)
    
    def update_transform(self, center: PointXYZ = None, rotation: np.ndarray = None):
        """
        Update the position and/or rotation of the box.
        
        Args:
            center: New center position (optional)
            rotation: New rotation matrix (optional)
        """
        if center is not None:
            self.center = center
        if rotation is not None:
            self.rotation = rotation
        else:
            # Use identity if rotation not provided (avoid re-creating)
            if rotation is None and not hasattr(self, '_identity_rotation'):
                self.rotation = np.eye(3, dtype=np.float64)
            
        # Update transform directly on existing collision object (faster than creating new one)
        self.collision_object.setTransform(fcl.Transform(self.rotation, [self.center.x, self.center.y, self.center.z]))
    
    def check_collision(self, other: 'Box3D') -> bool:
        """
        Check if this box collides with another Box3D object.
        
        Args:
            other: Another Box3D object to check collision against
            
        Returns:
            bool: True if boxes collide, False otherwise
        """
        # Cache request object, but create new result each time (FCL requirement)
        if not hasattr(self, '_collision_request'):
            self._collision_request = fcl.CollisionRequest()
        
        result = fcl.CollisionResult()
        fcl.collide(self.collision_object, other.collision_object, self._collision_request, result)
        
        return result.is_collision
    
    def distance_to(self, other: 'Box3D') -> float:
        """
        Calculate the minimum distance between this box and another Box3D object.
        
        Args:
            other: Another Box3D object
            
        Returns:
            float: Minimum distance between the boxes (0 if colliding)
        """
        # Cache request object, but create new result each time (FCL requirement)
        if not hasattr(self, '_distance_request'):
            self._distance_request = fcl.DistanceRequest()
        
        result = fcl.DistanceResult()
        fcl.distance(self.collision_object, other.collision_object, self._distance_request, result)
        
        return result.min_distance
    
    def get_vertices(self) -> List[PointXYZ]:
        """
        Get the 8 corner vertices of the box in world coordinates.
        
        Returns:
            List[PointXYZ]: List of 8 corner points
        """
        # Half extents
        hx, hy, hz = self.size.x / 2, self.size.y / 2, self.size.z / 2
        
        # Local vertices (box-centered)
        local_vertices = [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz]
        ]
        
        # Transform to world coordinates
        world_vertices = []
        for v in local_vertices:
            rotated = self.rotation @ np.array(v)
            world_pos = rotated + np.array([self.center.x, self.center.y, self.center.z])
            world_vertices.append(PointXYZ(world_pos[0], world_pos[1], world_pos[2]))
            
        return world_vertices

def check_box_collision(box1: Box3D, box2: Box3D) -> bool:
    """
    Convenience function to check collision between two 3D boxes.
    
    Args:
        box1: First Box3D object
        box2: Second Box3D object
        
    Returns:
        bool: True if boxes collide, False otherwise
    """
    return box1.check_collision(box2)

def create_axis_aligned_box(center: PointXYZ, size: PointXYZ) -> Box3D:
    """
    Create an axis-aligned 3D box (no rotation).
    
    Args:
        center: Center point of the box
        size: Dimensions of the box (x, y, z)
        
    Returns:
        Box3D: Axis-aligned box object
    """
    return Box3D(center, size)

def create_rotated_box(center: PointXYZ, size: PointXYZ, 
                       roll: float = 0, pitch: float = 0, yaw: float = 0) -> Box3D:
    """
    Create a 3D box with rotation specified by Euler angles (roll, pitch, yaw).
    
    Args:
        center: Center point of the box
        size: Dimensions of the box (x, y, z)
        roll: Rotation about x-axis in radians
        pitch: Rotation about y-axis in radians
        yaw: Rotation about z-axis in radians
        
    Returns:
        Box3D: Rotated box object
    """
    # Build rotation matrix from Euler angles (ZYX convention)
    Rx = DCM3D(roll, "x")
    Ry = DCM3D(pitch, "y")
    Rz = DCM3D(yaw, "z")
    
    # Combined rotation: Rz * Ry * Rx
    rotation = Rz @ Ry @ Rx
    
    return Box3D(center, size, rotation)


# store bot information globally, IDK if we want this
BotLength = 1 # meters
BotWidth = 0.5 # meters
BotMaxRadius = np.sqrt(BotLength**2 + BotWidth**2)/2
BotPolygon = [
    PointXY(-BotLength/2, -BotWidth/2),
    PointXY(BotLength/2, -BotWidth/2),
    PointXY(BotLength/2, BotWidth/2),
    PointXY(-BotLength/2, BotWidth/2)
]
BotCoverageRange = 2.5 # meters

def isPointInsidePolygon(Point: PointXY, polygon: List[PointXY]) -> bool:
    inPolygon = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        if ((polygon[i].y > Point.y) != (polygon[j].y > Point.y)) and (Point.x < (polygon[j].x - polygon[i].x) * (Point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x):
            inPolygon = not inPolygon
        j = i
    return inPolygon

# TODO: check if this is faster than in shapely (update: shapely is so much slower)
def do_two_polygons_intersect(polygon1: List[PointXY], polygon2: List[PointXY]) -> bool:
    for point in polygon1:
        if isPointInsidePolygon(point, polygon2):
            return True
    for point in polygon2:
        if isPointInsidePolygon(point, polygon1):
            return True
    # if one is completely inside the other it will return true (Currently this is useful but not accurate to the name of the function)
    # here we can use shapely to check if the polygons intersect
    # poly1 = Polygon([(p.x, p.y) for p in polygon1])
    # poly2 = Polygon([(p.x, p.y) for p in polygon2])
    # if poly1.intersects(poly2):
    #     return True
    # the first two check are very fast but there are edge cases that are not covered
    # include special case if polygon1 = polygon2?
    return False

def angle_between_two_vectors(v1: PointXY, v2: PointXY) -> float:
    v1Norm = np.sqrt(v1.x**2 + v1.y**2)
    v2Norm = np.sqrt(v2.x**2 + v2.y**2)
    return np.arccos((v1.x * v2.x + v1.y * v2.y) / (v1Norm * v2Norm))

def DCM2D(angle: float):
    """ this is a direction cosine matrix for 2D rotation
        It will express the vector currently in the 'rotated' frame to the 'base' frame 
        when using the theta measured from the base frame to the rotated frame (ORDER of measurement matters!)

        note: conventianally this is called an active rotation matrix, but passive and active both work as long as you know where to measure from
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def DCM3D(angle: float, axis: str):
    """ this is a direction cosine matrix for 3D rotation
        It will express the vector currently in the 'rotated' frame to the 'base' frame
        when using the theta measured around the axis of rotation of the base frame with positive following the RHR
        i.e. if i rotate 90 degrees about the z axis i expect the roated frame to appear 90 CCW from the base frame about the z axis
        such that when I point my thumb along the z axis my fingers curl in the direction of rotation"""
    if axis == "x":
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == "y":
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "z":
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid axis")

def RAD2DEG(angle):
    return angle * 180 / np.pi
def DEG2RAD(angle):
    return angle * np.pi / 180
def wrapTo2Pi(angle):
    """ Wrap angle to [0, 2*pi] """
    return angle % (2 * np.pi)
def wrapToPi(angle):
    """ Wrap angle to [-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def expand_polygon(polygon: List[PointXY], distance: float) -> List[PointXY]:
    shapely_polygon = Polygon([(p.x, p.y) for p in polygon])
    expanded_polygon = shapely_polygon.buffer(distance, join_style=2)  # join_style=2 for mitre join to handle angles > 180 degrees
    expanded_polygon = expanded_polygon.simplify(0.2, preserve_topology=False)
    expanded_points = [PointXY(p[0], p[1]) for p in expanded_polygon.exterior.coords]
    
    return expanded_points
# expandPolygon

def simplify_polygon(polygon: List[PointXY], tolerance: float) -> List[PointXY]:
    shapely_polygon = Polygon([(p.x, p.y) for p in polygon])
    simplified_polygon = shapely_polygon.simplify(tolerance, preserve_topology=False)
    simplified_points = [PointXY(p[0], p[1]) for p in simplified_polygon.exterior.coords]
    return simplified_points
# simplifyPolygon

def expand_polygons(polygons: List[List[PointXY]], distance: float) -> List[List[PointXY]]:
    return [expand_polygon(polygon, distance) for polygon in polygons]
# expandPolygons

def combine_polygons(polygons: List[List[PointXY]]) -> List[List[PointXY]]:
    # Convert each list of PointXY into a Shapely Polygon
    shapely_polys = [Polygon([(p.x, p.y) for p in ring]) for ring in polygons]
    
    # Combine them all together
    union_result = unary_union(shapely_polys)
    
    # The result could be a single Polygon or a MultiPolygon
    if union_result.geom_type == 'Polygon':
        # Just one Polygon
        return [[PointXY(x, y) for (x, y) in union_result.exterior.coords]]
    else:
        # A MultiPolygon - return each exterior ring as its own polygon list
        return [[PointXY(x, y) for (x, y) in poly.exterior.coords] for poly in union_result.geoms]
# combinePolygons

def subtract_polygonsOnBorder_from_parent(ParentPolygon: List[PointXY], Polygons: List[List[PointXY]]) -> List[PointXY]:
    shapely_ParentPolygon = Polygon([(p.x, p.y) for p in ParentPolygon])
    shapely_Polygons = [Polygon([(p.x, p.y) for p in ring]) for ring in Polygons]
    
    # For each obstacle:
    for i in range(len(shapely_Polygons)):
        if shapely_ParentPolygon.intersects(shapely_Polygons[i]) and not shapely_Polygons[i].within(shapely_ParentPolygon):
            
            # Subtract the obstacle from the bounding box
            shapely_ParentPolygon = shapely_ParentPolygon.difference(shapely_Polygons[i])
            # At this point, it should be just one polygon
            # lets not remove the obstacle from the list of obstacles
            Polygons.pop(i)
    # convertback to out datatype
    return [PointXY(x, y) for (x, y) in shapely_ParentPolygon.exterior.coords]

# subtractPolygonsOnBorder_from_parent

def find_line_intersection(line1: tuple[PointXY, PointXY], line2: tuple[PointXY, PointXY]) -> PointXY:
    # Convert lines to Shapely LineString
    shapely_line1 = shapely.geometry.LineString([(p.x, p.y) for p in line1])
    shapely_line2 = shapely.geometry.LineString([(p.x, p.y) for p in line2])
    
    # Find intersection
    intersection = shapely_line1.intersection(shapely_line2)
    
    # Check if the intersection is a point
    if intersection.is_empty or not intersection.geom_type == 'Point':
        return None
    
    return PointXY(intersection.x, intersection.y)

# findLineIntersection

def is_point_on_line(point, line, tolerance):
    # convert to shapely types
    point = shapely.geometry.Point(point.x, point.y)
    line = shapely.geometry.LineString([(p.x, p.y) for p in line])
    return line.distance(point) < tolerance

def getCentriod(polygon: List[PointXY]) -> PointXY:
    if polygon[0] == polygon[-1 % len(polygon)]:
        steps = len(polygon) - 1
    else:
        steps = len(polygon)
    x = 0
    y = 0

    for i in range(steps):
        x += polygon[i].x
        y += polygon[i].y
    return PointXY(x/len(polygon), y/len(polygon))

def getUnitVector(vector: List[PointXY]) -> List[PointXY]:
    magnitude = np.sqrt((vector[0].x - vector[1].x) ** 2 + (vector[0].y - vector[1].y) ** 2)
    return PointXY((vector[0].x - vector[1].x) / magnitude, (vector[0].y - vector[1].y) / magnitude)

# TODO: figure this out (low priority)
def combineAjdoiningPolygons(polygons: List[List[PointXY]], tolerance: float) -> List[List[PointXY]]:
    # run through the list of polygons and check if any are adjacent
    i = 0
    while i < len(polygons):
        ii = 0
        while ii < len(polygons[i]) - 1:
            parentEdge = (polygons[i][ii], polygons[i][ii + 1])
            j = i + 1
            while j < len(polygons):
                if i != j:
                    jj = 0
                    while jj < len(polygons[j]) - 1:
                        childEdge = (polygons[j][jj], polygons[j][jj + 1])
                        if (parentEdge[0] - childEdge[0]) ** 2 <= tolerance and (parentEdge[1] - childEdge[1]) ** 2 <= tolerance:
                            # combine the two
                            parent_polygon = polygons[i]
                            child_polygon = polygons[j]
                            
                            # Find the intersection index in the child polygon
                            intersection_index = jj
                            
                            # Remove the points from the child polygon
                            child_polygon.pop(intersection_index)
                            child_polygon.pop(intersection_index)
                            
                            # Insert the child polygon points into the parent polygon
                            parent_polygon[ii + 1:ii + 1] = child_polygon[intersection_index:] + child_polygon[:intersection_index]
                            
                            # Remove the child polygon from the list
                            polygons.pop(j)
                            
                            # Restart the process since the list has changed
                            i = 0
                            ii = 0
                            j = 0
                            jj = 0
                            continue
                        jj += 1
                j += 1
            ii += 1
        i += 1
    return polygons
                            

def roundPolygonPrecision(polygon: List[PointXY], decimalPrecision: int) -> List[PointXY]:
    newPolygon = []
    for point in polygon:
        point.x = round(point.x, decimalPrecision)
        point.y = round(point.y, decimalPrecision)
        newPolygon.append(point)
    return newPolygon

def roundPolygonsPrecision(polygons: List[List[PointXY]], decimalPrecision: int) -> List[List[PointXY]]:
    return [roundPolygonPrecision(polygon, decimalPrecision) for polygon in polygons]

def generate_random_starshaped_polygon(
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    min_vertices: int = 3, 
    max_vertices: int = 6, 
    min_radius: float = 0.5, 
    max_radius: float = 2.0
) -> List[PointXY]:
    """
    Generate a random non-self-intersecting star-shaped polygon.
    
    A star-shaped polygon is one where there exists at least one point inside
    from which the entire boundary is visible (i.e., a line drawn from this point
    to any vertex does not intersect any edge of the polygon).
    
    Args:
        x_min (float): Minimum x-coordinate bound
        x_max (float): Maximum x-coordinate bound
        y_min (float): Minimum y-coordinate bound 
        y_max (float): Maximum y-coordinate bound
        min_vertices (int): Minimum number of vertices
        max_vertices (int): Maximum number of vertices
        min_radius (float): Minimum distance from center to vertex
        max_radius (float): Maximum distance from center to vertex
        
    Returns:
        List[PointXY]: List of vertices forming the polygon
    """
    # Choose a random center within bounds (with some margin)
    margin = max_radius * 1.5
    center_x = np.random.uniform(x_min + margin, x_max - margin)
    center_y = np.random.uniform(y_min + margin, y_max - margin)
    
    # Random number of vertices
    num_vertices = np.random.randint(min_vertices, max_vertices + 1)
    
    # Generate angles around the center (sorted for connecting in order)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    
    # Random distances from center (with variation to create irregular shapes)
    # Using random distribution to allow for occasional more extreme shapes
    radii_factor = np.random.random(num_vertices) 
    radii = min_radius + (max_radius - min_radius) * radii_factor
    
    # Add some "noisiness" to make shapes less regular/circular
    noise_factor = 0.3  # Controls how much noise to add (0.0 to 1.0)
    for i in range(num_vertices):
        radii[i] *= (1 + noise_factor * (np.random.random() - 0.5))
    
    # Generate vertices
    vertices = []
    for i in range(num_vertices):
        radius = radii[i]
        angle = angles[i]
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Ensure the point is within bounds
        x = max(x_min + 0.1, min(x_max - 0.1, x))
        y = max(y_min + 0.1, min(y_max - 0.1, y))
        
        vertices.append(PointXY(x, y))
        
    return vertices

if __name__ == "__main__":
    polygon1 = [PointXY(1, 1), PointXY(1, 2), PointXY(2, 2), PointXY(2, 1)]
    polygon2 = [PointXY(1.5, 1.5), PointXY(1.5, 1.75), PointXY(1.75, 1.75), PointXY(1.75, 1.5)]
    point = PointXY(1.5, 1.5)
    print("Point Test: ", isPointInsidePolygon(point, polygon1))
    print("Intersection Test: ", do_two_polygons_intersect(polygon1, polygon2))

    polygon1 = [PointXY(1, 1), PointXY(1, 2), PointXY(2, 2), PointXY(2, 1)]
    polygon2 = [PointXY(2, 2), PointXY(1.5, 1.75), PointXY(1.75, 1.75), PointXY(1.75, 1.5)]

