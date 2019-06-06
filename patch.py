# Class for slide patches

from enum import Enum
from PIL import Image
from matplotlib.path import Path

class Patch(object):

    class Patch_Vertex(Enum):
        TOP_LEFT     = 1
        TOP_RIGHT    = 2
        BOTTOM_LEFT  = 3
        BOTTOM_RIGHT = 4

    def __init__(self, coords, img=None):
        """
        Initializer for Patch object

        Args:
            img (numpy uint8 array): Image contents of our patch
            coords: Coordinates of the image as returned by openslide DeepZoomGenerator's
                    get_tile_coordinates method (i.e., of the format
                    ((x_topLeft, y_topLeft), level, (width, height))).

        Returns:
            Patch object

        """

        self.img = img
        self.coords = coords
        self.top_left_vertex = coords[0]
        top_left_x = self.top_left_vertex[0]
        top_left_y = self.top_left_vertex[1]

        patch_dimensions = coords[2]
        self.width = patch_dimensions[0]
        self.height = patch_dimensions[1]

        self.top_right_vertex = (top_left_x + self.width, top_left_y)
        self.bottom_left_vertex = (top_left_x, top_left_y + self.height)
        self.bottom_right_vertex = (top_left_x + self.width, top_left_y + self.height)

    def save_img_to_disk(self, outfile_name):
        """
        Saves patch object's image contents to disk as a png file with name outfile_name

        Args:
            outfile_name (String): Name for resulting png file
        Returns:
            None (saves output to disk)
        """
        if self.img is None:
            return

        pil_img = Image.fromarray(self.img)
        pil_img.save(outfile_name + '.jpg')

    def vertex_in_annotation(self, patch_vertex, annotation):
        """
        Checks to see if a specific patch vertex is contained within a provided
        annotation

        Args:
            patch_vertex (Patch_Vertex): Enum representing which vertex of our patch we want to check
            annotation: (matplotlib.path.Path object): Path object representing the polygonal region enclosed
                                                       by a QuPath annotation

        Returns:
            in_annotation (bool): Is the given vertex in our polygon?

        """

        in_annotation = False
        if patch_vertex == Patch_Vertex.TOP_LEFT:
            in_annotation = annotation.contains_point(self.top_left_vertex)

        elif patch_vertex == Patch_Vertex.TOP_RIGHT:
            in_annotation = annotation.contains_point(self.top_right_vertex)

        elif patch_vertex == Patch_Vertex.BOTTOM_LEFT:
            in_annotation = annotation.contains_point(self.bottom_left_vertex)

        elif patch_vertex == Patch_Vertex.BOTTOM_RIGHT:
            in_annotation = annotation.contains_point(self.bottom_right_vertex)
        else:
            raise TypeError("Invalid vertex type provided to vertex_in_annotation")

        return in_annotation

    def in_annotation(self, annotation):
        """
        Checks to see if ALL of the patch's vertices are contained within a given annotation
        as given by a path object

        Args:
            annotation (Path): Path object representing a given annotation
        Returns:
            in_annotation (Boolean): True if patch contained within annotation
        """

        in_annotation = False
        if (annotation.contains_point(self.top_left_vertex) and
           annotation.contains_point(self.top_right_vertex) and
           annotation.contains_point(self.bottom_left_vertex) and
           annotation.contains_point(self.bottom_right_vertex)):
            in_annotation = True

        return in_annotation


    def on_annotation_boundary(self, annotation):
        """
        Checks to see if ANY of the patch's vertices are contained within a given annotation
        as given by a Path object

        Args:
            annotation (Path): Path object representing a given annotation
        Returns:
            on_annotation_boundary (Boolean): True if patch lies on annotation boundary
        """
        one_vertex_in = (annotation.contains_point(self.top_left_vertex) or
                annotation.contains_point(self.top_right_vertex) or
                annotation.contains_point(self.bottom_left_vertex) or
                annotation.contains_point(self.bottom_right_vertex))
        one_vertex_out = (not annotation.contains_point(self.top_left_vertex) or
            not annotation.contains_point(self.top_right_vertex) or
            not annotation.contains_point(self.bottom_left_vertex) or
            not annotation.contains_point(self.bottom_right_vertex))

        on_annotation_boundary = one_vertex_in and one_vertex_out

        return on_annotation_boundary

    def vertices_in_annotation(self, annotation, num_vertices):
        """
        Checks to see if the number of the patch's vertices that are contained within
        a given annotation as given by a Path object exceeds or equals a threshold value

        Args:
            annotation (Path): Path object representing a given annotation
            num_vertices (int): threshold for number of vertices
        Returns:
            vertices_in_annotation (Boolean): True if the number of vertices contained
                in the annotation exceeds or equals the supplied threshold value
        """
        if num_vertices > 4:
            return False
        if num_vertices == 4:
            return self.in_annotation(annotation)

        vertices_contained = 0

        if annotation.contains_point(self.top_left_vertex):
            vertices_contained += 1
        if annotation.contains_point(self.top_right_vertex):
            vertices_contained += 1
        if annotation.contains_point(self.bottom_left_vertex):
            vertices_contained += 1
        if annotation.contains_point(self.bottom_right_vertex):
            vertices_contained += 1

        return vertices_contained >= num_vertices
