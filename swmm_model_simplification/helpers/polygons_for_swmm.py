import numpy as np
import shapely.geometry as shp
from sklearn.neighbors import NearestNeighbors

"""
Edit polygons for SWMM (GUI)
Inner holes in the polygon will be included in the outer boundary of the polygon.
"""


def _to_coords_list(linear_ring):
    return list(linear_ring.coords)


def _new_start_point(list_, index):
    """
    Reorder list with index 'i' as new FIRST and LAST element.

    Args:
        list_ (list): old list
        index (int): index

    Returns:
        list: resorted list
    """
    return list_[index:] + list_[:index] + [list_[index]]


def _include_inner_rings(outer_ring, inner_rings):
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(outer_ring)

    # find the nearest ring to outer ring
    next_ring_index = np.argmin([np.min(nn.kneighbors(i)[0]) for i in inner_rings])
    next_ring = inner_rings.pop(next_ring_index)

    # combine on smallest gap
    distances, indices = nn.kneighbors(next_ring)
    index_cut_inner = int(np.argmin(distances))
    index_cut_outer = int(indices[index_cut_inner][0])
    new_ring = _new_start_point(next_ring, index_cut_inner) + _new_start_point(
        outer_ring, index_cut_outer
    )

    if inner_rings:
        # are inner rings left?
        return _include_inner_rings(new_ring, inner_rings)
    else:
        return new_ring


def remodel_poly(poly):
    """
    Edit polygons for SWMM (GUI).

    Inner holes in the polygon will be included in the outer boundary of the polygon.

    Args:
        poly (shapely.geometry.Polygon): Polygon to edit.

    Returns:
        shapely.geometry.Polygon: The edited Polygon.
    """
    new_points = _include_inner_rings(
        _to_coords_list(poly.exterior), [_to_coords_list(i) for i in poly.interiors]
    )
    return shp.Polygon(new_points)
