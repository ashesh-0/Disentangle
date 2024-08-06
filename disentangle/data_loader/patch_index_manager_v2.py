from dataclasses import dataclass

import numpy as np


@dataclass
class GridIndexManager:
    data_shape: tuple
    grid_shape: tuple
    patch_shape: tuple
    trim_boundary: bool

    def __post_init__(self):
        assert len(self.data_shape) == len(self.grid_shape), f"Data shape:{self.data_shape} and grid size:{self.grid_shape} must have the same dimension"
        assert len(self.data_shape) == len(self.patch_shape), f"Data shape:{self.data_shape} and patch shape:{self.patch_shape} must have the same dimension"
        innerpad = np.array(self.patch_shape) - np.array(self.grid_shape)
        for dim, pad in enumerate(innerpad):
            if pad < 0:
                raise ValueError(f"Patch shape:{self.patch_shape} must be greater than or equal to grid shape:{self.grid_shape} in dimension {dim}")
            if pad % 2 != 0:
                raise ValueError(f"Patch shape:{self.patch_shape} must have even padding in dimension {dim}")
    
    def get_dim_individual_size(self, dim:int):
        """
        Returns the number of the grid in the specified dimension, ignoring all other dimensions.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"

        if self.grid_shape[dim]==1 and self.patch_shape[dim]==1:
            return self.data_shape[dim]
        elif self.trim_boundary is False:
            return int(np.ceil(self.data_shape[dim] / self.grid_shape[dim]))
        else:
            excess_size = self.patch_shape[dim] - self.grid_shape[dim]
            return int(np.floor((self.data_shape[dim] - excess_size) / self.grid_shape[dim]))
    
    def grid_count(self):
        """
        Returns the total number of grids in the dataset.
        """
        return self.get_dim_size(0) * self.get_dim_individual_size(0)
    
    def get_dim_size(self, dim:int):
        """
        Returns the total number of grids for one value in the specified dimension.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        if dim == len(self.data_shape)-1:
            return 1
        
        return self.get_dim_individual_size(dim+1) * self.get_dim_size(dim+1)
    
    def get_dim_index(self, dim:int, coordinate:int):
        """
        Returns the index of the grid in the specified dimension.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        assert coordinate < self.data_shape[dim], f"Coordinate {coordinate} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim]==1 and self.patch_shape[dim]==1:
            return coordinate
        elif self.trim_boundary is False:
            return np.floor(coordinate / self.grid_shape[dim])
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim])//2
            # can be <0 if coordinate is in [0,grid_shape[dim]]
            return max(0, np.floor((coordinate - excess_size) / self.grid_shape[dim]))
        
    def dataset_idx_from_dim_indices(self, dim_indices:tuple):
        """
        Returns the index of the grid in the dataset.
        """
        assert len(dim_indices) == len(self.data_shape), f"Dimension indices {dim_indices} must have the same dimension as data shape {self.data_shape}"
        index = 0
        for dim in range(len(dim_indices)):
            index += dim_indices[dim] * self.get_dim_size(dim)
        return index
    
    def dataset_idx_from_location(self, location:tuple):
        assert len(location) == len(self.data_shape), f"Location {location} must have the same dimension as data shape {self.data_shape}"
        dim_indices = [self.get_dim_index(dim, location[dim]) for dim in range(len(location))]
        return self.dataset_idx_from_dim_indices(tuple(dim_indices))
    
    def get_topleft_location_from_dim_index(self, dim:int, dim_index:int):
        """
        Returns the top-left coordinate of the grid in the specified dimension.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        assert dim_index < self.get_dim_individual_size(dim), f"Dimension index {dim_index} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim]==1 and self.patch_shape[dim]==1:
            return dim_index
        elif self.trim_boundary is False:
            return dim_index * self.grid_shape[dim]
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim])//2
            return dim_index * self.grid_shape[dim] + excess_size

    def get_location_from_dataset_idx(self, dataset_idx:int):
        dim_indices = []
        for dim in range(len(self.data_shape)):
            dim_indices.append(dataset_idx // self.get_dim_size(dim))
            dataset_idx = dataset_idx % self.get_dim_size(dim)
        location = [self.get_topleft_location_from_dim_index(dim, dim_indices[dim]) for dim in range(len(self.data_shape))]
        return tuple(location)
    
    def on_boundary(self, dataset_idx:int, dim:int):
        """
        Returns True if the grid is on the boundary in the specified dimension.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        dim_index = dataset_idx // self.get_dim_size(dim)
        return dim_index == 0 or dim_index == self.get_dim_individual_size(dim) - 1
    
    def next_grid_along_dim(self, dataset_idx:int, dim:int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx + self.get_dim_size(dim)
        if new_idx >= self.grid_count():
            return None
        return new_idx
    
    def prev_grid_along_dim(self, dataset_idx:int, dim:int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(self.data_shape), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx - self.get_dim_size(dim)
        if new_idx < 0:
            return None

if __name__ == '__main__':
    data_shape = (1, 103, 103,2)
    grid_shape = (1, 16,16, 2)
    patch_shape = (1, 32, 32, 2)
    trim_boundary = True
    manager = GridIndexManager(data_shape, grid_shape, patch_shape, trim_boundary)
    gc = manager.grid_count()
    for i in range(gc):
        print(i, manager.get_location_from_dataset_idx(i))
    
    