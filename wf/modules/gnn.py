from typing import Literal

import torch
from torch import nn

import torch_geometric as pyg

import numpy as np

from scipy.spatial import cKDTree
from einops import rearrange

from wf.modules.ffn import FFN

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


def _to_pyg_batch(x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param x: shape (batch, nodes, dim)
    :param edge_index: shape (batch, 2, edges)
    :return:
    """
    batch_size, num_nodes, dim = x.shape
    num_edges = edge_index.shape[2]
    x_pyg = x.reshape(batch_size * num_nodes, dim)
    # edges:
    offset = (torch.arange(batch_size, device=x.device) * num_nodes).view(batch_size, 1, 1)
    edge_index_pyg = edge_index + offset
    edge_index_pyg = edge_index_pyg.permute(1, 0, 2).reshape(2, batch_size * num_edges)
    return x_pyg, edge_index_pyg


def _to_regular_batch(batch_size: int, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    :param x: shape (batch*nodes, dim)
    :param edge_index: (2, batch*edges)
    :param batch_size:
    :return:
    """
    num_nodes: int = x.shape[0] // batch_size
    x = x.reshape(batch_size, num_nodes, -1)
    if edge_index is not None:
        # edges:
        edge_index = edge_index.reshape(2, batch_size, -1).permute(1, 0, 2)
        offsets = (torch.arange(batch_size, device=x.device) * num_nodes).view(batch_size, 1, 1)
        edge_index = edge_index - offsets
        return x, edge_index
    return x


def _create_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1,  phi,    0],
        [ 1,  phi,    0],
        [-1, -phi,    0],
        [ 1, -phi,    0],
        [ 0,   -1,  phi],
        [ 0,    1,  phi],
        [ 0,   -1, -phi],
        [ 0,    1, -phi],
        [phi,   0,   -1],
        [phi,   0,    1],
        [-phi,  0,   -1],
        [-phi,  0,    1],
    ], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    return vertices, faces


def _subdivide_icosahedron(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices: list = vertices.tolist()
    midpoint_cache: dict[tuple[int, int], int] = dict()

    def _midpoint(i, j):
        key: tuple[int, int] = tuple(sorted((i, j)))
        if key in midpoint_cache:
            return midpoint_cache[key]

        v = (np.array(vertices[i]) + np.array(vertices[j])) / 2
        v /= np.linalg.norm(v)

        idx = len(vertices)
        vertices.append(v.tolist())
        midpoint_cache[key] = idx
        return idx

    new_faces = list()
    for (a, b, c) in faces:
        ab = _midpoint(a, b)
        bc = _midpoint(b, c)
        ca = _midpoint(c, a)

        new_faces.extend([
            [a,  ab, ca],
            [b,  bc, ab],
            [c,  ca, bc],
            [ab, bc, ca],
        ])

    return np.array(vertices), np.array(new_faces)


def _create_icosphere(level: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces = _create_icosahedron()
    for _ in range(level):
        vertices, faces = _subdivide_icosahedron(vertices, faces)

    # get edges (bidirectional):
    edges = set()
    for (a, b, c) in faces:
        edges.add((a, b))
        edges.add((b, c))
        edges.add((c, a))
    edges = np.array(list(edges))

    return vertices, faces, edges


class GridIcoSphere:
    def __init__(
            self,
            num_lat: int,
            num_lon: int,
            mesh_level: int = 0,
            edge_len_factor: float = 0.6,
    ) -> None:
        # 1. Setup the grid
        lat_angle = 180.0 / num_lat
        lon_angle = 360.0 / num_lon
        lat_coords = (np.arange(num_lat) * lat_angle) + (lat_angle / 2.0) - 90.0
        lon_coords = (np.arange(num_lon) * lon_angle) + (lon_angle / 2.0) - 180.0
        grid_coords_rad = rearrange(
            np.deg2rad(np.stack(np.meshgrid(lat_coords, lon_coords), axis=-1)),
            "lon lat c -> lat lon c",
        )
        grid_coords_cart = np.stack((
            np.cos(grid_coords_rad[:, :, 0]) * np.cos(grid_coords_rad[:, :, 1]),
            np.cos(grid_coords_rad[:, :, 0]) * np.sin(grid_coords_rad[:, :, 1]),
            np.sin(grid_coords_rad[:, :, 0]),
        ), axis=-1)
        self._grid_cart = grid_coords_cart

        # 2. Create the icosphere mesh
        vertices, faces, edges = _create_icosphere(mesh_level)
        trias = vertices[faces]
        mean_edge_len = np.sqrt(np.square(trias - np.roll(trias, 1, axis=-2)).sum(axis=-1)).mean()
        self._vertices = vertices
        self._faces = faces

        self.mesh_edge_indices = edges.T  # -> (2, num_edges)
        self.num_nodes: int = len(vertices)
        self.num_grid_cells: int = num_lat * num_lon

        # 3. generate mappings
        grid_tree = cKDTree(rearrange(grid_coords_cart, "lat lon c -> (lat lon) c"))
        self.mesh_to_grid_idx: list[list[int]] = grid_tree.query_ball_point(vertices, r=mean_edge_len * edge_len_factor)

        g2m = dict()
        for node_idx, node_edges in enumerate(self.mesh_to_grid_idx):
            for edge in node_edges:
                if edge not in g2m:
                    g2m[edge] = [node_idx]
                else:
                    g2m[edge].append(node_idx)
        self.grid_to_mesh_idx: list[list[int]] = [g2m.get(i, []) for i in range(self.num_grid_cells)]

    def plot(self, arrows: Literal["g2m", "m2g"] | None = None, ax: Axes3D | None = None) -> Axes3D:
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax: Axes3D = ax

        grid = self._grid_cart.reshape(-1, 3) * 1.01
        sphere = self._vertices[self._faces] * 1.6

        sphere = art3d.Poly3DCollection(
            sphere,
            facecolors="lightblue",
            edgecolors="blue",
            linewidths=2,
            alpha=0.1,
        )
        ax.add_collection(sphere)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=1)

        ax.scatter3D(
            grid[:, 0],
            grid[:, 1],
            grid[:, 2],
            c="red",
            s=15,
        )

        for node_idx, grid_idxs in enumerate(self.mesh_to_grid_idx):
            for grid_idx in grid_idxs:
                ax.add_collection(art3d.Line3DCollection(
                    np.array([[self._vertices[node_idx] * 1.6, grid[grid_idx]]]),
                    edgecolors="k",
                ))
            break

        ax.view_init(elev=45.0)
        ax.set_xlim(-1.5, 1.5);
        ax.set_ylim(-1.5, 1.5);
        ax.set_zlim(-1.5, 1.5);
        ax.set_aspect("equal")

        return ax


class Grid2Mesh(nn.Module):
    lengths: torch.Tensor
    flat_idx: torch.Tensor
    segment_idx: torch.Tensor

    def __init__(
            self,
            grid: GridIcoSphere,
            in_features: int,
            out_features: int,
            edge_embed: int | None = None,
            node_embed: int | None = None,
            ffn_factor: int = 2,
            activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.grid = grid
        self.edge_dim = edge_embed if edge_embed is not None else 0
        self.node_dim = node_embed if node_embed is not None else 0
        self.out_features = out_features

        self.register_buffer("lengths", torch.tensor([len(g) for g in grid.mesh_to_grid_idx], dtype=torch.long), persistent=False)
        self.register_buffer("flat_idx", torch.cat([torch.tensor(g, dtype=torch.long) for g in grid.mesh_to_grid_idx]), persistent=False)
        self.register_buffer("segment_idx", torch.repeat_interleave(torch.arange(grid.num_nodes), self.lengths), persistent=False)
        self.num_edges = len(self.flat_idx)

        # learnable parameters
        if self.edge_dim > 0:
            self.edge_embs = nn.Parameter(torch.randn(1, self.num_edges, self.edge_dim))
        else:
            self.edge_embs = None

        if self.node_dim > 0:
            self.node_embs = nn.Parameter(torch.randn(self.grid.num_nodes, self.node_dim))
        else:
            self.node_embs = None

        edge_ffn_dim = self.edge_dim + in_features + self.node_dim
        self.edge_ffn = FFN(edge_ffn_dim, ffn_factor, activation=activation)
        if edge_ffn_dim != self.out_features:
            self.edge_ffn = nn.Sequential(
                self.edge_ffn,
                activation(),
                nn.Linear(edge_ffn_dim, self.out_features),
            )
        self.node_ffn = FFN(self.out_features, ffn_factor, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, lat*lon, in_features)
        batch_size = x.shape[0]

        edge_features = x[:, self.flat_idx]  # -> (B, num_edges, in_features)
        if self.edge_embs is not None:
            edge_features = torch.cat([
                edge_features,
                self.edge_embs.expand(batch_size, self.num_edges, self.edge_dim),
            ], dim=-1)
        if self.node_embs is not None:
            edge_features = torch.cat([
                edge_features,
                self.node_embs[self.segment_idx].unsqueeze(0).expand(batch_size, self.num_edges, self.node_dim),
            ], dim=-1)
        edge_features = self.edge_ffn(edge_features)  # -> (B, num_edges, out_features)

        node_features = torch.zeros(batch_size, self.grid.num_nodes, self.out_features, device=x.device, dtype=x.dtype)
        node_features.index_reduce_(
            1,
            source=edge_features[:, self.flat_idx],
            reduce="mean",
            index=self.segment_idx,
            include_self=False,
        )
        return self.node_ffn(node_features)  # -> (B, num_nodes, out_features)


class Mesh2Grid(nn.Module):
    lengths: torch.Tensor
    flat_idx: torch.Tensor
    segment_idx: torch.Tensor

    def __init__(
            self,
            grid: GridIcoSphere,
            in_features: int,
            out_features: int,
            edge_embed: int | None = None,
            grid_embed: int | None = None,
            ffn_factor: int = 2,
            activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.grid = grid
        self.edge_dim = edge_embed if edge_embed is not None else 0
        self.grid_dim = grid_embed if grid_embed is not None else 0
        self.out_features = out_features

        self.register_buffer("lengths", torch.tensor([len(g) for g in grid.grid_to_mesh_idx], dtype=torch.long),
                             persistent=False)
        self.register_buffer("flat_idx", torch.cat([torch.tensor(g, dtype=torch.long) for g in grid.grid_to_mesh_idx]),
                             persistent=False)
        self.register_buffer("segment_idx", torch.repeat_interleave(torch.arange(grid.num_grid_cells), self.lengths),
                             persistent=False)
        self.num_edges = len(self.flat_idx)

        # learnable parameters
        if self.edge_dim > 0:
            self.edge_embs = nn.Parameter(torch.randn(1, self.num_edges, self.edge_dim))
        else:
            self.edge_embs = None

        if self.grid_dim > 0:
            self.grid_embs = nn.Parameter(torch.randn(self.grid.num_grid_cells, self.grid_dim))
        else:
            self.grid_embs = None

        edge_ffn_dim = self.edge_dim + in_features + self.grid_dim
        self.edge_ffn = FFN(edge_ffn_dim, ffn_factor, activation=activation)
        if edge_ffn_dim != self.out_features:
            self.edge_ffn = nn.Sequential(
                self.edge_ffn,
                activation(),
                nn.Linear(edge_ffn_dim, self.out_features),
            )
        self.grid_ffn = FFN(self.out_features, ffn_factor, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, num_nodes, in_features)
        batch_size = x.shape[0]

        edge_features = x[:, self.flat_idx]  # -> (B, num_edges, in_features)
        if self.edge_embs is not None:
            edge_features = torch.cat([
                edge_features,
                self.edge_embs.expand(batch_size, self.num_edges, self.edge_dim),
            ], dim=-1)
        if self.grid_embs is not None:
            edge_features = torch.cat([
                edge_features,
                self.grid_embs[self.segment_idx].unsqueeze(0).expand(batch_size, self.num_edges, self.grid_dim),
            ], dim=-1)
        edge_features = self.edge_ffn(edge_features)  # -> (B, num_edges, out_features)

        grid_features = torch.zeros(batch_size, self.grid.num_grid_cells, self.out_features, device=x.device, dtype=x.dtype)
        grid_features.index_reduce_(
            1,
            source=edge_features[:, self.flat_idx],
            reduce="mean",
            index=self.segment_idx,
            include_self=False,
        )
        return self.grid_ffn(grid_features)  # -> (B, lat*lon, out_features)


class GridGraphConv(nn.Module):
    edge_index: torch.Tensor

    def __init__(
            self,
            dim: int,
            num_lat: int,
            num_lon: int,
            num_layers: int = 1,
            mesh_level: int = 2,
            edge_embed: int | None = None,
            node_embed: int | None = None,
            ffn_factor: int = 2,
            activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.grid_sphere = GridIcoSphere(num_lat, num_lon, mesh_level=mesh_level)
        self.register_buffer(
            "edge_index",
            torch.tensor(self.grid_sphere.mesh_edge_indices, dtype=torch.long).unsqueeze(0),  # -> (1, 2, num_edges)
            persistent=False,
        )

        self.to_mesh = Grid2Mesh(
            self.grid_sphere,
            in_features=dim,
            out_features=dim,
            edge_embed=edge_embed,
            node_embed=node_embed,
            ffn_factor=ffn_factor,
            activation=activation,
        )
        self.to_grid = Mesh2Grid(
            self.grid_sphere,
            in_features=dim,
            out_features=dim,
            edge_embed=edge_embed,
            grid_embed=node_embed,
            ffn_factor=ffn_factor,
            activation=activation,
        )

        conv_layers = list()
        for _ in range(num_layers):
            conv_layers.extend([
                (pyg.nn.GCNConv(dim, dim), "x, edge_index -> x"),
                activation(),
            ])
        self.conv_layers = pyg.nn.Sequential(
            "x, edge_index",
            conv_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, lat*lon, dim)
        batch_size = x.shape[0]

        x = self.to_mesh(x)
        x, edge_index = _to_pyg_batch(x, edge_index=self.edge_index.expand(batch_size, 2, -1))

        x = self.conv_layers(x, edge_index=edge_index)

        x = _to_regular_batch(batch_size, x=x)
        return self.to_grid(x)


if __name__ == "__main__":
    B = 3
    lat, lon = 15, 30
    dim = 32

    grid = GridIcoSphere(lat, lon, mesh_level=0)

    gc = GridGraphConv(
        dim,
        lat,
        lon,
        2,
        mesh_level=2,
        edge_embed=8,
        node_embed=6,
        ffn_factor=2,
    )


    x = torch.randn(B, lat*lon, dim)
    print(x.shape)
    x = gc(x)
    print(x.shape)

    grid.plot()
    plt.show()