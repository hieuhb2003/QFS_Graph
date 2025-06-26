import os
from dataclasses import dataclass
from typing import Any, final

import numpy as np


from magix.types import KnowledgeGraph
from magix.utils import (
    logger,
)

from magix.base import (
    BaseGraphStorage,
)
import pipmaster as pm

if not pm.is_installed("networkx"):
    pm.install("networkx")
if not pm.is_installed("graspologic"):
    pm.install("graspologic")

try:
    from graspologic import embed
    import networkx as nx
except ImportError as e:
    raise ImportError(
        "`networkx` library is not installed. Please install it via pip: `pip install networkx`."
    ) from e


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self) -> None:
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        for node in nodes:
            if self._graph.has_node(node):
                self._graph.remove_node(node)

    def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        raise NotImplementedError

    async def get_node_data(self, node_id: str) -> dict | None:
        return self._graph.nodes[node_id]

    async def get_edge_data(self, head, tgt):
        return self._graph.edges[head, tgt]

    async def get_connected_nodes(self, node_id: str) -> list[tuple[str, dict]]:
        """Trả về tất cả các node có kết nối trực tiếp với node được chỉ định.
        
        Args:
            node_id (str): ID của node cần tìm các kết nối.
            
        Returns:
            list[tuple[str, dict]]: Danh sách các tuple gồm (node_id, node_data) của tất cả
                                    các node kết nối trực tiếp với node được chỉ định.
                                    Trả về danh sách rỗng nếu node_id không tồn tại hoặc
                                    không có kết nối.
        """
        if not self._graph.has_node(node_id):
            logger.warning(f"Node {node_id} không tồn tại trong đồ thị.")
            return []
            
        connected_nodes = []
        
        # Lấy tất cả các node kề (neighbors)
        neighbors = list(self._graph.neighbors(node_id))
        
        # Thu thập thông tin cho mỗi node kề
        # for neighbor_id in neighbors:
        #     node_data = self._graph.nodes[neighbor_id]
        #     connected_nodes.append((neighbor_id, node_data))
            
        # logger.info(f"Tìm thấy {len(connected_nodes)} node kết nối với node {node_id}")
        return neighbors
        
    async def get_connected_nodes_with_edges(self, node_id: str) -> list[tuple[str, dict, dict]]:
        """Trả về tất cả các node có kết nối trực tiếp với node được chỉ định,
        kèm theo thông tin về cạnh kết nối.
        
        Args:
            node_id (str): ID của node cần tìm các kết nối.
            
        Returns:
            list[tuple[str, dict, dict]]: Danh sách các tuple gồm (node_id, node_data, edge_data)
                                         của tất cả các node kết nối trực tiếp với node được chỉ định.
                                         Trả về danh sách rỗng nếu node_id không tồn tại hoặc 
                                         không có kết nối.
        """
        if not self._graph.has_node(node_id):
            logger.warning(f"Node {node_id} không tồn tại trong đồ thị.")
            return []
            
        connected_nodes = []
        
        # Lấy tất cả các node kề (neighbors)
        neighbors = list(self._graph.neighbors(node_id))
        
        # Thu thập thông tin cho mỗi node kề và cạnh kết nối
        for neighbor_id in neighbors:
            node_data = self._graph.nodes[neighbor_id]
            # Lấy thông tin về cạnh
            if self._graph.has_edge(node_id, neighbor_id):
                edge_data = self._graph.edges[node_id, neighbor_id]
            else:  # Trong trường hợp đồ thị có hướng
                edge_data = self._graph.edges[neighbor_id, node_id]
                
            connected_nodes.append((neighbor_id, node_data, edge_data))
            
        logger.info(f"Tìm thấy {len(connected_nodes)} node kết nối với node {node_id}")
        return connected_nodes