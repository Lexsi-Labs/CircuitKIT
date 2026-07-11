import heapq
import json
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from einops import einsum
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .visualization import generate_random_color, get_color


class Node:
    """
    A node in the transformer computation graph.

    Serves as the base class for InputNode, AttentionNode, MLPNode, and LogitNode.
    All score/membership state is stored on the parent Graph's tensors; node
    properties are thin views into those tensors indexed by forward_index.

    Attributes:
        name (str): Unique node identifier, e.g. 'input', 'a0.h3', 'm2', 'logits'.
        layer (int): Transformer layer this node belongs to.
        in_hook (str): TransformerLens hook name for the node's input.
        out_hook (str): TransformerLens hook name for the node's output.
        index (Tuple): Slice/index used to extract this node's activations from
            the hooked tensor (relevant for attention heads, which share a hook).
        graph (Graph): Parent graph that owns this node's state tensors.
        parents (Set[Node]): Nodes with edges pointing into this node.
        children (Set[Node]): Nodes this node has edges pointing to.
        parent_edges (Set[Edge]): Edges whose child is this node.
        child_edges (Set[Edge]): Edges whose parent is this node.
        qkv_inputs (Optional[List[str]]): Hook names for q/k/v inputs; non-None
            only for AttentionNode.
    """

    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set["Node"]
    parent_edges: Set["Edge"]
    children: Set["Node"]
    child_edges: Set["Edge"]
    in_graph: bool
    score: Optional[float]
    neurons: Optional[torch.Tensor]
    neurons_scores: Optional[torch.Tensor]
    qkv_inputs: Optional[List[str]]

    def __init__(
        self,
        name: str,
        layer: int,
        in_hook: List[str],
        out_hook: str,
        index: Tuple,
        graph: "Graph",
        qkv_inputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.graph = graph
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __repr__(self):
        return f"Node({self.name}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)

    # Nodes just report back their in_graph/score/neurons_in_graph/neurons_scores status from the graph
    @property
    def in_graph(self):
        return bool(
            self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)].item()
        )

    @in_graph.setter
    def in_graph(self, value):
        self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def score(self):
        if self.graph.nodes_scores is None:
            return None
        return self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def score(self, value):
        if self.graph.nodes_scores is None:
            raise RuntimeError(
                f"Cannot set score for node {self.name} because the graph does not have node scores enabled"
            )
        self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons(self):
        if self.graph.neurons is None:
            return None
        return self.graph.neurons[self.graph.forward_index(self, attn_slice=False)]

    @neurons.setter
    def neurons(self, value):
        if self.graph.neurons is None:
            raise RuntimeError(
                f"Cannot set score for node {self.name} because the graph does not have node scores enabled"
            )
        self.graph.neurons[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons_scores(self):
        if self.graph.neurons_scores is None:
            return None
        return self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)]

    @neurons_scores.setter
    def neurons_scores(self, value):
        if self.graph.neurons_scores is None:
            raise RuntimeError(
                f"Cannot set score for node {self.name} because the graph does not have node scores enabled"
            )
        self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)] = value


class LogitNode(Node):
    """
    Terminal node representing the final residual stream / logit output.

    Always considered in-graph; setting in_graph raises ValueError.
    Uses hook_resid_post of the last layer as its in_hook; out_hook is empty
    since nothing reads its output within the graph.
    """

    def __init__(self, n_layers: int, graph: "Graph"):
        """
        Args:
            n_layers (int): Total number of transformer layers in the model.
            graph (Graph): Parent graph instance.
        """
        name = "logits"
        index = slice(None)
        super().__init__(
            name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", "", index, graph
        )

    @property
    def in_graph(self):
        return True

    @in_graph.setter
    def in_graph(self, value):
        raise ValueError("Cannot set in_graph for logits node (always True)")

    @property
    def score(self):
        return None

    @property
    def neurons(self):
        return None

    @property
    def neurons_scores(self):
        return None


class MLPNode(Node):
    """
    Node representing an MLP sublayer.

    Named 'm{layer}', e.g. 'm0'. Supports two hook modes:
    - 'mlp_out' (default): hooks hook_mlp_in / hook_mlp_out, attribution at
      the residual-stream level (d_model neurons).
    - 'post_act': hooks mlp.hook_post for both in and out, attribution at the
      post-activation level (d_mlp neurons).
    """

    def __init__(self, layer: int, graph: "Graph", mlp_hook: str = "mlp_out"):
        """
        Args:
            layer (int): Transformer layer index.
            graph (Graph): Parent graph instance.
            mlp_hook (str): Which MLP hook to use. 'mlp_out' targets the MLP's
                residual-stream output; 'post_act' targets post-activation
                neurons. Defaults to 'mlp_out'.
        """
        name = f"m{layer}"
        index = slice(None)

        if mlp_hook == "post_act":
            in_hook_name = f"blocks.{layer}.mlp.hook_post"
            out_hook_name = f"blocks.{layer}.mlp.hook_post"
        else:
            in_hook_name = f"blocks.{layer}.hook_mlp_in"
            out_hook_name = f"blocks.{layer}.hook_mlp_out"

        super().__init__(name, layer, in_hook_name, out_hook_name, index, graph)

    @property
    def d_neuron(self) -> int:
        if self.graph.cfg.get("mlp_hook") == "post_act":
            return self.graph.cfg["d_mlp"]
        return self.graph.cfg["d_model"]


class AttentionNode(Node):
    """
    Node representing a single attention head.

    Named 'a{layer}.h{head}', e.g. 'a2.h5'. Uses hook_attn_in as in_hook and
    attn.hook_result as out_hook; qkv_inputs holds the three per-head input
    hook names used for edge-level attribution.

    Attributes:
        head (int): Head index within the layer.
    """

    head: int

    def __init__(self, layer: int, head: int, graph: "Graph"):
        """
        Args:
            layer (int): Transformer layer index.
            head (int): Attention head index within the layer.
            graph (Graph): Parent graph instance.
        """
        name = f"a{layer}.h{head}"
        self.head = head
        index = (slice(None), slice(None), head)
        super().__init__(
            name,
            layer,
            f"blocks.{layer}.hook_attn_in",
            f"blocks.{layer}.attn.hook_result",
            index,
            graph,
            qkv_inputs=[f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"],
        )

    @property
    def d_neuron(self) -> int:
        return self.graph.cfg["d_model"]


class InputNode(Node):
    """
    Node representing the token embedding (input to the residual stream).

    Named 'input'. Uses hook_embed as out_hook; in_hook is empty since no
    activation feeds into embeddings within the graph.
    """

    def __init__(self, graph: "Graph"):
        """
        Args:
            graph (Graph): Parent graph instance.
        """
        name = "input"
        index = slice(None)
        super().__init__(name, 0, "", "hook_embed", index, graph)

    @property
    def d_neuron(self) -> int:
        return self.graph.cfg["d_model"]


class Edge:
    """
    A directed edge between two nodes in the computation graph.

    Represents the flow of information from a source (parent) node to a
    destination (child) node. For attention heads, each edge is specific to
    one of q/k/v inputs. Score and in_graph state are stored in the parent
    Graph's tensors and accessed via matrix_index.

    Attributes:
        name (str): Edge identifier, e.g. 'input->a0.h0<q>' or 'm0->logits'.
        parent (Node): Source node.
        child (Node): Destination node.
        qkv (Optional[Literal['q','k','v']]): Which attention input this edge
            targets; None for MLP/logit destinations.
        hook (str): TransformerLens hook name at the child's input.
        index (Tuple): Index into the hooked tensor for the child node.
        graph (Graph): Parent graph that owns this edge's state tensors.
        matrix_index (Tuple[int, int]): (forward_index, backward_index) used
            to address score and in_graph in the graph's 2-D tensors.
    """

    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    graph: "Graph"

    def __init__(
        self,
        graph: "Graph",
        parent: Node,
        child: Node,
        qkv: Optional[Literal["q", "k", "v"]] = None,
    ):
        """
        Args:
            graph (Graph): Parent graph instance.
            parent (Node): Source node of the edge.
            child (Node): Destination node of the edge.
            qkv (Optional[Literal['q','k','v']]): Required when child is an
                AttentionNode; specifies which input projection this edge feeds.

        Raises:
            ValueError: If child is an AttentionNode and qkv is None.
        """
        self.graph = graph
        self.name = (
            f"{parent.name}->{child.name}" if qkv is None else f"{parent.name}->{child.name}<{qkv}>"
        )
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.matrix_index = (
            graph.forward_index(parent, attn_slice=False),
            graph.backward_index(child, qkv, attn_slice=False),
        )

        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(
                    f"Edge({self.name}): Edges to attention heads must have a non-none value for qkv."
                )
            self.hook = f"blocks.{child.layer}.hook_{qkv}_input"
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook

    def __repr__(self):
        return f"Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)

    @property
    def score(self):
        return self.graph.scores[self.matrix_index]

    @score.setter
    def score(self, value):
        self.graph.scores[self.matrix_index] = value

    @property
    def in_graph(self):
        return bool(self.graph.in_graph[self.matrix_index].item())

    @in_graph.setter
    def in_graph(self, value):
        self.graph.in_graph[self.matrix_index] = value


class GraphConfig(dict):
    """
    Dictionary subclass that also exposes its keys as attributes.

    Used to store graph-level model configuration (n_layers, n_heads,
    d_model, d_mlp, parallel_attn_mlp, mlp_hook) with both dict-style
    and dot-notation access, e.g. cfg['n_layers'] or cfg.n_layers.
    """

    def __init__(self, *args, **kwargs):
        super(GraphConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Graph:
    """
    Computation graph of a transformer model for circuit discovery.

    Nodes represent components (embeddings, attention heads, MLPs, logits);
    edges represent residual-stream connections between them. All membership
    and score state is stored in flat tensors indexed by forward/backward
    indices, enabling efficient batch operations.

    Naming conventions:
        - Forward nodes (sources):  'input', 'a{L}.h{H}', 'm{L}', 'logits'
        - Backward nodes (dests):   'a{L}.h{H}<q/k/v>', 'm{L}', 'logits'
        - Edges:                    '{parent}->{child}' or '{parent}->{child}<{qkv}>'

    Index layout (n_forward = 1 + n_layers*(n_heads+1)):
        0          : input
        1..n_heads : a0.h0 .. a0.h{n_heads-1}
        n_heads+1  : m0
        ...repeated per layer...

    Note: The logits node has no forward index slot and does not appear in
    nodes_in_graph (its in_graph is always True by definition). Its
    _forward_index returns n_forward, used only as a sentinel by prev_index().

    Index layout (n_backward = n_layers*(3*n_heads+1) + 1):
        Per layer: q-inputs for all heads, k-inputs, v-inputs, then MLP input.
        Last entry: logits.

    Attributes:
        nodes (Dict[str, Node]): All nodes keyed by name.
        edges (Dict[str, Edge]): All edges keyed by name.
        n_forward (int): Number of forward (source) node slots.
        n_backward (int): Number of backward (destination) node slots.
            Attention heads contribute 3 slots each (q, k, v).
        scores (Tensor): Edge attribution scores [n_forward, n_backward].
        in_graph (Tensor): Boolean mask of active edges [n_forward, n_backward].
        real_edge_mask (Tensor): Boolean mask of structurally valid edges
            [n_forward, n_backward]. Prevents causal violations (e.g. m2->m0).
        forward_to_backward (Tensor): Boolean mapping [n_forward, n_backward]
            indicating which backward indices a forward node corresponds to.
            Used for fast node↔edge consistency checks in prune().
        nodes_in_graph (Tensor): Boolean mask of active nodes [n_forward].
        nodes_scores (Optional[Tensor]): Node attribution scores [n_forward],
            or None if node scores are disabled. NaN = unscored (always kept).
        neurons_in_graph (Optional[Tensor]): Boolean mask of active neurons
            [n_forward, max_d], or None if neuron level is disabled.
            max_d = max(d_model, d_mlp) when mlp_hook='post_act', else d_model.
        neurons_scores (Optional[Tensor]): Neuron attribution scores
            [n_forward, max_d], or None if neuron level is disabled.
            NaN = unscored (always kept).
        cfg (GraphConfig): Model configuration (n_layers, n_heads, d_model,
            d_mlp, parallel_attn_mlp, mlp_hook).
    """

    nodes: Dict[
        str, Node
    ]  # Maps from node names ('input', 'a0.h0', 'm0', 'logits', etc.) to Node objects
    edges: Dict[
        str, Edge
    ]  # Maps from edge names ('input->a0.h0', 'a0.h0->m0', etc.) to Edge objects. Attn edges are denoted as 'input->a0.h0<q>', 'input->a0.h0<k>', 'input->a0.h0<v>'
    n_forward: int  # the number of forward (source) nodes
    n_backward: int  # the number of backward (destination) nodes
    scores: torch.Tensor  # (n_forward, n_backward) tensor of edge scores
    in_graph: torch.Tensor  # (n_forward, n_backward) tensor of whether the edge is in the graph
    neurons_scores: Optional[
        torch.Tensor
    ]  # (n_forward, d_model) tensor of neuron scores for each forward node. If a neuron's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
    neurons_in_graph: Optional[
        torch.Tensor
    ]  # (n_forward, d_model) tensor of whether the neuron is in the graph
    nodes_scores: Optional[
        torch.Tensor
    ]  # (n_forward) tensor of source node scores. If None, nodes have no scores. If a node's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
    nodes_in_graph: torch.Tensor  # (n_forward) tensor of whether the (source) node is in the graph
    forward_to_backward: torch.Tensor
    real_edge_mask: (
        torch.Tensor
    )  # (n_forward, n_backward) tensor of whether the edge is real (some edges are not real, e.g. m10->m2)
    cfg: GraphConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(self, parent: Node, child: Node, qkv: Optional[Literal["q", "k", "v"]] = None):
        """
        Create an edge between parent and child and register it in the graph.

        Marks the edge as real in real_edge_mask and updates parent/child
        adjacency sets on both nodes.

        Args:
            parent (Node): Source node.
            child (Node): Destination node.
            qkv (Optional[Literal['q','k','v']]): Required when child is an
                AttentionNode; specifies the q/k/v input being connected.

        Raises:
            ValueError: If child is an AttentionNode and qkv is None.
        """
        edge = Edge(self, parent, child, qkv)
        self.real_edge_mask[edge.matrix_index] = True
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def prev_index(self, node: Node) -> Union[int, slice]:
        """
        Return the exclusive upper bound on forward indices that can feed into node.

        Used to slice activation_difference[:, :, :prev_index] when constructing
        a node's input — only nodes with forward_index < prev_index precede it
        in the residual stream.

        Args:
            node (Node): The destination node.

        Returns:
            int: Exclusive upper bound.
                - InputNode  → 0 (nothing precedes it).
                - LogitNode  → n_forward (everything precedes it).
                - AttentionNode → index of the first attention head in that layer.
                - MLPNode    → index after all attention heads in that layer
                  (or same as attention start if parallel_attn_mlp is True).

        Raises:
            ValueError: If node is of an unrecognised type.
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg["parallel_attn_mlp"]:
                return 1 + node.layer * (self.cfg["n_heads"] + 1)
            else:
                return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        elif isinstance(node, AttentionNode):
            i = 1 + node.layer * (self.cfg["n_heads"] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    @classmethod
    def _n_forward(cls, cfg) -> int:
        return 1 + cfg.n_layers * (cfg.n_heads + 1)

    @classmethod
    def _n_backward(cls, cfg) -> int:
        return cfg.n_layers * (3 * cfg.n_heads + 1) + 1

    @classmethod
    def _forward_index(cls, cfg, node_name: str, attn_slice: bool = False) -> int:
        """
        Return the forward (source) index for a node given its name and a config.

        Args:
            cfg: Object with n_layers and n_heads attributes.
            node_name (str): Node name, e.g. 'input', 'logits', 'm2', 'a1.h3'.
            attn_slice (bool): If True and node is an AttentionNode, return a
                slice covering all heads in that layer instead of a single head
                index. Useful for layer-level operations. Defaults to False.

        Returns:
            Union[int, slice]: Integer index for most nodes; slice for attention
                nodes when attn_slice=True.

        Raises:
            ValueError: If node_name is not a recognised format.
        """
        if node_name == "input":
            return 0
        elif node_name == "logits":
            return 1 + cfg.n_layers * (cfg.n_heads + 1)
        elif node_name[0] == "m":
            layer = int(node_name[1:])
            return 1 + layer * (cfg.n_heads + 1) + cfg.n_heads
        elif node_name[0] == "a":
            layer, head = node_name.split(".")
            layer = int(layer[1:])
            head = int(head[1:])
            i = 1 + layer * (cfg.n_heads + 1)
            return slice(i, i + cfg.n_heads) if attn_slice else i + head
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def forward_index(self, node: Node, attn_slice=True) -> int:
        return Graph._forward_index(self.cfg, node.name, attn_slice)

    @classmethod
    def _backward_index(cls, cfg, node_name: str, qkv=None, attn_slice=False) -> int:
        """
        Return the backward (destination) index for a node given its name and a config.

        Args:
            cfg: Dict-like object with n_heads key.
            node_name (str): Node name, e.g. 'logits', 'm2', 'a1.h3'.
            qkv (Optional[Literal['q','k','v']]): Required for attention nodes;
                selects which input projection's index to return.
            attn_slice (bool): If True and node is an AttentionNode, return a
                slice covering all heads in that layer for the given qkv.
                Defaults to False.

        Returns:
            Union[int, slice]: Integer index for most nodes; slice for attention
                nodes when attn_slice=True.

        Raises:
            ValueError: If node_name is 'input' (no backward index exists) or
                is an unrecognised format.
            AssertionError: If node is an attention node and qkv is not in 'qkv'.
        """
        if node_name == "input":
            raise ValueError("No backward for input node")
        elif node_name == "logits":
            return -1
        elif node_name[0] == "m":
            layer = int(node_name[1:])
            return (layer) * (3 * cfg["n_heads"] + 1) + 3 * cfg["n_heads"]
        elif node_name[0] == "a":
            assert qkv in "qkv", f"Must give qkv for AttentionNode, but got {qkv}"
            layer, head = node_name.split(".")
            layer = int(layer[1:])
            head = int(head[1:])
            i = layer * (3 * cfg["n_heads"] + 1) + ("qkv".index(qkv) * cfg["n_heads"])
            return slice(i, i + cfg["n_heads"]) if attn_slice else i + head
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def backward_index(self, node: Node, qkv=None, attn_slice=True) -> int:
        return Graph._backward_index(self.cfg, node.name, qkv, attn_slice)

    def get_dst_nodes(self) -> List[str]:
        """
        Return an ordered list of all destination node names.

        Iterates layers in order; within each layer outputs q/k/v slots for
        every head, then the MLP slot. Appends 'logits' last. The ordering
        matches the backward-index layout used in scores and in_graph.

        Returns:
            List[str]: Destination node names, e.g.
                ['a0.h0<q>', 'a0.h1<q>', ..., 'a0.h0<v>', ..., 'm0', ..., 'logits'].
        """
        heads = []
        for layer in range(self.cfg["n_layers"]):
            for letter in "qkv":
                for attention_head in range(self.cfg["n_heads"]):
                    heads.append(f"a{layer}.h{attention_head}<{letter}>")
            heads.append(f"m{layer}")
        heads.append("logits")
        return heads

    def weighted_edge_count(self) -> float:
        """
        Count active edges weighted by the fraction of active neurons per source node.

        At edge level (neurons_in_graph is None): equivalent to count_included_edges().
        At neuron level: sums over all (src, dst, neuron) triplets where both the
        edge [src→dst] and the neuron at src are active, then divides by d_model.
        A source node with half its neurons active contributes 0.5 per outgoing
        active edge rather than 1.

        Returns:
            float: Weighted edge count.
        """
        if self.neurons_in_graph is not None:
            return (
                einsum(
                    self.in_graph.float(),
                    self.neurons_in_graph.float(),
                    "forward backward, forward d_model ->",
                )
                / self.cfg["d_model"]
            ).item()
        else:
            return float(self.count_included_edges())

    def count_included_edges(self) -> int:
        return self.in_graph.sum().item()

    def count_included_nodes(self) -> int:
        return self.nodes_in_graph.sum().item()

    def count_included_neurons(self) -> int:
        return self.neurons_in_graph.sum().item()

    def reset(self, empty=True):
        """
        Reset all circuit membership flags.

        Args:
            empty (bool): If True, remove everything from the circuit
                (nodes_in_graph, in_graph, neurons_in_graph all set to False).
                If False, add all structurally valid components
                (nodes and neurons set to True; edges set to True masked by
                real_edge_mask to preserve causal validity). Defaults to True.
        """
        if empty:
            self.nodes_in_graph *= False
            self.in_graph *= False
            if self.neurons_in_graph is not None:
                self.neurons_in_graph *= False
        else:
            self.nodes_in_graph[:] = True
            self.in_graph[:] = True
            self.in_graph &= self.real_edge_mask
            if self.neurons_in_graph is not None:
                self.neurons_in_graph[:] = True

    def apply_threshold(
        self,
        threshold: float,
        absolute: bool = True,
        reset: bool = True,
        level: Literal["edge", "node", "neuron"] = "edge",
        prune=True,
    ):
        """
        Include all components whose score meets or exceeds a threshold.

        Unscored components (NaN score) are always included regardless of threshold.

        Args:
            threshold (float): Minimum score required for inclusion.
            absolute (bool): Compare against absolute score values. Defaults to True.
            reset (bool): Clear the circuit before applying the threshold.
                When True and level is 'node' or 'neuron', outgoing edges of
                included nodes are also activated. Defaults to True.
            level (Literal['edge','node','neuron']): Granularity at which to
                apply the threshold. Defaults to 'edge'.
            prune (bool): Call prune() after selection to ensure full
                connectivity. Defaults to True.

        Raises:
            ValueError: If level is not 'edge', 'node', or 'neuron'.
        """
        threshold = float(threshold)
        if reset:
            self.reset()

        if level == "neuron":
            unscored_neurons = torch.isnan(self.neurons_scores)
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)

            # We definitely want unscored neurons to be in the graph
            neuron_score_copy[unscored_neurons] = torch.inf
            included_neurons = neuron_score_copy >= threshold
            self.neurons_in_graph[:] = included_neurons

            if reset:
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == "node":
            unscored_nodes = torch.isnan(self.nodes_scores)

            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)

            node_score_copy[unscored_nodes] = torch.inf
            included_nodes = node_score_copy >= threshold
            self.nodes_in_graph[:] = included_nodes

            if reset:
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == "edge":
            edge_scores = self.scores.clone()
            if absolute:
                edge_scores = torch.abs(edge_scores)

            # masking out the edges that are not real
            edge_scores[~self.real_edge_mask] = -torch.inf

            surpass_threshold = edge_scores >= threshold
            self.in_graph[:] = surpass_threshold

            if reset:
                nodes_with_outgoing = self.in_graph.any(dim=1)
                nodes_with_ingoing = (
                    einsum(
                        self.in_graph.any(dim=0).float(),
                        self.forward_to_backward.float(),
                        "backward, forward backward -> forward",
                    )
                    > 0
                )
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing
        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune()

    def apply_random(
        self,
        n: int,
        level: Literal["edge", "node", "neuron"] = "edge",
        reset: bool = True,
        prune: bool = True,
        seed: int = None,
    ):
        """
        Randomly select n components and add them to the circuit.

        Only scored components (non-NaN) participate in random selection.
        Unscored components are always re-added after selection.
        Primarily used to build random baselines for faithfulness evaluation.

        Args:
            n (int): Number of scored components to randomly select.
            level (Literal['edge','node','neuron']): Granularity of selection.
                Defaults to 'edge'.
            reset (bool): Clear the circuit before selection. When True and
                level is 'node' or 'neuron', outgoing edges of selected nodes
                are also activated. Defaults to True.
            prune (bool): Call prune() after selection. Defaults to True.
            seed (Optional[int]): Torch random seed for reproducibility.
                Defaults to None.

        Raises:
            AssertionError: If n exceeds the number of scored components at
                the requested level.
            ValueError: If level is not 'edge', 'node', or 'neuron'.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if reset:
            self.reset()

        if level == "neuron":
            scored_neurons = ~torch.isnan(self.neurons_scores)
            n_scored_neurons = scored_neurons.sum()
            n = min(n, int(n_scored_neurons.item()))

            # Create a flat mask of valid neurons. squeeze(-1) (not bare
            # squeeze) keeps the index dimension when exactly one neuron is
            # scored — a bare squeeze() would collapse [1, 1] to a 0-D scalar
            # and make valid_indices.size(0) raise IndexError.
            valid_indices = torch.nonzero(scored_neurons.view(-1)).squeeze(-1)

            # Randomly select n indices from valid indices
            perm = torch.randperm(valid_indices.size(0))
            selected_indices = valid_indices[perm[:n]]

            # Set selected neurons to be in the graph
            self.neurons_in_graph.view(-1)[selected_indices] = True

            # Unscored neurons must also be re-added to the graph
            self.neurons_in_graph.view(-1)[~scored_neurons.view(-1)] = True

            if reset:
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == "node":
            scored_nodes = ~torch.isnan(self.nodes_scores)
            n_scored_nodes = scored_nodes.sum()
            n = min(n, int(n_scored_nodes.item()))

            valid_indices = torch.nonzero(scored_nodes.view(-1)).squeeze(-1)
            perm = torch.randperm(valid_indices.size(0))
            selected_indices = valid_indices[perm[:n]]

            self.nodes_in_graph.view(-1)[selected_indices] = True
            self.nodes_in_graph.view(-1)[~scored_nodes.view(-1)] = True  # Keep unscored

            if reset:
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == "edge":
            n = min(n, int(self.real_edge_mask.sum().item()))

            valid_indices = torch.nonzero(self.real_edge_mask.view(-1)).squeeze(-1)
            perm = torch.randperm(valid_indices.size(0))
            selected_indices = valid_indices[perm[:n]]

            self.in_graph.view(-1)[selected_indices] = True

            if reset:
                nodes_with_outgoing = self.in_graph.any(dim=1)
                nodes_with_ingoing = (
                    einsum(
                        self.in_graph.any(dim=0).float(),
                        self.forward_to_backward.float(),
                        "backward, forward backward -> forward",
                    )
                    > 0
                )
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing

        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune()

    def apply_topn(
        self,
        n: int,
        absolute: bool = True,
        level: Literal["edge", "node", "neuron"] = "edge",
        reset: bool = True,
        prune: bool = True,
    ):
        """
        Include only the top-n highest-scoring components in the circuit.

        Unscored components (NaN score) are always kept regardless of n.
        Out-of-scope nodes can be pinned to inf before calling this method
        to ensure they are always included (see api.py _compute_n_topn).

        Args:
            n (int): Number of scored components to keep.
            absolute (bool): Rank by absolute score values. Defaults to True.
            level (Literal['edge','node','neuron']): Granularity of selection.
                Defaults to 'edge'.
            reset (bool): Clear the circuit before selection. When True and
                level is 'node' or 'neuron', all outgoing edges of selected
                nodes are also activated. Defaults to True.
            prune (bool): Call prune() after selection to ensure full
                connectivity. Defaults to True.

        Raises:
            AssertionError: If n exceeds the number of scored components at
                the requested level.
            ValueError: If level is not 'edge', 'node', or 'neuron'.
        """
        if reset:
            self.reset()

        if level == "neuron":
            scored_neurons = ~torch.isnan(self.neurons_scores)
            n_scored_neurons = scored_neurons.sum()
            n = min(n, int(n_scored_neurons.item()))
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)

            neuron_score_copy[~scored_neurons] = -torch.inf
            sorted_neurons = torch.argsort(neuron_score_copy.view(-1), descending=True)

            # set the topn neurons to be in the graph
            self.neurons_in_graph.view(-1)[sorted_neurons[:n]] = True
            # set those outside the topn not to be in the graph
            self.neurons_in_graph.view(-1)[sorted_neurons[n:]] = False
            # unscored neurons must also be re-added to the graph
            self.neurons_in_graph.view(-1)[~scored_neurons.view(-1)] = True
            # remove any nodes with no neurons in the graph

            if reset:
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == "node":
            scored_nodes = ~torch.isnan(self.nodes_scores)
            n_scored_nodes = scored_nodes.sum()
            n = min(n, int(n_scored_nodes.item()))

            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)

            node_score_copy[~scored_nodes] = -torch.inf
            sorted_nodes = torch.argsort(node_score_copy.view(-1), descending=True)

            # set the topn neurons to be in the graph
            self.nodes_in_graph.view(-1)[sorted_nodes[:n]] = True
            # set those outside the topn not to be in the graph
            self.nodes_in_graph.view(-1)[sorted_nodes[n:]] = False
            # unscored nodes must also be re-added to the graph
            self.nodes_in_graph.view(-1)[~scored_nodes.view(-1)] = True

            if reset:
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        # get top-n edges
        elif level == "edge":
            n = min(n, int(self.real_edge_mask.sum().item()))

            edge_scores = self.scores.clone()
            if absolute:
                edge_scores = torch.abs(edge_scores)

            # masking out the edges that are not real
            edge_scores[~self.real_edge_mask] = -torch.inf

            sorted_edges = torch.argsort(edge_scores.view(-1), descending=True)
            self.in_graph.view(-1)[sorted_edges[:n]] = True
            self.in_graph.view(-1)[sorted_edges[n:]] = False
            if reset:
                nodes_with_outgoing = self.in_graph.any(dim=1)
                nodes_with_ingoing = (
                    einsum(
                        self.in_graph.any(dim=0).float(),
                        self.forward_to_backward.float(),
                        "backward, forward backward -> forward",
                    )
                    > 0
                )
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing

        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune()

    def apply_greedy(
        self, n_edges: int, absolute: bool = True, reset: bool = True, prune: bool = True
    ):
        """
        Select edges greedily, seeding from edges into the logit node and expanding upstream.

        Begins with all edges whose child is the logit node as initial candidates
        (LogitNode.in_graph is always True, so these are the only edges with an
        in-graph child after a reset). Adds the highest-scoring candidate edge,
        marks its parent in-graph if not already, then merges that parent's own
        parent edges into the candidate pool. Repeats until n_edges are selected.
        This guarantees the resulting subgraph is always connected to the output.

        Args:
            n_edges (int): Number of edges to include.
            absolute (bool): Rank edges by absolute score. Defaults to True.
            reset (bool): Clear the circuit before selection. Defaults to True.
            prune (bool): Call prune() after selection. Defaults to True.

        Raises:
            ValueError: If n_edges exceeds the total number of edges in the graph.
        """
        if n_edges > len(self.edges):
            raise ValueError(
                f"n ({n_edges}) is greater than the number of edges ({len(self.edges)})"
            )

        if reset:
            self.nodes_in_graph *= False
            self.in_graph *= False

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted(
            [edge for edge in self.edges.values() if edge.child.in_graph],
            key=lambda edge: abs_id(edge.score),
            reverse=True,
        )

        edges = heapq.merge(candidate_edges, key=lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted(
                    [parent_edge for parent_edge in parent.parent_edges],
                    key=lambda edge: abs_id(edge.score),
                    reverse=True,
                )
                edges = heapq.merge(
                    edges, parent_parent_edges, key=lambda edge: abs_id(edge.score), reverse=True
                )

        if prune:
            self.prune()

    def prune(self):
        """
        Remove disconnected components to produce a fully connected circuit.

        Iteratively removes nodes lacking both incoming and outgoing edges, then
        drops edges whose parent or child is no longer in the graph. Repeats
        until stable. If neuron-level tracking is enabled, first removes nodes
        with no active neurons, and finally clears neurons belonging to removed nodes.

        The input node is treated as always having incoming edges; the logits
        node is treated as always having a destination. This operation only
        removes components — it never adds them.
        """
        # remove neuronless nodes
        if self.neurons_in_graph is not None:
            self.nodes_in_graph *= self.neurons_in_graph.any(dim=1)

        old_new_same = False
        # Could take twice as many iterations as there are layers! But will probably not
        while not old_new_same:
            # remove nodes with 0 incoming or outgoing edges
            nodes_with_outgoing = self.in_graph.any(dim=1)
            nodes_with_ingoing = (
                einsum(
                    self.in_graph.any(dim=0).float(),
                    self.forward_to_backward.float(),
                    "backward, forward backward -> forward",
                )
                > 0
            )
            nodes_with_ingoing[0] = True  # input node always treated as if it has incoming edges
            old_nodes_in_graph = self.nodes_in_graph.clone()
            self.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing

            # remove edges with missing parents or children
            forward_in_graph = self.nodes_in_graph.float()
            backward_in_graph = self.nodes_in_graph.float() @ self.forward_to_backward.float()
            backward_in_graph[-1] = 1  # logits node is always present
            edge_remask = (
                einsum(forward_in_graph, backward_in_graph, "forward, backward -> forward backward")
                > 0
            )
            old_edges_in_graph = self.in_graph.clone()
            self.in_graph *= edge_remask
            old_new_same = torch.all(old_nodes_in_graph == self.nodes_in_graph) and torch.all(
                old_edges_in_graph == self.in_graph
            )

        # remove neurons from nodes not in the graph
        if self.neurons_in_graph is not None:
            self.neurons_in_graph *= self.nodes_in_graph.view(-1, 1)

    @classmethod
    def from_model(
        cls,
        model_or_config: Union[HookedTransformer, HookedTransformerConfig, Dict],
        neuron_level: bool = False,
        node_scores: bool = False,
        mlp_hook: str = "mlp_out",
    ) -> "Graph":
        """
        Instantiate a fully-connected Graph from a model, config, or config dict.

        Builds all structurally valid edges (respecting causal ordering and
        parallel_attn_mlp topology). All edges start as real but none are
        in_graph; use apply_topn / apply_threshold to populate the circuit.
        Unscored nodes/neurons default to NaN and are always kept by selection
        methods.

        Args:
            model_or_config (Union[HookedTransformer, HookedTransformerConfig, Dict]):
                Source of model configuration. Must provide n_layers, n_heads,
                d_model, and parallel_attn_mlp. d_mlp defaults to 4*d_model if absent.
            neuron_level (bool): Allocate neurons_in_graph and neurons_scores
                tensors for per-neuron circuit discovery. Defaults to False.
            node_scores (bool): Allocate nodes_scores tensor initialised to NaN.
                Required for apply_topn / apply_threshold at node level.
                Defaults to False.
            mlp_hook (str): MLP hook granularity. 'mlp_out' attributes at the
                residual-stream level (d_model); 'post_act' attributes at
                post-activation neurons (d_mlp). Defaults to 'mlp_out'.

        Returns:
            Graph: Fully initialised graph with all valid edges registered in
                real_edge_mask and all components set to not in_graph.

        Raises:
            ValueError: If model_or_config is not a supported type.
        """
        graph = Graph()
        graph.cfg = GraphConfig()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            d_mlp = cfg.d_mlp if hasattr(cfg, "d_mlp") else 4 * cfg.d_model
            graph.cfg.update(
                {
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "parallel_attn_mlp": cfg.parallel_attn_mlp,
                    "d_model": cfg.d_model,
                    "d_mlp": d_mlp,
                }
            )
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            d_mlp = cfg.d_mlp if hasattr(cfg, "d_mlp") else 4 * cfg.d_model
            graph.cfg.update(
                {
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "parallel_attn_mlp": cfg.parallel_attn_mlp,
                    "d_model": cfg.d_model,
                    "d_mlp": d_mlp,
                }
            )
        elif isinstance(model_or_config, dict):
            d_mlp = model_or_config.get("d_mlp", 4 * model_or_config.get("d_model", 0))
            graph.cfg.update(model_or_config)
            graph.cfg["d_mlp"] = d_mlp
            graph.cfg["parallel_attn_mlp"] = model_or_config.get("parallel_attn_mlp", False)

            if "mlp_hook" in model_or_config:
                mlp_hook = model_or_config["mlp_hook"]

        else:
            raise ValueError(f"Invalid input type: {type(model_or_config)}")

        graph.cfg["mlp_hook"] = mlp_hook

        graph.n_forward = 1 + graph.cfg["n_layers"] * (graph.cfg["n_heads"] + 1)
        graph.n_backward = graph.cfg["n_layers"] * (3 * graph.cfg["n_heads"] + 1) + 1
        graph.forward_to_backward = torch.zeros((graph.n_forward, graph.n_backward)).bool()

        graph.scores = torch.zeros((graph.n_forward, graph.n_backward))
        graph.real_edge_mask = torch.zeros((graph.n_forward, graph.n_backward)).bool()
        graph.in_graph = torch.zeros((graph.n_forward, graph.n_backward)).bool()
        graph.nodes_in_graph = torch.zeros(graph.n_forward).bool()
        if node_scores:
            graph.nodes_scores = torch.zeros(graph.n_forward)
            graph.nodes_scores[:] = torch.nan
        else:
            graph.nodes_scores = None
        if neuron_level:
            max_d = (
                max(graph.cfg["d_model"], graph.cfg["d_mlp"])
                if graph.cfg["mlp_hook"] == "post_act"
                else graph.cfg["d_model"]
            )
            graph.neurons_in_graph = torch.zeros((graph.n_forward, max_d)).bool()
            graph.neurons_scores = torch.zeros((graph.n_forward, max_d))
            graph.neurons_scores[:] = torch.nan
        else:
            graph.neurons_in_graph = None
            graph.neurons_scores = None

        input_node = InputNode(graph)
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg["n_layers"]):
            attn_nodes = [AttentionNode(layer, head, graph) for head in range(graph.cfg["n_heads"])]
            mlp_node = MLPNode(layer, graph, mlp_hook=mlp_hook)

            for attn_node in attn_nodes:
                graph.nodes[attn_node.name] = attn_node
                for letter in "qkv":
                    graph.forward_to_backward[
                        graph.forward_index(attn_node, attn_slice=False),
                        graph.backward_index(attn_node, attn_slice=False, qkv=letter),
                    ] = True
            graph.nodes[mlp_node.name] = mlp_node
            graph.forward_to_backward[
                graph.forward_index(mlp_node, attn_slice=False),
                graph.backward_index(mlp_node, attn_slice=False),
            ] = True

            if graph.cfg["parallel_attn_mlp"]:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in "qkv":
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in "qkv":
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg["n_layers"], graph)
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        return graph

    def to_json(self, filename: str):
        """
        Serialise the graph to a JSON file.

        Saves cfg, per-node in_graph/score/neurons state, and per-edge
        score/in_graph values. Use from_json to reload.

        Note:
            JSON is not space-efficient for large neuron-level graphs.
            Prefer to_pt for production use.

        Args:
            filename (str): Output file path (will be created or overwritten).
        """
        # non serializable info
        d = {"cfg": dict(self.cfg)}
        node_dict = {}
        for node_name, node in self.nodes.items():
            node_dict[node_name] = {"in_graph": bool(node.in_graph)}
            if self.nodes_scores is not None:
                node_dict[node_name]["score"] = float(node.score)
            if self.neurons_in_graph is not None:
                node_dict[node_name]["neurons"] = self.neurons_in_graph[
                    self.forward_index(node)
                ].tolist()
                node_dict[node_name]["neurons_scores"] = self.neurons_scores[
                    self.forward_index(node)
                ].tolist()
        d["nodes"] = node_dict

        edge_dict = {}
        for edge_name, edge in self.edges.items():
            edge_dict[edge_name] = {"score": edge.score.item(), "in_graph": bool(edge.in_graph)}

        d["edges"] = edge_dict

        with open(filename, "w") as f:
            json.dump(d, f)

    def to_pt(self, filename: str):
        """
        Serialise the graph to a PyTorch .pt file.

        Saved dict keys:
            - 'cfg'             : GraphConfig dict.
            - 'src_nodes'       : List[str] of forward node names (excludes logits).
            - 'dst_nodes'       : List[str] of backward node names (from get_dst_nodes).
            - 'edges_scores'    : Tensor [n_forward, n_backward] of edge scores.
            - 'edges_in_graph'  : Tensor [n_forward, n_backward] bool.
            - 'nodes_in_graph'  : Tensor [n_forward] bool.
            - 'nodes_scores'    : Tensor [n_forward] (only if nodes_scores is set).
            - 'neurons_in_graph': Tensor [n_forward, max_d] (only if neuron level).
            - 'neurons_scores'  : Tensor [n_forward, max_d] (only if neuron level).

        Args:
            filename (str): Output file path (will be created or overwritten).
        """
        src_nodes = [node.name for node in self.nodes.values() if not isinstance(node, LogitNode)]
        dst_nodes = self.get_dst_nodes()
        d = {
            "cfg": dict(self.cfg),
            "src_nodes": src_nodes,
            "dst_nodes": dst_nodes,
            "edges_scores": self.scores,
            "edges_in_graph": self.in_graph,
            "nodes_in_graph": self.nodes_in_graph,
        }
        if self.nodes_scores is not None:
            d["nodes_scores"] = self.nodes_scores
        if self.neurons_in_graph is not None:
            d["neurons_in_graph"] = self.neurons_in_graph
            d["neurons_scores"] = self.neurons_scores
        torch.save(d, filename)

    @classmethod
    def from_json(cls, json_path: str) -> "Graph":
        """
        Load a Graph from a JSON file produced by to_json.

        Expected JSON structure:
            - 'cfg'   : dict compatible with from_model (n_layers, n_heads, etc.).
            - 'nodes' : Dict[str, dict] mapping node name to a dict with keys:
                - 'in_graph'        : bool
                - 'score'           : float (optional)
                - 'neurons'         : List[bool] (optional)
                - 'neurons_scores'  : List[float] (optional)
            - 'edges' : Dict[str, dict] mapping edge name to:
                - 'score'    : float
                - 'in_graph' : bool

        If no node in the file has a 'score'/'neurons'/'neurons_scores' field,
        the corresponding graph tensor is set to None.

        Note:
            Not space-efficient for large neuron-level graphs; prefer from_pt.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            Graph: Reconstructed graph with all scores and membership flags restored.
        """
        with open(json_path, "r") as f:
            d = json.load(f)
            _required = ["cfg", "nodes", "edges"]
            _missing = [k for k in _required if k not in d.keys()]
            if _missing:
                raise ValueError(
                    f"Circuit JSON file '{json_path}' is missing required "
                    f"top-level keys: {_missing}. A valid circuit JSON must "
                    f"contain all of: {_required}. Re-export the circuit with "
                    f"Graph.to_json(), or pass a file produced by CircuitKit."
                )

        g = Graph.from_model(d["cfg"], neuron_level=True, node_scores=True)
        any_node_scores, any_neurons, any_neurons_scores = False, False, False
        for name, node_dict in d["nodes"].items():
            if name == "logits":
                continue
            g.nodes[name].in_graph = node_dict["in_graph"]
            if "score" in node_dict:
                any_node_scores = True
                g.nodes[name].score = node_dict["score"]
            if "neurons" in node_dict:
                any_neurons = True
                g.neurons_in_graph[g.forward_index(g.nodes[name])] = torch.tensor(
                    node_dict["neurons"]
                ).float()
            if "neurons_scores" in node_dict:
                any_neurons_scores = True
                g.neurons_scores[g.forward_index(g.nodes[name])] = torch.tensor(
                    node_dict["neurons_scores"]
                ).float()

        if not any_node_scores:
            g.nodes_scores = None
        if not any_neurons:
            g.neurons_in_graph = None
        if not any_neurons_scores:
            g.neurons_scores = None

        for name, info in d["edges"].items():
            g.edges[name].score = info["score"]
            g.edges[name].in_graph = info["in_graph"]

        return g

    @classmethod
    def from_pt(cls, pt_path: str) -> "Graph":
        """
        Load a Graph from a PyTorch .pt file produced by to_pt.

        Required keys in the file:
            - 'cfg'             : dict compatible with from_model.
            - 'src_nodes'       : List[str] of forward node names.
            - 'dst_nodes'       : List[str] of backward node names.
            - 'edges_scores'    : Tensor [n_forward, n_backward].
            - 'edges_in_graph'  : Tensor [n_forward, n_backward] bool.
            - 'nodes_in_graph'  : Tensor [n_forward] bool.

        Optional keys:
            - 'nodes_scores'    : Tensor [n_forward].
            - 'neurons_in_graph': Tensor [n_forward, max_d] bool.
            - 'neurons_scores'  : Tensor [n_forward, max_d].

        Args:
            pt_path (str): Path to the .pt file.

        Returns:
            Graph: Reconstructed graph with all scores and membership flags restored.

        Raises:
            ValueError: If required keys are missing or tensor shapes mismatch.
        """
        # GraphConfig is a dict subclass which the safe unpickler rejects;
        # the file is written by our own to_pt(), so weights_only=False is
        # acceptable here. A future cleanup could save cfg as a plain dict.
        d = torch.load(pt_path, weights_only=False)  # noqa: S614
        required_keys = [
            "cfg",
            "src_nodes",
            "dst_nodes",
            "edges_scores",
            "edges_in_graph",
            "nodes_in_graph",
        ]
        _missing = [k for k in required_keys if k not in d.keys()]
        if _missing:
            raise ValueError(
                f"Circuit .pt file '{pt_path}' is missing required keys: "
                f"{_missing}. A valid circuit .pt file must contain all of: "
                f"{required_keys} (found: {sorted(d.keys())}). Re-export the "
                f"circuit with Graph.to_pt(), or pass a file produced by "
                f"CircuitKit."
            )
        if d["edges_scores"].shape != d["edges_in_graph"].shape:
            raise ValueError(
                f"Circuit .pt file '{pt_path}' is corrupt: 'edges_scores' has "
                f"shape {tuple(d['edges_scores'].shape)} but 'edges_in_graph' "
                f"has shape {tuple(d['edges_in_graph'].shape)}; they must "
                f"match. Re-export the circuit with Graph.to_pt()."
            )

        g = Graph.from_model(d["cfg"])

        g.in_graph[:] = d["edges_in_graph"]
        g.scores[:] = d["edges_scores"]
        g.nodes_in_graph[:] = d["nodes_in_graph"]

        if "nodes_scores" in d:
            g.nodes_scores = d["nodes_scores"]

        if "neurons_in_graph" in d:
            g.neurons_in_graph = d["neurons_in_graph"]

        if "neurons_scores" in d:
            g.neurons_scores = d["neurons_scores"]

        return g

    def to_image(
        self,
        filename: str,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str = "dot",
        seed: Optional[int] = None,
    ):
        """
        Render the current circuit as a PNG using Graphviz.

        Only nodes and edges currently in_graph are drawn. Edge thickness is
        scaled linearly between minimum_penwidth and maximum_penwidth based on
        the normalised absolute score. Edge colour encodes qkv type and score
        sign via get_color.

        Args:
            filename (str): Output image path (e.g. 'circuit.png').
            colorscheme (str): Matplotlib colormap name used to assign a random
                fill colour to each node. Defaults to 'Pastel2'.
            minimum_penwidth (float): Minimum edge thickness in points.
                Defaults to 0.6.
            maximum_penwidth (float): Maximum edge thickness in points.
                Defaults to 5.0.
            layout (str): Graphviz layout engine, e.g. 'dot', 'neato'.
                Defaults to 'dot'.
            seed (Optional[int]): NumPy random seed for reproducible node
                colours. Defaults to None.

        Raises:
            ImportError: If pygraphviz is not installed.
        """

        import pygraphviz as pgv

        g = pgv.AGraph(
            directed=True, bgcolor="white", overlap="false", splines="true", layout=layout
        )

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(
                    node.name,
                    fillcolor=colors[node.name],
                    color="black",
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )

        scores = self.scores.view(-1).abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                normalized_score = (
                    (abs(edge.score) - min_score) / (max_score - min_score)
                    if max_score != min_score
                    else abs(edge.score)
                )
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(
                    edge.parent.name,
                    edge.child.name,
                    penwidth=str(penwidth),
                    color=get_color(edge.qkv, edge.score),
                )
        g.draw(filename, prog="dot")
