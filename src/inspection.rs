use crate::graph::NodeAddress;

/// The metadata of a Graph
// TODO: Feedback edges
pub struct GraphInspection {
    /// All the nodes currently in the Graph (including those pending removal)
    pub nodes: Vec<NodeInspection>,
    /// Nodes that are not connected to any graph output in any chain
    pub unconnected_nodes: Vec<usize>,
    /// Node indices that are in the Graph, but will be removed as soon as it is safe.
    pub nodes_pending_removal: Vec<usize>,
    /// The indices of nodes connected to the graph output(s)
    pub graph_output_input_edges: Vec<EdgeInspection>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub graph_id: crate::graph::GraphId,
}

pub struct NodeInspection {
    pub name: String,
    pub address: NodeAddress,
    pub input_channels: Vec<String>,
    pub output_channels: Vec<String>,
    pub input_edges: Vec<EdgeInspection>,
    /// If this node is a Graph, this contains the inspection of the inner graph
    pub graph_inspection: Option<GraphInspection>,
}

/// Metadata for an edge.
pub struct EdgeInspection {
    pub source: EdgeSource,
    pub from_index: usize,
    pub to_index: usize,
}

/// Edge source type used for inspection. The index of a node is only valid for that specific GraphInspection.
pub enum EdgeSource {
    Node(usize),
    Graph,
}
