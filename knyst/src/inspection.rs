//! # Inspection
//!
//! Metadata from the structs in this module can be used to visualise and/or
//! manipulate a graph based on the whole graph structure.
use crate::graph::NodeId;

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
    /// Number of inputs to the graph
    pub num_inputs: usize,
    /// Number of outputs from the graph
    pub num_outputs: usize,
    /// The ID of the graph
    pub graph_id: crate::graph::GraphId,
}

/// Metadata about a node in a graph
pub struct NodeInspection {
    /// The name of the node (usually the name of the Gen inside it)
    pub name: String,
    /// The address of the node, useable to schedule changes to the node or free it
    pub address: NodeId,
    /// The names of the inputs channels to the node
    pub input_channels: Vec<String>,
    /// The names of the output channels from the node
    pub output_channels: Vec<String>,
    /// Edges going into this node
    pub input_edges: Vec<EdgeInspection>,
    /// If this node is a Graph, this contains the inspection of the inner graph
    pub graph_inspection: Option<GraphInspection>,
}

/// Metadata for an edge.
#[allow(missing_docs)]
pub struct EdgeInspection {
    pub source: EdgeSource,
    pub from_index: usize,
    pub to_index: usize,
}

/// Edge source type used for inspection. The index of a node is only valid for that specific GraphInspection.
#[allow(missing_docs)]
pub enum EdgeSource {
    Node(usize),
    Graph,
}
