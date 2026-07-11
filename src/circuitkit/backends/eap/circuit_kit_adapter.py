from collections import defaultdict

import numpy as np
import torch as t
from tqdm import tqdm  # Use standard tqdm for better script compatibility

# Import from the new EAP-IG library
from .graph import AttentionNode, Graph, MLPNode

# Assuming your circuit_kit files are accessible


# --- ADDED HELPER FUNCTION FROM ORIGINAL REPO ---
import logging

logger = logging.getLogger(__name__)

def calculate_manual_perplexity(model, tokenizer, predictions, device):
    """
    Calculates the average perplexity for a list of predictions.
    """
    total_neg_log_likelihood = 0
    total_tokens = 0

    with t.no_grad():
        for text in tqdm(predictions, desc="Calculating Perplexity"):
            if not text:
                continue

            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            seq_len = input_ids.shape[1]

            if seq_len < 2:
                continue

            logits = model(input_ids)[:, :-1, :]
            labels = input_ids[:, 1:]

            log_probs = t.nn.functional.log_softmax(logits, dim=-1)
            # Use view to flatten tensors for NLLLoss
            loss_fn = t.nn.NLLLoss(reduction="sum")
            neg_log_likelihood = loss_fn(log_probs.view(-1, model.cfg.d_vocab), labels.view(-1))

            total_neg_log_likelihood += neg_log_likelihood.item()
            total_tokens += seq_len - 1

    if total_tokens == 0:
        return float("inf")

    avg_neg_log_likelihood = total_neg_log_likelihood / total_tokens
    perplexity = t.exp(t.tensor(avg_neg_log_likelihood))
    return perplexity.item()


# --- END HELPER FUNCTION ---


def convert_eap_graph_to_circuitkit_scores(graph: Graph) -> dict[str, float]:
    """
    Converts node scores from an EAP-IG Graph object to the dictionary format
    expected by CircuitKit's pruning functions.

    Maps 'aL.hH' -> 'AL.H' and 'mL' -> 'MLP L'.
    """
    if graph.nodes_scores is None:
        raise ValueError(
            "The EAP-IG graph does not contain node scores. "
            "Run `attribute_node` before this step."
        )

    node_scores_dict = {}
    for node in graph.nodes.values():
        if isinstance(node, (AttentionNode, MLPNode)):
            # Get the score for this node
            score = node.score.item()
            # get absolute value for score
            score = abs(score)

            # Convert the node name to CircuitKit format
            if isinstance(node, AttentionNode):
                # EAP-IG: 'a1.h5' -> CircuitKit: 'A1.5'
                circuit_kit_name = f"A{node.layer}.{node.head}"
            elif isinstance(node, MLPNode):
                # EAP-IG: 'm1' -> CircuitKit: 'MLP 1'
                circuit_kit_name = f"MLP {node.layer}"
            else:
                continue  # Skip input/logit nodes

            node_scores_dict[circuit_kit_name] = score

    logger.info(f"Converted {len(node_scores_dict)} node scores from EAP-IG graph to CircuitKit format.")
    return node_scores_dict


def convert_eap_edge_scores_to_node_scores(graph: Graph) -> dict[str, float]:
    """
    Converts edge scores from an EAP-IG Graph object to node scores for CircuitKit.
    A node's score is calculated as the mean absolute score of all its outgoing edges.
    """
    outgoing_scores_by_node = defaultdict(list)
    for edge in graph.edges.values():
        parent_name = edge.parent.name
        score = edge.score.item()
        outgoing_scores_by_node[parent_name].append(abs(score))

    node_scores_dict = {}
    for eap_name, scores in outgoing_scores_by_node.items():
        if not scores:
            continue
        mean_score = np.mean(scores)
        circuit_kit_name = ""
        if eap_name.startswith("a"):
            layer, head = eap_name.split(".")
            circuit_kit_name = f"A{layer[1:]}.{head[1:]}"
        elif eap_name.startswith("m"):
            circuit_kit_name = f"MLP {eap_name[1:]}"
        if circuit_kit_name:
            node_scores_dict[circuit_kit_name] = mean_score

    logger.info(f"Converted {len(graph.edges)} edge scores into {len(node_scores_dict)} node scores.")
    return node_scores_dict
