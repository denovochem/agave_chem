import time

import numpy as np

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.utils.chem_utils import canonicalize_reaction_smiles

mcs_mapper = MCSReactionMapper(mapper_name="mcs", mapper_weight=1)
neural_mapper = NeuralReactionMapper(mapper_name="neural_mapper", mapper_weight=1)
expert_mapper = TemplateReactionMapper("expert_default")


def calculate_metrics(y_true, y_probs, threshold=0.5):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_probs >= threshold).astype(int)

    # True Positives, True Negatives, False Positives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Basic Metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # AUC-ROC Calculation (Trapezoidal rule)
    # Sort probabilities and corresponding truth values
    desc_score_indices = np.argsort(y_probs)[::-1]
    y_probs = y_probs[desc_score_indices]
    y_true = y_true[desc_score_indices]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Calculate area under the curve using the trapezoidal rule
    auc = np.trapezoid(tpr, fpr)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "AUC": auc,
        "Confusion Matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


gold_reactions = []
with open(
    "/home/csnbritt/projects/denovochem_projects/agave_chem/scripts/benchmarks/gold_reactions_filtered.txt",
    "r",
) as file:
    for line in file:
        gold_reactions.append(line.strip())

mapped_count = 0
identical_count = 0
total_start = time.time()

for i, gold_reaction in enumerate(gold_reactions):
    identical = None

    rxn_start = time.time()
    reactants = gold_reaction.split(">>")[0]
    products = gold_reaction.split(">>")[1]
    products_list = products.split(".")
    sorted_products_list = sorted(products_list, key=len)
    biggest_product = sorted_products_list[-1]
    better_reaction = canonicalize_reaction_smiles(
        reactants + ">>" + biggest_product,
        remove_mapping=False,
        canonicalize_tautomer=True,
    )
    unmapped_better_reaction = canonicalize_reaction_smiles(
        better_reaction, remove_mapping=True
    )
    try:
        out = expert_mapper.map_reaction(unmapped_better_reaction)
    except Exception:
        print("oops")
    if time.time() - rxn_start > 5:
        print("SLOW:", time.time() - rxn_start)
    print(i, mapped_count, identical_count, identical)
print(time.time() - total_start)
