import numpy as np

from torch_geometric.data import HeteroData
from rmsds import (
    evaluate_all_rmsds,
    summarize_rmsds,
)


def evaluate_all_predictions(
    results: list[tuple[HeteroData, float]], ground_truth: tuple[HeteroData, float]
):
    best_pred = results[0][0]
    meter = evaluate_all_rmsds(ground_truth, best_pred)

    return summarize_rmsds(
        meter.complex_rmsd_list, meter.interface_rmsd_list, meter.ligand_rmsd_list
    )


def evaluate_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    eval_result = evaluate_pose(ground_truth, best_pred)
    rmsds = np.array(eval_result["rmsd"])
    reverse_diffusion_metrics = {
        "rmsds_lt2": (100 * (rmsds < 2).sum() / len(rmsds)),
        "rmsds_lt5": (100 * (rmsds < 5).sum() / len(rmsds)),
        "rmsds_lt10": (100 * (rmsds < 10).sum() / len(rmsds)),
        "rmsds_mean": rmsds.mean(),
        "rmsds_median": np.median(rmsds),
    }
    return reverse_diffusion_metrics


def evaluate_pose(data_list, samples_list):
    """
    Evaluate sampled pose vs. ground truth
    """
    all_rmsds = []
    rmsds_with_name = {}
    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        true_xyz = true_graph["ligand"].pos
        pred_xyz = pred_graph["ligand"].pos
        if true_xyz.shape != pred_xyz.shape:
            print(true_graph["name"], pred_graph["name"])
        assert true_xyz.shape == pred_xyz.shape
        rmsd = compute_rmsd(true_xyz, pred_xyz)
        all_rmsds.append(rmsd)
        rmsds_with_name[true_graph["name"]] = rmsd

    scores = {
        "rmsd": all_rmsds,
        "rmsds_with_name": rmsds_with_name,
    }

    return scores


def dump_predictions(args, results):
    with open(args.prediction_storage, "wb") as f:
        pickle.dump(results, f)


def load_predictions(args):
    with open(args.prediction_storage, "rb") as f:
        results = pickle.load(f)
    return results
