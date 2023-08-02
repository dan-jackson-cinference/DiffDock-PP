import numpy as np
from nptyping import Float, NDArray, Shape
from scipy import spatial
from torch_geometric.data import HeteroData

from evaluation.compute_rmsd import (
    evaluate_all_rmsds,
    rigid_transform_Kabsch_3D,
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


def evaluate_rmsds(true_graph: HeteroData, pred_graph: HeteroData):
    rec_xyz = true_graph["receptor"].pos
    true_xyz = true_graph["ligand"].pos
    pred_xyz = pred_graph["ligand"].pos

    crmsd = compute_complex_rmsd(pred_xyz, true_xyz, rec_xyz)
    lrmsd = compute_ligand_rmsd(pred_xyz, true_xyz)
    irmsd = compute_interface_rmsd(pred_xyz, true_xyz, rec_xyz)

    return {"crmsd": crmsd, "lrmsd": lrmsd, "irmsd": irmsd}


def compute_complex_rmsd(
    ligand_coors_pred: NDArray[Shape["2, 2"], Float],
    ligand_coors_true: NDArray[Shape["2, 2"], Float],
    receptor_coors: NDArray[Shape["2, 2"], Float],
):
    complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
    complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

    R, t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    complex_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)

    return complex_rmsd


def compute_ligand_rmsd(
    ligand_coors_pred: NDArray[Shape["2, 2"], Float],
    ligand_coors_true: NDArray[Shape["2, 2"], Float],
):
    ligand_rmsd = compute_rmsd(ligand_coors_pred, ligand_coors_true)

    return ligand_rmsd


def compute_interface_rmsd(
    ligand_coors_pred: NDArray[Shape["2, 2"], Float],
    ligand_coors_true: NDArray[Shape["2, 2"], Float],
    receptor_coors: NDArray[Shape["2, 2"], Float],
):
    ligand_receptor_distance = spatial.distance.cdist(ligand_coors_true, receptor_coors)
    positive_tuple = np.where(ligand_receptor_distance < 8.0)

    active_ligand = positive_tuple[0]
    active_receptor = positive_tuple[1]

    ligand_coors_pred = ligand_coors_pred[active_ligand, :]
    ligand_coors_true = ligand_coors_true[active_ligand, :]
    receptor_coors = receptor_coors[active_receptor, :]

    complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
    complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

    R, t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    interface_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)

    return interface_rmsd


def compute_rmsd(
    pred: NDArray[Shape["2, 2"], Float], true: NDArray[Shape["2, 2"], Float]
) -> NDArray[Shape["2, 2"], Float]:
    return np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1)))
