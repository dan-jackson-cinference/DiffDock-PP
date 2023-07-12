def evaluate_all_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    meter = evaluate_all_rmsds(ground_truth, best_pred)
    _ = meter.summarize()
    return meter


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


def dump_predictions(args, results):
    with open(args.prediction_storage, "wb") as f:
        pickle.dump(results, f)


def load_predictions(args):
    with open(args.prediction_storage, "rb") as f:
        results = pickle.load(f)
    return results
