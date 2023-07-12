import torch
from torch import nn


def evaluate_confidence(model: nn.Module, loader: DataLoader, args):
    all_confidences = []
    all_confidences_with_name = {}

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_pred = []
    all_loss = []
    print("loader len: ", len(loader))
    for data in tqdm(loader, total=len(loader)):
        # data, rmsd = batch
        # move to CUDA

        if args.num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()
            set_time(data, 0, 0, 0, batch_size=args.batch_size, device=device)
        try:
            with torch.no_grad():
                pred = model(data)
            # print("prediction",pred)
            all_pred.append(pred.detach().cpu())

        except RuntimeError as err:
            if "out of memory" in str(err):
                print("| WARNING: ran out of memory, skipping batch")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            raise err

    all_pred = torch.cat(all_pred).tolist()  # TODO -> maybe list inside

    return all_pred
