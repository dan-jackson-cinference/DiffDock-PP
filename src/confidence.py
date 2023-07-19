import torch
from torch import Tensor, device, nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from geom_utils import set_time


def evaluate_confidence(
    model: nn.Module, loader: DataLoader, device: device, num_gpu: int = 1
) -> list[float]:
    model.eval()
    all_pred: list[Tensor] = []
    print("loader len: ", len(loader))
    for data in tqdm(loader, total=len(loader)):
        if num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()
            set_time(data, 0, 0, 0, batch_size=loader.batch_size, device=device)
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

    all_pred: list[float] = torch.cat(all_pred).tolist()  # TODO -> maybe list inside

    return all_pred
