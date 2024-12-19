from .utils import *
import torcheval.metrics as ms


if is_notebook():
    from tqdm.notebook import tqdm as tqbar
else:
    from tqdm import tqdm as tqbar


@torch.inference_mode()
def infer(model, loader, batch_fun=lambda x, y: x, y_len=1, device="cpu", tq_bar=False):
    """
    Performs inference on a given model using a loader, updating batch-level metrics using the provided function.

    Args:
        model (torch.nn.Module): The model to run inference on.
        loader (DataLoader): The DataLoader to provide batches of data for inference.
        batch_fun (function, optional): A function that processes the model's predictions and the true values
                                        for each batch and returns a list of computations. Defaults to just returns the model output.
        y_len (int, optional): The number of elements in the target batch. Defaults to 1.
        device (str, optional): The device to compute metrics on, e.g., "cpu" or "cuda".

    Returns:
        tuple: List of metrics, as many as are output by batch_fun.
    """

    model_training = model.training

    model.eval()
    batch_metrics = []
    batch_num = 0

    enumerator = tqbar(enumerate(loader)) if tq_bar else enumerate(loader)

    for _, batch in enumerator:
        batch = [a.to(device) for a in batch]
        outputs = model(*batch[:-y_len]).to(device)
        Y = list(b.to(device) for b in batch[-y_len:])
        computed = batch_fun(outputs, *Y, batch_num=batch_num)
        if isinstance(computed, List):
            if len(batch_metrics) == 0:  # instantiate
                batch_metrics = [[m] for m in computed]
            else:
                for i, m in enumerate(computed):
                    batch_metrics[i].append(m)
        batch_num += 1

    model.train(model_training)

    return [
        torch.cat(m, dim=0) if isinstance(m, torch.Tensor) else m for m in batch_metrics
    ]


def eval_classifier(trainer):
    metrics = [
        ms.BinaryConfusionMatrix(),
        ms.BinaryAccuracy(),
        ms.BinaryPrecision(),
        ms.BinaryRecall(),
        ms.BinaryF1Score(),
    ]
    trainer.eval(torchevals=[])

    return [m.compute() for m in metrics]


def loader_columns(loader, columns=slice(-1, None), cpu=True):
    outs = [batch[columns] for batch in loader]
    out = torch.cat(outs, dim=0)
    if cpu:
        return out.cpu().numpy()
    return out
