import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import get_ts_metircs


def evaluate_model_link_prediction(args, device: torch.device, model: nn.Module, evaluate_idx_data_loader: DataLoader,
                                   graph: torch.Tensor, loss_func: nn.Module, status: str, kg_list: list):

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        for batch_ts, batch_label, batch_policy_set, batch_future_policy_set in evaluate_idx_data_loader:
            batch_ts = batch_ts.to(device)
            batch_label = batch_label.to(device).squeeze()
            
            embeds, _ = model[0](batch_ts, graph, kg_list, batch_policy_set.squeeze(), batch_future_policy_set.squeeze())
            predicts = model[1](embeds)

            loss = loss_func(input=predicts, target=batch_label)
            evaluate_losses.append(loss.item())
            if not args.ood:
                evaluate_metrics.append(get_ts_metircs(predicts=predicts, labels=batch_label))
            else:
                if status == 'val':
                    evaluate_metrics.append(get_ts_metircs(predicts=predicts[:40], labels=batch_label[:40]))
                else:
                    evaluate_metrics.append(get_ts_metircs(predicts=predicts[40:], labels=batch_label[40:]))

    if args.range == -1:
        return evaluate_losses, evaluate_metrics, None
    else:
        return evaluate_losses, evaluate_metrics, predicts.cpu().numpy()
