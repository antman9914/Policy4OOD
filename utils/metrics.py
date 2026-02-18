import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_ts_metircs(predicts: torch.Tensor, labels: torch.Tensor):
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    mae_3 = mean_absolute_error(labels[:, :3], predicts[:, :3])
    rmse_3 = root_mean_squared_error(labels[:, :3], predicts[:, :3])
    mae_6 = mean_absolute_error(labels[:, 3:6], predicts[:, 3:6])
    rmse_6 = root_mean_squared_error(labels[:, 3:6], predicts[:, 3:6])
    mae = mean_absolute_error(labels, predicts)
    rmse = root_mean_squared_error(labels, predicts)
    return {'mae_3': mae_3, 'rmse_3': rmse_3, 'mae_6': mae_6, 'rmse_6': rmse_6, 'mae': mae, 'rmse': rmse}
