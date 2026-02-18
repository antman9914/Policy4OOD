import logging
import time
import os
import numpy as np
import pandas as pd
import warnings
import json
import torch
import torch.nn as nn

from models.Policy4OOD import Policy4OOD
from utils.utils import set_random_seed, get_parameter_sizes, create_optimizer
from evaluate_model_utils import evaluate_model_link_prediction
from utils.metrics import get_ts_metircs
from utils.DataLoader import get_idx_data_loader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_state_idx():
    state_dict, state_map = {}, {}
    state_list = pd.read_csv('processed_data/state_ref.csv')
    state_name, state_abbr = state_list['state_name'], state_list['state_abbr']
    idx = 0
    for i in range(len(state_name)):
        if state_abbr[i] != 'HI' and state_abbr[i] != 'AK':
            state_dict[state_name[i]] = state_abbr[i]
            state_map[state_abbr[i]] = idx
            idx += 1
    return state_map


def loadkg(args, kg_dir='KG'):
    """
    Load knowledge graph data from the KG directory.

    Returns:
        kg_list: List of lists following the KG directory structure.
                 Each entry corresponds to a state index and contains a list of
                 [adj, edge, node] numpy arrays for each subdirectory.
        kg_idx_array: numpy array with shape (49, 72) where each row contains
                      the kg_idx values at the index corresponding to the directory name.
    """
    kg_list = [[] for _ in range(49)]
    kg_idx_array = np.zeros((49, 72), dtype=np.int64) - 1
    state_map = get_state_idx()

    for state_dir in os.listdir(kg_dir):
        state_path = os.path.join(kg_dir, state_dir)
        if not os.path.isdir(state_path):
            continue

        state_idx = int(state_dir)

        # Load kg_idx.npy and assign to the corresponding row
        #####################################################
        # To conduct counterfactual analysis in temporal dimension, 
        # match state with state_map and modify the data within kg_idx_array
        #####################################################
        kg_idx_path = os.path.join(state_path, 'kg_idx.npy')
        if os.path.exists(kg_idx_path):
            kg_idx_array[state_idx] = np.load(kg_idx_path)

        # Load subdirectories containing adj.npy, edge.npy, node.npy
        subdirs = sorted([d for d in os.listdir(state_path)
                         if os.path.isdir(os.path.join(state_path, d))], key=int)

        for subdir in subdirs:
            subdir_path = os.path.join(state_path, subdir)
            adj = np.load(os.path.join(subdir_path, 'adj.npy'))
            edge = np.load(os.path.join(subdir_path, 'edge.npy'))
            node = np.load(os.path.join(subdir_path, 'node.npy'))
            kg_list[state_idx].append([torch.from_numpy(adj), torch.from_numpy(edge), torch.from_numpy(node)])

    return kg_list, kg_idx_array


def load_data(return_std=False):
    # Data preprocessing: Normalization
    graph = np.load('processed_data/adj.npy').astype(np.longlong)
    graph_ood = np.load('processed_data/adj_ood.npy').astype(np.longlong)
    orig_time_series = np.load('processed_data/ts.npy').astype(np.float32)
    graph, time_series = torch.from_numpy(graph), torch.from_numpy(orig_time_series).transpose(1, 2)
    graph_ood = torch.from_numpy(graph_ood)
    ts_mean = time_series.mean((0, 1), keepdim=True)
    ts_std = time_series.std((0, 1), keepdim=True) + 1e-6
    time_series = (time_series - ts_mean) / ts_std
    time_series[torch.isnan(time_series)] = 0.

    if return_std:
        return graph, time_series, ts_mean.numpy(), ts_std.numpy()
    else:
        return graph, graph_ood, time_series


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args()
    device = torch.device('cpu')

    graph, graph_ood, time_series = load_data()
    kg_list, kg_idx_array = loadkg(args)

    if args.range == -1:
        raw_train_ts, raw_val_ts, raw_test_ts = time_series[:, :48, :], time_series[:, 36:60, :], time_series[:, 48:, :]
        train_data_loader = get_idx_data_loader(raw_train_ts, kg_idx_array[:, :48], (args.input_step, args.output_step), args.batch_size, shuffle=True)
        val_data_loader = get_idx_data_loader(raw_val_ts, kg_idx_array[:, 36:60], (args.input_step, args.output_step), args.batch_size, shuffle=False)
        test_data_loader = get_idx_data_loader(raw_test_ts, kg_idx_array[:, 48:], (args.input_step, args.output_step), args.batch_size, shuffle=False)
    else:
        time_gap = args.input_step + args.output_step
        raw_train_ts, raw_val_ts, raw_test_ts = time_series[:, :48, :], time_series[:, args.range:args.range + time_gap, :], time_series[:, 48:, :]
        train_data_loader = get_idx_data_loader(raw_train_ts, kg_idx_array[:, :48], (args.input_step, args.output_step), args.batch_size, shuffle=True)
        val_data_loader = get_idx_data_loader(raw_val_ts, kg_idx_array[:, args.range:args.range+time_gap], (args.input_step, args.output_step), args.batch_size, shuffle=False)
        test_data_loader = get_idx_data_loader(raw_test_ts, kg_idx_array[:, 48:], (args.input_step, args.output_step), args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.run_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        os.makedirs(f"./logs/{args.run_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.run_name}/{args.save_model_name}/{str(time.time())}.log")
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        encoder = Policy4OOD(raw_train_ts.size(-1), args)
        
        pred_head = nn.Sequential(nn.Linear(args.hidden_dim + args.time_feat_dim, raw_train_ts.size(-1)), nn.ReLU(), nn.Linear(raw_train_ts.size(-1), args.output_step))
        model = nn.Sequential(encoder, pred_head)

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.run_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        
        # # Uncomment this part if you would like to train with GPU.
        # model = model.to(device)
        # graph = graph.to(device)
        # graph_ood = graph_ood.to(device)

        save_model_folder = f"./saved_models/{args.run_name}/{args.save_model_name}/"
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.run_name)
        best_result = 0.
        duration = 0
        save_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pkl")
        loss_func = nn.MSELoss()

        if args.inference_only:
            model.load_state_dict(torch.load(save_model_path, map_location=device))
        else:
            for epoch in range(args.num_epochs):
                model.train()
                train_losses, train_metrics = [], []
                for batch_ts, batch_label, batch_policy_set, batch_future_policy_set in train_data_loader:
                    batch_ts = batch_ts.to(device)
                    batch_label = batch_label.to(device).squeeze()
                    
                    embeds, vq_loss = model[0](batch_ts, graph, kg_list, batch_policy_set.squeeze(), batch_future_policy_set.squeeze())
                    predicts = model[1](embeds)
                    if not args.ood:
                        loss = loss_func(input=predicts, target=batch_label) + vq_loss
                        train_metrics.append(get_ts_metircs(predicts=predicts, labels=batch_label))
                    else:
                        loss = loss_func(input=predicts[:40], target=batch_label[:40])
                        train_metrics.append(get_ts_metircs(predicts=predicts[:40], labels=batch_label[:40]))
                    train_losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                val_losses, val_metrics, _ = evaluate_model_link_prediction(args=args, device=device,
                                                                        model=model,
                                                                        evaluate_idx_data_loader=val_data_loader,
                                                                        graph=graph,
                                                                        loss_func=loss_func, status='val', kg_list=kg_list)
                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
                test_losses, test_metrics, _ = evaluate_model_link_prediction(args=args, device=device,
                                                                        model=model,
                                                                        evaluate_idx_data_loader=test_data_loader,
                                                                        graph=graph,
                                                                        loss_func=loss_func, status='test', kg_list=kg_list)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                    
                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), False))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break

            # load the best model
            early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        val_losses, val_metrics, val_predict = evaluate_model_link_prediction(args=args, device=device,
                                                                    model=model,
                                                                    evaluate_idx_data_loader=val_data_loader,
                                                                    graph=graph,
                                                                    loss_func=loss_func, status='val', kg_list=kg_list)
        test_losses, test_metrics, test_predict = evaluate_model_link_prediction(args=args, device=device,
                                                                model=model,
                                                                evaluate_idx_data_loader=test_data_loader,
                                                                graph=graph,
                                                                loss_func=loss_func, status='test', kg_list=kg_list)
        if args.range != -1:
            os.makedirs('results', exist_ok=True)
            np.save('results/val_results.npy', val_predict)
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)
        save_result_folder = f"./saved_results/{args.run_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.run_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
