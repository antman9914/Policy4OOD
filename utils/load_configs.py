import argparse
import sys


def get_link_prediction_args():
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--run_name', type=str, default='base', help='name of the model, note that EdgeBank is only applicable for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use. However, Policy4OOD is trained on CPU by default.')
    parser.add_argument('--input_step', type=int, default=6, help="number of time steps for model input")
    parser.add_argument('--output_step', type=int, default=6, help="number of time steps for model prediction.")
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--enc_layers', type=int, default=1, help='number of model layers')
    parser.add_argument('--text_dim', type=int, default=384, help='dimension of raw policy text embedding')
    parser.add_argument('--policy_dim', type=int, default=6, help='dimension of policy encoding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden layer')
    parser.add_argument('--time_feat_dim', type=int, default=16, help='dimension of the time embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--ood', action="store_true", default=False, help="whether evaluate in OOD setting")
    parser.add_argument('--inference_only', action="store_true", default=False, help="whether skip pre-train and do inference")
    parser.add_argument('--range', type=int, default=-1, help='Specify a time range for case study')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    return args
