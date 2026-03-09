import torch
import torch.nn.functional as F
import os
import sys
import argparse
import pickle
import time

from g4satbench.utils.options import add_model_options
from g4satbench.utils.logger import Logger
from g4satbench.utils.utils import set_seed
from g4satbench.utils.format_print import FormatTable
from g4satbench.data.dataloader import get_dataloader
from g4satbench.models.gnn import GNN
from g4satbench.models.graph_model import GraphModel
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to be tested')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='testation splits')
    parser.add_argument('--test_sample_size', type=int, default=None, help='The number of instance in each testing splits')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'core_variable'], default=None, help='Directory with testating data')
    parser.add_argument('--decoding', type=str, choices=['standard', '2-clustering', 'multiple_assignments'], default='standard', help='Decoding techniques for satisfying assignment prediction')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--problem_types', type=str, nargs='+',
                        choices=['k-clique', 'k-domset', 'k-vercov', 'k-color', 'matching', 'k-indset', 'automorph'],
                        default=None, help='Training Problem types')
    parser.add_argument('--graph_layer_num', type=int, default=12, help='Number of gnn layers of the graph model')
    parser.add_argument('--dropout', type=bool, default=False, help='Whether to use dropout')
    parser.add_argument('--gragh_gnn_type', type=str, default='gcn', choices=['gnn', 'gcn', 'sage'],
                        help='Type of GNN for graph model')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max'],
                        help='Pooling method for graph model')

    parser.add_argument('--graph_model', type=bool, default=True,
                        help='Whether to test graph model')

    add_model_options(parser)
    opts = parser.parse_args()
    
    assert opts.checkpoint is not None

    set_seed(opts.seed)

    opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))

    dataset = '_'.join(opts.problem_types)
    splits_name = '_'.join(opts.test_splits)

    checkpoint_name = os.path.splitext(os.path.basename(opts.checkpoint))[0]

    if opts.task == 'assignment':
        opts.log = os.path.join(opts.log_dir, f'eval_task={opts.task}_dataset={dataset}_splits={splits_name}_decoding={opts.decoding}_n_iterations={opts.n_iterations}_checkpoint={checkpoint_name}.txt')
    else:
        opts.log = os.path.join(opts.log_dir, f'eval_task={opts.task}_dataset={dataset}_splits={splits_name}_n_iterations={opts.n_iterations}_checkpoint={checkpoint_name}.txt')
    
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)
    
    if not opts.graph_model:
        model = GNN(opts)
        model.to(opts.device)
    
        print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)
    
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.to(opts.device)
    else:
        graph_model_dict = {}
        for i, problem in enumerate(opts.problem_types):
            graph_model_dict[problem] = GraphModel(opts)

            print('Loading graph model checkpoint from %s..' % opts.checkpoint)
            if opts.device.type == 'cpu':
                checkpoint = torch.load(opts.checkpoint[i], map_location='cpu')
            else:
                checkpoint = torch.load(opts.checkpoint[i])

            graph_model_dict[problem].load_state_dict(checkpoint['state_dict'], strict=False)
            graph_model_dict[problem].to(opts.device)

    test_loader = get_dataloader(opts.test_dir, opts.test_splits, opts.test_sample_size, opts.problem_types, opts, 'test')

    print('Evaluating...')
    test_cnt = dict()
    test_tot = 0

    format_table_dict = {}
    for problem in opts.problem_types:
        format_table_dict[problem] = FormatTable()

    for problem in opts.problem_types:
        test_cnt[problem] = 0

    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        for problem in opts.test_problem_types:
            format_table_dict[problem].reset()

    if not opts.graph_model:
        model.eval()
    else:
        for problem in opts.test_problem_types:
            graph_model_dict[problem].eval()

    t0 = time.time()

    for data in test_loader:
        for key in list(data.keys()):
            data[key] = data[key].to(opts.device)
        batch_size = opts.batch_size

        with torch.no_grad():
            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                if opts.task == 'core_variable':
                    assert not opts.graph_model
                for problem in opts.test_problem_types:
                    if not opts.graph_model:
                        _, pred = model(data[problem])
                    else:
                        _, pred = graph_model_dict[problem](data[problem + '_graph'])

                    label = data[problem].y

                    format_table_dict[problem].update(pred, label)

            elif opts.task == 'assignment':
                for problem in opts.test_problem_types:
                    c_size = data[problem].c_size.sum().item()
                    c_batch = data[problem].c_batch
                    l_edge_index = data[problem].l_edge_index
                    c_edge_index = data[problem].c_edge_index

                    if not opts.graph_model:
                        _, v_pred = model(data[problem])
                    else:
                        _, v_pred = graph_model_dict[problem](data[problem + '_graph'])

                    v_assign = (v_pred > 0.5).float()
                    l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                    c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size),
                                        max=1)
                    sat_batch = (
                                scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data[problem].c_size).float()
                    test_cnt[problem] += sat_batch.sum().item()

            else:
                raise NotImplementedError

        test_tot += batch_size

    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        for problem in opts.problem_types:
            format_table_dict[problem].print_stats()
    else:
        assert opts.task == 'assignment'
        for problem in opts.problem_types:
            test_acc = test_cnt[problem] / test_tot
            print('Problem: %s, testating accuracy: %f' % (problem, test_acc))

    t = time.time() - t0
    print('Solving Time: %f' % t)


if __name__ == '__main__':
    main()
