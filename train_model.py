import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
import random
import time
from collections import defaultdict

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from g4satbench.utils.options import add_model_options
from g4satbench.utils.utils import set_seed, safe_log, safe_div
from g4satbench.utils.logger import Logger
from g4satbench.utils.format_print import FormatTable
from g4satbench.utils.loss_func import contrastive_loss
from g4satbench.data.dataloader import get_dataloader
from g4satbench.models.gnn import GNN
from g4satbench.models.graph_model import GraphModel
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['sat', 'unsat'], default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None, help='The number of instance in each training splits')
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained checkpoint')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the validating data')
    parser.add_argument('--valid_sample_size', type=int, default=None, help='The number of instance in each validating splits')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'core_variable'], default=None, help='Label')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--loss', type=str, choices=[None, 'supervised', 'unsupervised_1', 'unsupervised_2'], default=None, help='Loss type for assignment prediction')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between two model savings')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--problem_types', type=str, nargs='+', choices=['k-clique', 'k-domset', 'k-vercov', 'k-color', 'matching', 'k-indset', 'automorph', 'k-clique-medium', 'k-domset-medium', 'k-vercov-medium', 'k-color-medium', 'matching-medium', 'k-indset-medium', 'automorph-medium'], default=None, help='Training Problem types')
    parser.add_argument('--valid_problem_types', type=str, nargs='+', choices=['k-clique', 'k-domset', 'k-vercov', 'k-color', 'matching', 'k-indset', 'automorph', 'k-clique-medium', 'k-domset-medium', 'k-vercov-medium', 'k-color-medium', 'matching-medium', 'k-indset-medium', 'automorph-medium'], default=None, help='Validation Problem types')
    parser.add_argument('--graph_layer_num', type=int, default=12, help='Number of gnn layers of the graph model')
    parser.add_argument('--dropout', type=bool, default=False, help='Whether to use dropout')
    parser.add_argument('--gragh_gnn_type', type=str, default='gcn', choices=['gnn', 'gcn', 'sage'], help='Type of GNN for graph model')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max'], help='Pooling method for graph model')

    parser.add_argument('--beta', type=float, default=1., help='Beta for classification loss')
    parser.add_argument('--gamma', type=float, default=1., help='Gamma for graph classification loss')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--contrast_start_epoch', type=int, default=0, help='The epoch to start contrastive learning')
    parser.add_argument('--single_tune_start_epoch', type=int, default=100, help='The epoch to start single tuning')

    parser.add_argument('--print_interval', type=int, default=5000, help='Print interval')

    # Adjust training strategy
    parser.add_argument('--graph_model', type=bool, default=True, help='Whether to use graph model for each problem type')
    parser.add_argument('--freeze_g_emb', type=bool, default=False, help='Whether to freeze the graph embedding during contrastive learning')
    parser.add_argument('--sat_supervision', type=bool, default=True, help='Whether to use supervised loss for sat model')
    parser.add_argument('--graph_checkpoint', type=str, nargs='+', default=None, help='pretrained graph model checkpoint')

    parser.add_argument('--debug_mode', type=bool, default=False, help='Whether to use debug mode')
    parser.add_argument('--stage', type=int, default=2, help='Stage of the training')

    add_model_options(parser)

    opts = parser.parse_args()

    if opts.debug_mode:
        print('-' * 40)
        print('Debug mode is on')
        print('-' * 40)

    set_seed(opts.seed)

    dataset = '_'.join(opts.problem_types)
    splits_name = '_'.join(opts.train_splits)

    cur_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))

    if opts.task == 'assignment':
        exp_name = f'stage={opts.stage}_train_task={opts.task}_dataset={dataset}_splits={splits_name}_label={opts.label}_loss={opts.loss}/' + \
            f'{cur_time}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr:.0e}_weight_decay={opts.weight_decay:.0e}_seed={opts.seed}'
    else:
        exp_name = f'stage={opts.stage}_train_task={opts.task}_dataset={dataset}_splits={splits_name}/' + \
            f'{cur_time}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr:.0e}_weight_decay={opts.weight_decay:.0e}_seed={opts.seed}'
    
    if opts.checkpoint is not None:
        opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint, '../../', exp_name))
    else:
        opts.log_dir = os.path.join('runs', exp_name)

    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    if not opts.debug_mode:
        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(opts.checkpoint_dir, exist_ok=True)

        opts.log = os.path.join(opts.log_dir, 'log.txt')
        sys.stdout = Logger(opts.log, sys.stdout)
        sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)


    model = GNN(opts)
    model.to(opts.device)

    graph_model_dict = {}
    for problem in opts.problem_types:
        if opts.graph_model:
            graph_model_dict[problem] = GraphModel(opts)
        else:
            graph_model_dict[problem] = GNN(opts)
        graph_model_dict[problem].to(opts.device)

    if opts.checkpoint is not None:
        print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)

        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if opts.graph_checkpoint is not None:
        for i, problem in enumerate(opts.problem_types):
            print('Loading model checkpoint from %s..' % opts.graph_checkpoint[i])
            if opts.device.type == 'cpu':
                checkpoint = torch.load(opts.graph_checkpoint[i], map_location='cpu')
            else:
                checkpoint = torch.load(opts.graph_checkpoint[i])

            graph_model_dict[problem].load_state_dict(checkpoint['state_dict'], strict=False)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    graph_optimizer_dict = {}
    for problem in opts.problem_types:
        graph_optimizer_dict[problem] = optim.Adam(graph_model_dict[problem].parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    train_loader = get_dataloader(opts.train_dir, opts.train_splits, opts.train_sample_size, opts.problem_types, opts, 'train')

    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts.valid_splits, opts.valid_sample_size, opts.valid_problem_types, opts, 'valid')
    else:
        valid_loader = None

    if opts.scheduler is not None:
        graph_scheduler_dict = {}
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
            for problem in opts.problem_types:
                graph_scheduler_dict[problem] = ReduceLROnPlateau(graph_optimizer_dict[problem], factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)
            for problem in opts.problem_types:
                graph_scheduler_dict[problem] = StepLR(graph_optimizer_dict[problem], step_size=opts.lr_step_size, gamma=opts.lr_factor)

    # for printing
    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        format_table_dict = {}
        for problem in opts.problem_types:
            format_table_dict[problem] = FormatTable()
            format_table_dict['SAT_' + problem] = FormatTable()

    best_loss = dict()
    for problem in opts.problem_types:
        best_loss[problem] = float('inf')
    best_loss['contrast'] = float('inf')

    training_mode = 'joint'
    for epoch in range(opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = dict()
        train_cnt = dict()
        train_tot = dict()
        for problem in opts.problem_types:
            train_loss[problem] = 0
            train_cnt[problem] = 0
            train_tot[problem] = 0
        idx = 0

        contrast_loss_dict = defaultdict(list)
        classification_loss_dict = defaultdict(list)
        if opts.sat_supervision:
            sat_classification_loss_dict = defaultdict(list)

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            for problem in opts.problem_types:
                format_table_dict[problem].reset()
                format_table_dict['SAT_' + problem].reset()

        model.train()
        for problem in opts.problem_types:
            graph_model_dict[problem].train()
        for data in train_loader:
            for key in list(data.keys()):
                data[key] = data[key].to(opts.device)
            # data = data.to(opts.device)
            batch_size = opts.batch_size

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                if opts.task == 'core_variable':
                    assert not opts.graph_model
                if training_mode == 'joint':
                    for p_idx, problem in enumerate(opts.problem_types):
                        optimizer.zero_grad()
                        graph_optimizer_dict[problem].zero_grad()
                        if opts.sat_supervision:
                            emb, sat_pred = model(data[problem])
                        else:
                            emb, _ = model(data[problem])

                        if opts.graph_model:
                            graph_emb, graph_pred = graph_model_dict[problem](data[problem + '_graph'])
                        else:
                            graph_emb, graph_pred = graph_model_dict[problem](data[problem])

                        # remove the gradient of graph_emb
                        if opts.freeze_g_emb:
                            graph_emb = graph_emb.detach()

                        label = data[problem].y
                        class_loss = opts.gamma * F.binary_cross_entropy(graph_pred, label)
                        contrast_loss = contrastive_loss(emb, graph_emb, opts.temperature)
                        if opts.sat_supervision:
                            sat_loss = F.binary_cross_entropy(sat_pred, label)
                            class_loss += sat_loss
                            sat_classification_loss_dict[problem].append(sat_loss.item())

                        if epoch < opts.contrast_start_epoch:
                            loss = class_loss
                        else:
                            loss = opts.beta * class_loss + contrast_loss
                        train_loss[problem] += loss.item() * batch_size
                        train_tot[problem] += batch_size
                        contrast_loss_dict[problem].append(contrast_loss.item())
                        classification_loss_dict[problem].append(class_loss.item())
                        format_table_dict[problem].update(graph_pred, label)
                        format_table_dict['SAT_' + problem].update(sat_pred, label)
                        if p_idx == 0:
                            iter_loss = loss
                        else:
                            iter_loss += loss
                    iter_loss /= len(opts.problem_types)
                    iter_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
                    optimizer.step()
                    for problem in opts.problem_types:
                        torch.nn.utils.clip_grad_norm_(graph_model_dict[problem].parameters(), opts.clip_norm)
                        graph_optimizer_dict[problem].step()

                else:
                    for problem in opts.problem_types:
                        if problem == training_mode:
                            optimizer.zero_grad()
                            graph_optimizer_dict[problem].zero_grad()
                        if opts.sat_supervision:
                            emb, sat_pred = model(data[problem])
                        else:
                            emb, _ = model(data[problem])

                        if opts.graph_model:
                            graph_emb, graph_pred = graph_model_dict[problem](data[problem + '_graph'])
                        else:
                            graph_emb, graph_pred = graph_model_dict[problem](data[problem])

                        # remove the gradient of graph_emb
                        if opts.freeze_g_emb:
                            graph_emb = graph_emb.detach()

                        label = data[problem].y
                        class_loss = opts.gamma * F.binary_cross_entropy(graph_pred, label)
                        contrast_loss = contrastive_loss(emb, graph_emb, opts.temperature)

                        if opts.sat_supervision:
                            sat_loss = F.binary_cross_entropy(sat_pred, label)
                            class_loss += sat_loss
                            sat_classification_loss_dict[problem].append(sat_loss.item())

                        if epoch < opts.contrast_start_epoch:
                            loss = class_loss
                        else:
                            loss = opts.beta * class_loss + contrast_loss
                        train_loss[problem] += loss.item() * batch_size
                        train_tot[problem] += batch_size
                        if problem == training_mode:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
                            torch.nn.utils.clip_grad_norm_(graph_model_dict[problem].parameters(), opts.clip_norm)
                            optimizer.step()
                            graph_optimizer_dict[problem].step()
                        contrast_loss_dict[problem].append(contrast_loss.item())
                        classification_loss_dict[problem].append(class_loss.item())

                        format_table_dict[problem].update(graph_pred, label)

            elif opts.task == 'assignment':
                if training_mode == 'joint':
                    for problem in opts.problem_types:
                        optimizer.zero_grad()
                        graph_optimizer_dict[problem].zero_grad()

                        c_size = data[problem].c_size.sum().item()
                        c_batch = data[problem].c_batch
                        l_edge_index = data[problem].l_edge_index
                        c_edge_index = data[problem].c_edge_index

                        emb, _ = model(data[problem])

                        if opts.graph_model:
                            graph_emb, v_pred = graph_model_dict[problem](data[problem + '_graph'])
                        else:
                            graph_emb, v_pred = graph_model_dict[problem](data[problem])

                        contrast_loss = contrastive_loss(emb, graph_emb, opts.temperature)
                        if opts.loss == 'supervised':
                            label = data[problem].y
                            class_loss = F.binary_cross_entropy(v_pred, label)

                        elif opts.loss == 'unsupervised_1':
                            assert not opts.graph_model
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                            s_max_nom = l_pred[l_edge_index] * s_max_denom

                            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                            c_pred = safe_div(c_nom, c_denom)

                            s_min_denom = (-c_pred / 0.1).exp()
                            s_min_nom = c_pred * s_min_denom
                            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                            score = safe_div(s_nom, s_denom)
                            class_loss = (1 - score).mean()

                        elif opts.loss == 'unsupervised_2':
                            assert not opts.graph_model
                            # calculate the loss in Eq. 6
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                            c_loss = -safe_log(1 - l_pred_aggr.exp())
                            class_loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
                        else:
                            raise NotImplementedError

                        v_assign = (v_pred > 0.5).float()
                        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data[problem].c_size).float()

                        train_cnt[problem] += sat_batch.sum().item()

                        if epoch < opts.contrast_start_epoch:
                            loss = class_loss
                        else:
                            loss = opts.beta * class_loss + contrast_loss
                        train_loss[problem] += loss.item() * batch_size
                        train_tot[problem] += batch_size
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
                        optimizer.step()
                        graph_optimizer_dict[problem].step()
                        contrast_loss_dict[problem].append(contrast_loss.item())
                        classification_loss_dict[problem].append(class_loss.item())
                else:
                    for problem in opts.problem_types:
                        if problem == training_mode:
                            optimizer.zero_grad()
                            graph_optimizer_dict[problem].zero_grad()

                        c_size = data[problem].c_size.sum().item()
                        c_batch = data[problem].c_batch
                        l_edge_index = data[problem].l_edge_index
                        c_edge_index = data[problem].c_edge_index

                        emb, _ = model(data[problem])

                        if opts.graph_model:
                            graph_emb, v_pred = graph_model_dict[problem](data[problem + '_graph'])
                        else:
                            graph_emb, v_pred = graph_model_dict[problem](data[problem])

                        contrast_loss = contrastive_loss(emb, graph_emb, opts.temperature)
                        if opts.loss == 'supervised':
                            label = data[problem].y
                            class_loss = F.binary_cross_entropy(v_pred, label)

                        elif opts.loss == 'unsupervised_1':
                            assert not opts.graph_model
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                            s_max_nom = l_pred[l_edge_index] * s_max_denom

                            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                            c_pred = safe_div(c_nom, c_denom)

                            s_min_denom = (-c_pred / 0.1).exp()
                            s_min_nom = c_pred * s_min_denom
                            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                            score = safe_div(s_nom, s_denom)
                            class_loss = (1 - score).mean()

                        elif opts.loss == 'unsupervised_2':
                            assert not opts.graph_model
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0,
                                                      dim_size=c_size)
                            c_loss = -safe_log(1 - l_pred_aggr.exp())
                            class_loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
                        else:
                            raise NotImplementedError

                        v_assign = (v_pred > 0.5).float()
                        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size),
                                            max=1)
                        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data[problem].c_size).float()

                        train_cnt[problem] += sat_batch.sum().item()

                        if epoch < opts.contrast_start_epoch:
                            loss = class_loss
                        else:
                            loss = opts.beta * class_loss + contrast_loss
                        train_loss[problem] += loss.item() * batch_size
                        train_tot[problem] += batch_size
                        if problem == training_mode:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
                            optimizer.step()
                            graph_optimizer_dict[problem].step()
                        contrast_loss_dict[problem].append(contrast_loss.item())
                        classification_loss_dict[problem].append(class_loss.item())
            
            else:
                raise NotImplementedError

            idx += 1
            if idx % opts.print_interval == 0:
                avg_cont_loss = 0
                avg_class_loss = 0
                print('Iteration {}/{}'.format(idx, len(train_loader)))
                for problem_type in opts.problem_types:
                    constrast_loss_mean = sum(contrast_loss_dict[problem_type][-opts.print_interval:]) / opts.print_interval
                    classification_loss_mean = sum(classification_loss_dict[problem_type][-opts.print_interval:]) / opts.print_interval
                    if opts.sat_supervision:
                        sat_classification_loss_mean = sum(sat_classification_loss_dict[problem_type][-opts.print_interval:]) / opts.print_interval
                        print('Problem type: %s, Contrastive loss: %.4f, Graph Classification loss: %.4f, SAT Classification loss: %.4f' % (
                            problem_type, constrast_loss_mean, classification_loss_mean - sat_classification_loss_mean, sat_classification_loss_mean))
                    else:
                        print('Problem type: %s, Contrastive loss: %.4f, Classification loss: %.4f' % (
                            problem_type, constrast_loss_mean, classification_loss_mean))

                    avg_cont_loss += constrast_loss_mean
                    avg_class_loss += classification_loss_mean
                avg_cont_loss /= len(opts.problem_types)
                avg_class_loss /= len(opts.problem_types)
                print('Avg contrastive loss: %.4f, Avg classification loss: %.4f' % (avg_cont_loss, avg_class_loss))

            if epoch >= opts.single_tune_start_epoch and idx > 100:
                # compute the problem with the highest average contrastive loss of last 100 iterations
                max_contrast_loss = 0
                for problem in opts.problem_types:
                    avg_contrast_loss = sum(contrast_loss_dict[problem][-100:]) + sum(classification_loss_dict[problem][-100:])
                    if avg_contrast_loss > max_contrast_loss:
                        max_contrast_loss = avg_contrast_loss
                        training_mode = problem

        for problem in opts.problem_types:
            train_loss[problem] /= train_tot[problem]
            print('Problem: %s, Training LR: %f, Training loss: %f' % (problem, graph_optimizer_dict[problem].param_groups[0]['lr'], train_loss[problem]))

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            print('Graph Model Performance:')
            for problem in opts.problem_types:
                format_table_dict[problem].print_stats()
            print('SAT Model Performance:')
            for problem in opts.problem_types:
                format_table_dict['SAT_' + problem].print_stats()
        else:
            assert opts.task == 'assignment'
            for problem in opts.problem_types:
                train_acc = train_cnt[problem] / train_tot[problem]
                print('Problem: %s, Training accuracy: %f' % (problem, train_acc))

        if epoch % opts.save_model_epochs == 0 and not opts.debug_mode:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )

            for save_p in opts.problem_types:
                torch.save({
                    'state_dict': graph_model_dict[save_p].state_dict(),
                    'epoch': epoch,
                    'optimizer': graph_optimizer_dict[save_p].state_dict()},
                    os.path.join(opts.checkpoint_dir, 'graph_model_%s_%d.pt' % (save_p, epoch))
                )

        if opts.valid_dir is not None:
            print('Validating...')
            valid_loss = dict()
            valid_cnt = dict()
            valid_tot = 0

            for problem in opts.valid_problem_types:
                valid_loss[problem] = 0
                valid_loss['contrast_' + problem] = 0
                valid_cnt[problem] = 0

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                for problem in opts.valid_problem_types:
                    format_table_dict[problem].reset()
                    format_table_dict['SAT_' + problem].reset()

            model.eval()
            for problem in opts.valid_problem_types:
                graph_model_dict[problem].eval()
            for data in valid_loader:
                loss = dict()
                contrast_loss = dict()
                for key in list(data.keys()):
                    data[key] = data[key].to(opts.device)
                batch_size = opts.batch_size

                with torch.no_grad():
                    if opts.task == 'satisfiability' or opts.task == 'core_variable':
                        if opts.task == 'core_variable':
                            assert not opts.graph_model
                        for problem in opts.valid_problem_types:
                            emb, sat_pred = model(data[problem])

                            if opts.graph_model:
                                graph_emb, graph_pred = graph_model_dict[problem](data[problem + '_graph'])
                            else:
                                graph_emb, graph_pred = graph_model_dict[problem](data[problem])

                            label = data[problem].y
                            loss[problem] = F.binary_cross_entropy(graph_pred, label)
                            contrast_loss[problem] = contrastive_loss(emb, graph_emb, opts.temperature)

                            format_table_dict[problem].update(graph_pred, label)
                            format_table_dict['SAT_' + problem].update(sat_pred, label)
                    
                    elif opts.task == 'assignment':
                        for problem in opts.valid_problem_types:
                            c_size = data[problem].c_size.sum().item()
                            c_batch = data[problem].c_batch
                            l_edge_index = data[problem].l_edge_index
                            c_edge_index = data[problem].c_edge_index

                            emb, _ = model(data[problem])

                            if opts.graph_model:
                                graph_emb, v_pred = graph_model_dict[problem](data[problem + '_graph'])
                            else:
                                graph_emb, v_pred = graph_model_dict[problem](data[problem])
                            contrast_loss[problem] = contrastive_loss(emb, graph_emb, opts.temperature)

                            if opts.loss == 'supervised':
                                label = data[problem].y
                                loss[problem] = F.binary_cross_entropy(v_pred, label)

                            elif opts.loss == 'unsupervised_1':
                                # calculate the loss in Eq. 4 and Eq. 5
                                l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                                s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                                s_max_nom = l_pred[l_edge_index] * s_max_denom

                                c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                                c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                                c_pred = safe_div(c_nom, c_denom)

                                s_min_denom = (-c_pred / 0.1).exp()
                                s_min_nom = c_pred * s_min_denom
                                s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                                s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                                score = safe_div(s_nom, s_denom)
                                loss[problem] = (1 - score).mean()

                            elif opts.loss == 'unsupervised_2':
                                # calculate the loss in Eq. 6
                                l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                                l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                                c_loss = -safe_log(1 - l_pred_aggr.exp())
                                loss[problem] = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
                            else:
                                raise NotImplementedError

                            v_assign = (v_pred > 0.5).float()
                            l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                            c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                            sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data[problem].c_size).float()
                            valid_cnt[problem] += sat_batch.sum().item()
                    
                    else:
                        raise NotImplementedError

                for problem in opts.valid_problem_types:
                    valid_loss[problem] += loss[problem].item() * batch_size
                    valid_loss['contrast_' + problem] += contrast_loss[problem].item() * batch_size
                valid_tot += batch_size

            for problem in opts.valid_problem_types:
                valid_loss[problem] /= valid_tot
                print('Problem: %s, Validating loss: %f' % (problem, valid_loss[problem]))

            for problem in opts.valid_problem_types:
                valid_loss['contrast_' + problem] /= valid_tot
                print('Problem: %s, Validating contrast loss: %f' % (problem, valid_loss['contrast_' + problem]))

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                print('Graph Model Performance:')
                for problem in opts.valid_problem_types:
                    format_table_dict[problem].print_stats()
                print('SAT Model Performance:')
                for problem in opts.valid_problem_types:
                    format_table_dict['SAT_' + problem].print_stats()
            else:
                assert opts.task == 'assignment'
                for problem in opts.valid_problem_types:
                    valid_acc = valid_cnt[problem] / valid_tot
                    print('Problem: %s, Validating accuracy: %f' % (problem, valid_acc))

            if not opts.debug_mode:
                if opts.stage == 2:
                    for problem in opts.valid_problem_types:
                        if valid_loss['contrast_' + problem] < best_loss[problem]:
                            best_loss[problem] = valid_loss['contrast_' + problem]
                            torch.save({
                                'state_dict': graph_model_dict[problem].state_dict(),
                                'epoch': epoch,
                                'optimizer': graph_optimizer_dict[problem].state_dict()},
                                os.path.join(opts.checkpoint_dir, 'best_graph_model_%s.pt' % problem)
                            )
                else:
                    for problem in opts.valid_problem_types:
                        if valid_loss[problem] < best_loss[problem]:
                            best_loss[problem] = valid_loss[problem]
                            torch.save({
                                'state_dict': graph_model_dict[problem].state_dict(),
                                'epoch': epoch,
                                'optimizer': graph_optimizer_dict[problem].state_dict()},
                                os.path.join(opts.checkpoint_dir, 'best_graph_model_%s.pt' % problem)
                            )

            contrast_loss_list = [valid_loss['contrast_' + p] for p in opts.valid_problem_types]
            mean_constrast_loss = sum(contrast_loss_list) / len(contrast_loss_list)
            if mean_constrast_loss < best_loss['contrast'] and not opts.debug_mode:
                best_loss['contrast'] = mean_constrast_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()},
                    os.path.join(opts.checkpoint_dir, 'best_model.pt')
                )

            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    for problem in opts.problem_types:
                        graph_scheduler_dict[problem].step(valid_loss[problem])
                    scheduler.step(mean_constrast_loss)
                else:
                    scheduler.step()
                    for problem in opts.problem_types:
                        graph_scheduler_dict[problem].step()
        else:
            if opts.scheduler is not None:
                scheduler.step()
                for problem in opts.problem_types:
                    graph_scheduler_dict[problem].step()


if __name__ == '__main__':
    main()
