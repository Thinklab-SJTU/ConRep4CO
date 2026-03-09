import os
import glob
import torch
import pickle
import itertools
import networkx as nx

from collections import defaultdict
from torch_geometric.data import Dataset, Data
from g4satbench.utils.utils import parse_cnf_file, clean_clauses
from g4satbench.data.data import construct_lcg, construct_vcg


class SATDataset(Dataset):
    def __init__(self, data_dir, splits, sample_size, use_contrastive_learning, mode, problem_types, opts):
        self.opts = opts
        self.splits = splits
        self.problem_types = problem_types
        self.mode = mode
        self.sample_size = sample_size
        self.root_dir = data_dir

        self.all_files = {}
        self.all_labels = {}
        self.all_graph_files = {}
        for problem in self.problem_types:
            data_dir = os.path.join(self.root_dir, problem, mode)
            self.all_files[problem] = self._get_files(data_dir)
        self.split_len = self._get_split_len()
        for problem in self.problem_types:
            data_dir = os.path.join(self.root_dir, problem, mode)
            self.all_labels[problem] = self._get_labels(problem)
            # fetch graph files, if they exist
            try:
                self.all_graph_files[problem] = self._get_graph_files(data_dir)
            except:
                pass
            super().__init__(data_dir)

        self.use_contrastive_learning = use_contrastive_learning
        if self.use_contrastive_learning:
            self.positive_indices = self._get_positive_indices()
            
        # super().__init__(data_dir)
    
    def _get_files(self, data_dir):
        files = {}
        for split in self.splits:
            # For accelerating the data fetching process, we use the file id to fetch the data.
            split_files = []
            if data_dir.split('/')[-1] == 'train':
                for file_id in range(80000):
                    file_id = str(file_id).zfill(5)
                    split_files.append(data_dir + f'/{split}/{file_id}.cnf')
            else:
                for file_id in range(10000):
                    file_id = str(file_id).zfill(5)
                    split_files.append(data_dir + f'/{split}/{file_id}.cnf')

            # Here is the original code for fetching the data.
            # split_files = list(sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True)))

            if self.sample_size is not None and len(split_files) > self.sample_size:
                split_files = split_files[:self.sample_size]
            files[split] = split_files
        return files

    def _get_graph_files(self, data_dir):
        graph_files = {}
        for split in self.splits:
            # For accelerating the data fetching process, we use the file id to fetch the data.
            split_files = []
            if data_dir.split('/')[-1] == 'train':
                for file_id in range(80000):
                    file_id = str(file_id).zfill(5)
                    split_files.append(data_dir + f'/{split}_graph/{file_id}.gml')
            else:
                for file_id in range(10000):
                    file_id = str(file_id).zfill(5)
                    split_files.append(data_dir + f'/{split}_graph/{file_id}.gml')

            # Here is the original code for fetching the data.
            # split_files = list(sorted(glob.glob(data_dir + f'/{split}_graph/*.gml', recursive=True)))

            if self.sample_size is not None and len(split_files) > self.sample_size:
                split_files = split_files[:self.sample_size]
            graph_files[split] = split_files
        return graph_files
    
    def _get_labels(self, problem):
        labels = {}
        if self.opts.label == 'satisfiability':
            for split in self.splits:
                if split == 'sat' or split == 'augmented_sat':
                    labels[split] = [torch.tensor(1., dtype=torch.float)] * self.split_len
                else:
                    # split == 'unsat' or split == 'augmented_unsat'
                    labels[split] = [torch.tensor(0., dtype=torch.float)] * self.split_len
        elif self.opts.label == 'assignment':
            for split in self.splits:
                assert split == 'sat' or split == 'augmented_sat'
                labels[split] = []
                for cnf_filepath in self.all_files[problem][split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_assignment.pkl')
                    with open(assignment_file, 'rb') as f:
                        assignment = pickle.load(f)
                    labels[split].append(torch.tensor(assignment, dtype=torch.float))
        elif self.opts.label == 'core_variable':
            for split in self.splits:
                assert split == 'unsat' or split == 'augmented_unsat'
                labels[split] = []
                for cnf_filepath in self.all_files[problem][split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_core_variable.pkl')
                    with open(assignment_file, 'rb') as f:
                        core_variable = pickle.load(f)
                    labels[split].append(torch.tensor(core_variable, dtype=torch.float))
        else:
            assert self.opts.label == None
            for split in self.splits:
                labels[split] = [None] * self.split_len
        
        return labels

    def _get_split_len(self):
        lens = [len(self.all_files[p][split]) for p in self.problem_types for split in self.splits]
        assert len(set(lens)) == 1
        return lens[0]
    
    def _get_file_name(self, split, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        return f'{split}/{filename}_{self.opts.graph}.pt'

    @staticmethod
    def _get_graph_file_name(split, graph_filepath):
        filename = os.path.splitext(os.path.basename(graph_filepath))[0]
        return f'{split}_graph/{filename}.gml'

    def _get_positive_indices(self):
        # calculate the index to map the original instance to its augmented one, and vice versa.
        positive_indices = []
        for offset, split in enumerate(self.splits):
            if split == 'sat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_sat')-offset, dtype=torch.long))
            elif split == 'augmented_sat':
                positive_indices.append(torch.tensor(self.splits.index('sat')-offset, dtype=torch.long))
            elif split == 'unsat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_unsat')-offset, dtype=torch.long))
            elif split == 'augmented_unsat':
                positive_indices.append(torch.tensor(self.splits.index('unsat')-offset, dtype=torch.long))
        return positive_indices
    
    @property
    def processed_file_names(self):
        problem = self.processed_dir.split('/')[-3]

        names = []
        for split in self.splits:
            for cnf_filepath in self.all_files[problem][split]:
                names.append(self._get_file_name(split, cnf_filepath))
        return names

    def _save_data(self, split, cnf_filepath):
        file_name = self._get_file_name(split, cnf_filepath)
        saved_path = os.path.join(self.processed_dir, file_name)
        if os.path.exists(saved_path):
            return

        try:
            n_vars, clauses, learned_clauses = parse_cnf_file(cnf_filepath, split_clauses=True)
        except:
            n_vars, clauses = parse_cnf_file(cnf_filepath, split_clauses=False)
            learned_clauses = []
        
        # limit the size of the learned clauses to 1000
        if len(learned_clauses) > 1000:
            clauses = clauses + learned_clauses[:1000]
        else:
            clauses = clauses + learned_clauses
        
        clauses = clean_clauses(clauses)

        if self.opts.graph == 'lcg':
            data = construct_lcg(n_vars, clauses)
        elif self.opts.graph == 'vcg':
            data = construct_vcg(n_vars, clauses)

        torch.save(data, saved_path)

    @staticmethod
    def _load_graph_data(graph_file_path):
        graph = nx.read_gml(graph_file_path)
        init_edges = list(graph.edges())
        # make the elements in the tuple in graph.edges() to be int
        edges = [(int(u), int(v)) for u, v in init_edges]

        k_path = graph_file_path.replace('.gml', '.json')
        if os.path.exists(k_path):
            with open(k_path, 'r') as f:
                str_k = f.read()
                k = torch.tensor(int(str_k), dtype=torch.float32)
        else:
            k = torch.tensor(0, dtype=torch.float32)

        cliques = nx.find_cliques(graph)
        max_clique_size = max([len(clique) for clique in cliques]) / graph.number_of_nodes()
        min_vertex_cover = nx.approximation.min_weighted_vertex_cover(graph)
        min_vertex_cover_size = len(min_vertex_cover) / graph.number_of_nodes()



        # init_graph_feature = torch.ones((graph.number_of_nodes(), 3))
        # for i in range(graph.number_of_nodes()):
            # init_graph_feature[i][2] = i / graph.number_of_nodes()

        return Data(num_nodes=graph.number_of_nodes(), edge_index=torch.tensor(edges).t().contiguous(), k=k, max_clique_size=max_clique_size, min_vertex_cover_size=min_vertex_cover_size)


    def process(self):
        for split in self.splits:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)

        problem = self.processed_dir.split('/')[-3]
        
        for split in self.splits:
            for cnf_filepath in self.all_files[problem][split]:
                self._save_data(split, cnf_filepath)
    
    def len(self):
        if self.opts.data_fetching == 'parallel':
            return self.split_len
        else:
            # self.opts.data_fetching == 'sequential'
            return self.split_len * len(self.splits)

    def get(self, idx):
        if self.opts.data_fetching == 'parallel':
            data_dict = defaultdict(list)
            for problem in self.problem_types:
                data_dir = os.path.join(self.root_dir, problem, self.mode)
                for split_idx, split in enumerate(self.splits):
                    cnf_filepath = self.all_files[problem][split][idx]
                    label = self.all_labels[problem][split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(data_dir, 'processed', file_name)
                    data = torch.load(saved_path)
                    data.y = label
                    if self.use_contrastive_learning:
                        data.positive_index = self.positive_indices[split_idx]
                    data_dict[problem].append(data)

                    try:
                        graph_filepath = self.all_graph_files[problem][split][idx]
                        graph_file_name = self._get_graph_file_name(split, graph_filepath)
                        saved_graph_path = os.path.join(data_dir, graph_file_name)
                        graph_data = self._load_graph_data(saved_graph_path)
                        data_dict[problem + '_graph'].append(graph_data)
                    except:
                        pass

            return data_dict
        else:
            raise NotImplementedError
            # self.opts.data_fetching == 'sequential'
            for split in self.splits:
                if idx >= self.split_len:
                    idx -= self.split_len
                else:
                    cnf_filepath = self.all_files[split][idx]
                    label = self.all_labels[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)
                    data = torch.load(saved_path)
                    data.y = label
                    return [data]


 # def __getitem__(self, idx):
 #        data = {}
 #        for problem_type in self.problem_types:
 #            graph_data_file = self.graph_data_dict[problem_type][idx]
 #            graph_data, edges = load_graph(graph_data_file)
 #            if problem_type == 'GPP':
 #                # take initial vertex features by one-hot encoding
 #                init_graph_feature = torch.zeros((graph_data.number_of_nodes(), 3))
 #                for i in range(graph_data.number_of_nodes()):
 #                    if graph_data.nodes[str(i)]['bipartite'] == 0:
 #                        init_graph_feature[i][0] = 1
 #                    else:
 #                        init_graph_feature[i][1] = 1
 #                    init_graph_feature[i][2] = i / graph_data.number_of_nodes()
 #            else:
 #                init_graph_feature = torch.ones((graph_data.number_of_nodes(), 3))
 #                for i in range(graph_data.number_of_nodes()):
 #                    init_graph_feature[i][2] = i / graph_data.number_of_nodes()
 #
 #            g_data = Data(x=init_graph_feature,
 #                          edge_index=torch.tensor(edges).t().contiguous())
 #
 #            sat_data_file = self.sat_data_dict[problem_type][idx]
 #            sat_data = sat_to_LCG(sat_data_file)
 #            init_sat_feature = torch.zeros((sat_data.number_of_nodes(), 3))
 #            for ii in range(sat_data.number_of_nodes()):
 #                if sat_data.nodes[ii]['type'] == 'v':
 #                    init_sat_feature[ii][0] = 1
 #                else:
 #                    init_sat_feature[ii][1] = 1
 #                init_sat_feature[ii][2] = ii / sat_data.number_of_nodes()
 #            s_data = Data(x=init_sat_feature,
 #                          edge_index=torch.tensor(list(sat_data.edges())).t().contiguous())
 #
 #            graph_idx = sat_data_file.split('/')[-1].split('.')[0]
 #
 #            data[problem_type] = {
 #                'graph': g_data,
 #                'sat': s_data,
 #                'sat_solving': self.sat_solving_dict[problem_type][graph_idx],
 #            }
 #        return data
