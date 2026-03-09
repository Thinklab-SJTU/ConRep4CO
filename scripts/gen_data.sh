# datasets
# k-clique
python g4satbench/generators/k-clique.py datasets/k-clique/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 4
python g4satbench/generators/k-clique.py datasets/k-clique-medium/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python g4satbench/generators/k-clique.py datasets/k-clique-hard/ --train_instances 0 --valid_instances 0 --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-domset
python g4satbench/generators/k-domset.py datasets/k-domset/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 2 --max_k 3
python g4satbench/generators/k-domset.py datasets/k-domset-medium/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python g4satbench/generators/k-domset.py datasets/k-domset-hard/ --train_instances 0 --valid_instances 0 --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-vercov
python g4satbench/generators/k-vercov.py datasets/k-vercov/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 5
python g4satbench/generators/k-vercov.py datasets/k-vercov-medium/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 10 --max_v 20  --min_k 6 --max_k 8
python g4satbench/generators/k-vercov.py datasets/k-vercov-hard/ --train_instances 0 --valid_instances 0 --test_instances 10000 --min_v 15 --max_v 25  --min_k 9 --max_k 10
