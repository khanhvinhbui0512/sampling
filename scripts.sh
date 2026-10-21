CUDA_VISIBLE_DEVICES=7 python src/qm9/train_base_model.py --config config-50000/qm9/qm9_base_model_training.yaml
CUDA_VISIBLE_DEVICES=7 python src/qm9/base_model_rollout.py --config config-50000/qm9/qm9_rollout.yaml
CUDA_VISIBLE_DEVICES=7 python src/qm9/verifier_training.py --config config-50000/qm9/qm9_verifier_training.yaml
CUDA_VISIBLE_DEVICES=7 python src/qm9/inference.py --config config-50000/qm9/qm9_inference.yaml --algorithm Base
CUDA_VISIBLE_DEVICES=7 python src/qm9/inference.py --config config-50000/qm9/qm9_inference.yaml --algorithm BoN
CUDA_VISIBLE_DEVICES=7 python src/qm9/inference.py --config config-50000/qm9/qm9_inference.yaml --algorithm VGR
CUDA_VISIBLE_DEVICES=6 python src/qm9/inference.py --config config-50000/qm9/qm9_inference.yaml --algorithm VGB
CUDA_VISIBLE_DEVICES=7 python src/qm9/inference.py --config config-50000/qm9/qm9_inference.yaml --algorithm VGB-Momentum


CUDA_VISIBLE_DEVICES=6 python src/qm9/verifier_training.py --config config/qm9/qm9_verifier_training.yaml


CUDA_VISIBLE_DEVICES=5 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm Base
CUDA_VISIBLE_DEVICES=5 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm BoN
CUDA_VISIBLE_DEVICES=5 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm VGR
CUDA_VISIBLE_DEVICES=5 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm VGB
CUDA_VISIBLE_DEVICES=5 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm VGB-Momentum

CUDA_VISIBLE_DEVICES=6 python src/qm9/inference.py --config config/qm9/qm9_inference.yaml --algorithm CARS
