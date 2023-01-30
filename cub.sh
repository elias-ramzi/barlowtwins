# CUDA_VISIBLE_DEVICES=1 python main.py \
# '/local/DEEPLEARNING/image_retrieval/CUB_200_2011' \
# --dataset cub200 \
# --pretrained \
# --checkpoint-dir 'experiments/barlowtwins/sop_resnet50_bs64_lr0.4' \
# --batch-size 64 \
# --learning-rate-weights 0.4 \
# --epochs 200 \
# --transform base


CUDA_VISIBLE_DEVICES=0 python main.py \
'/local/DEEPLEARNING/image_retrieval/CUB_200_2011' \
--dataset cub200 \
--pretrained \
--checkpoint-dir 'experiments/barlowtwins/sop_resnet50_bs64_lr1.0' \
--batch-size 64 \
--learning-rate-weights 1.0 \
--epochs 200 \
--transform base


CUDA_VISIBLE_DEVICES=0 python main.py \
'/local/DEEPLEARNING/image_retrieval/CUB_200_2011' \
--dataset cub200 \
--pretrained \
--checkpoint-dir 'experiments/barlowtwins/sop_resnet50_bs64_lr0.1' \
--batch-size 64 \
--learning-rate-weights 0.1 \
--epochs 200 \
--transform base


CUDA_VISIBLE_DEVICES=0 python main.py \
'/local/DEEPLEARNING/image_retrieval/CUB_200_2011' \
--dataset cub200 \
--pretrained \
--checkpoint-dir 'experiments/barlowtwins/sop_resnet50_bs64_lr0.05' \
--batch-size 64 \
--learning-rate-weights 0.05 \
--epochs 200 \
--transform base


CUDA_VISIBLE_DEVICES=0 python main.py \
'/local/DEEPLEARNING/image_retrieval/CUB_200_2011' \
--dataset cub200 \
--pretrained \
--checkpoint-dir 'experiments/barlowtwins/sop_resnet50_easy' \
--batch-size 64 \
--learning-rate-weights 0.01 \
--epochs 100 \
--transform easy
