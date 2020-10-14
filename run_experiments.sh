#!/bin/bash
for ACQ in 2 3 4 5 6 7 8 16 32
do
  for WEQ in 2 3 4 5 6 7 8 16 32
  do
  if (($ACQ <= 8)) & (($WEQ <= 8)); then
    INQ=8
  else
    INQ=32
  fi
  python nn_benchmark/main.py \
      --network QuantTFC \
      --dataset MNIST \
      --batch_size 100 \
      --lr 0.01 \
      --optim ADAM \
      --loss CrossEntropy \
      --scheduler STEP \
      --milestones '34,37' \
      --momentum 0.9 \
      --epochs 40 \
      --acq $ACQ \
      --weq $WEQ \
      --inq $INQ
  done
done
