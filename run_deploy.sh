#!/bin/bash
for ACQ in 2 4 8 16 32
do
  WEQ=$ACQ
  if (($ACQ <= 8)) & (($WEQ <= 8)); then
    INQ=8
  else
    INQ=32
  fi
  python finn_experiments/updated_exp_qnt_tfc.py --acq $ACQ --weq $WEQ --inq $INQ
done
