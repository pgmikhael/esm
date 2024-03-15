#!/bin/bash
END=20
for ((i=10;i<=END;i++)); do
	echo python guided_lm_design.py task=free_generation init_sequence="" seed=$i  free_generation_length=10 cuda_device_idx=1 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
	python guided_lm_design.py task=free_generation init_sequence="" seed=$i  free_generation_length=10 cuda_device_idx=1 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
done
