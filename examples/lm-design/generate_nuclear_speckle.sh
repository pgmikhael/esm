#!/bin/sh
for i in 27 28 29 30 31 32 33 34
do	
	echo python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=3 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
	python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=3 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
done
