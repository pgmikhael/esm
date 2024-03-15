#!/bin/sh
for i in 7 8
do	
	echo python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=1 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
	python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=1 tasks.fixedbb.accept_reject.energy_cfg.class_idx=nuclear_speckle
done
