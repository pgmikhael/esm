#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10 
do	
	echo python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=3 tasks.fixedbb.accept_reject.energy_cfg.class_idx=stress_granule
	python guided_lm_design.py task=free_generation seed=$i cuda_device_idx=3 tasks.fixedbb.accept_reject.energy_cfg.class_idx=stress_granule
done
