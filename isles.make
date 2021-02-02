CC = python3
SHELL = /usr/bin/zsh
PP = PYTHONPATH="$(PYTHONPATH):."

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/isles

.PHONY = all boundary plot train metrics hausdorff pack

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
reset:=$(shell tput sgr0)

# CFLAGS = -O
# DEBUG = --debug
EPC = 100
# EPC = 5

K = 2
BS = 8
G_RGX = (case_\d+_\d+)_\d+
P_RGX = (case_\d+)_\d+_\d+
NET = UNet
B_DATA = [('in_npy', tensor_transform, False), ('gt_npy', gt_transform, True)]

TRN = $(RD)/gdl $(RD)/gdl_surface_steal $(RD)/gdl_3d_surface_steal $(RD)/gdl_hausdorff_w

GRAPH = $(RD)/tra_loss.png $(RD)/val_loss.png \
		$(RD)/val_dice.png $(RD)/tra_dice.png \
		$(RD)/val_3d_hausdorff.png \
		$(RD)/val_3d_hd95.png
BOXPLOT =  $(RD)/val_dice_boxplot.png
PLT = $(GRAPH) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-isles.tar.gz

all: $(PACK)

plot: $(PLT)

train: $(TRN)

pack: report $(PACK)
$(PACK): $(PLT) $(TRN)
	$(info $(red)tar cf $@$(reset))
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available
$(LIGHTPACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	$(eval PLTS:=$(filter %.png, $^))
	$(eval FF:=$(filter-out %.png, $^))
	$(eval TGT:=$(addsuffix /best_epoch, $(FF)) $(addsuffix /*.npy, $(FF)) $(addsuffix /best_epoch.txt, $(FF)) $(addsuffix /metrics.csv, $(FF)))
	tar cf - $(PLTS) $(TGT) | pigz > $@
	chmod -w $@



# Extraction and slicing
data/ISLES/train/in_npy data/ISLES/val/in_npy: data/ISLES
data/ISLES: data/isles/TRAINING
	$(info $(yellow)$(CC) $(CFLAGS) preprocess/slice_isles.py$(reset))
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_isles.py --source_dir $< --dest_dir $@_tmp --n_augment=0 --retain=20
	mv $@_tmp $@
data/ISLES/test: data/isles/TESTING

data/isles/TESTING data/isles/TRAINING: data/isles
data/isles: data/ISLES.lineage data/ISLES2018_Training.zip data/ISLES2018_Testing.zip
	$(info $(yellow)unzip data/ISLES2018_Training.zip data/ISLES2018_Testing.zip$(reset))
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	rm -r $@_tmp/__MACOSX
	rm -r $@_tmp/*/*/*CT_4DPWI*  # For space efficiency1
	mv $@_tmp $@


# Training
$(RD)/gdl: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1)]"
$(RD)/gdl: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True)]"

$(RD)/gdl_surface_w: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1), \
	('SurfaceLoss', {'idc': [1]}, 0.1)]"
$(RD)/gdl_surface_w: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl_surface_w: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]"

$(RD)/gdl_hausdorff_w: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1), \
	('HausdorffLoss', {'idc': [1]}, 0.1)]"
$(RD)/gdl_hausdorff_w: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl_hausdorff_w: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', gt_transform, True)]"


$(RD)/hausdorff: OPT = --losses="[('HausdorffLoss', {'idc': [1]}, 0.1)]"
$(RD)/hausdorff: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/hausdorff: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True)]"


$(RD)/gdl_surface_add: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1), \
	('SurfaceLoss', {'idc': [1]}, 0.01)]"
$(RD)/gdl_surface_add: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl_surface_add: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"

$(RD)/gdl_surface_steal: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1), \
	('SurfaceLoss', {'idc': [1]}, 0.01)]"
$(RD)/gdl_surface_steal: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl_surface_steal: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"


$(RD)/gdl_3d_surface_steal: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, 1), \
	('SurfaceLoss', {'idc': [1]}, 0.01)]"
$(RD)/gdl_3d_surface_steal: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/gdl_3d_surface_steal: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('3d_distmap', raw_npy_transform, False)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"

$(RD)/surface: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 0.1)]"
$(RD)/surface: data/ISLES/train/in_npy data/ISLES/val/in_npy
$(RD)/surface: DATA = --folders="$(B_DATA)+[('gt_npy', dist_map_transform, False)]"


$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --in_memory --l_rate=0.001 --schedule \
		--use_spacing \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --modalities=5 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Metrics
## Those need to be computed once the training is over, as we have to reconstruct the whole 3D volume
metrics: $(TRN) \
	$(addsuffix /val_3d_dsc.npy, $(TRN)) \
	$(addsuffix /val_3d_hausdorff.npy, $(TRN)) \
	$(addsuffix /val_3d_hd95.npy, $(TRN))

$(RD)/%/val_3d_dsc.npy $(RD)/%/val_3d_hausdorff.npy $(RD)/%/val_3d_hd95.npy: data/ISLES/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_dsc 3d_hausdorff 3d_hd95 \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


hausdorff: $(TRN) \
	$(addsuffix /val_hausdorff.npy, $(TRN))

$(RD)/%/val_hausdorff.npy: data/ISLES/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics hausdorff \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


boundary: $(TRN) \
	$(addsuffix /val_boundary.npy, $(TRN))

$(RD)/%/val_boundary.npy: data/ISLES/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics boundary \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


# Plotting
$(RD)/tra_loss.png $(RD)/val_loss.png: COLS = 0 1
$(RD)/tra_loss.png $(RD)/val_loss.png: OPT = --ylim -1 1 --dynamic_third_axis --no_mean
$(RD)/tra_loss.png $(RD)/val_loss.png: plot.py $(TRN)

$(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1
$(RD)/val_dice.png $(RD)/tra_dice.png: plot.py $(TRN)

$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: COLS = 1
$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: OPT = --ylim 0 40 --min
$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: plot.py $(TRN)

$(RD)/val_dice_boxplot.png: COLS = 1
$(RD)/val_dice_boxplot.png: moustache.py $(TRN)

$(RD)/%.png: | metrics
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _boxplot,,$(@F)))  # Needed to use same recipe for both histogram and plots
	$(eval metric:=$(subst _hist,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)


# Viewing
view: $(TRN)
	viewer/viewer.py -n 3 --img_source data/ISLES/val/tmax data/ISLES/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(shell basename -a -s '/' $^) $(DEBUG)
	# viewer -n 3 --img_source data/ISLES/val/flair data/ISLES/val/gt $(addsuffix /iter000/val, $^) --crop 10 \

report: $(TRN) | metrics
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --axises 1 --precision 3 \
		--metrics val_dice val_hausdorff val_3d_dsc val_3d_hausdorff val_3d_hd95
