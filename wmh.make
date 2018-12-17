CC = python3.6
SHELL = /usr/bin/zsh

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5


G_RGX = (\d+_\d+)_\d+
NET = UNet
B_DATA = [('in_npy', torch.tensor, False), ('gt_npy', gt_transform, True)]

# TRN = results/wmh/gdl results/wmh/gdl_surface_w \
# 	results/wmh/gdl_surface_add results/wmh/gdl_surface_steal
TRN = results/wmh/gdl results/wmh/gdl_surface_steal
HAUS = $(addsuffix /hausdorff.npy, $(TRN))

GRAPH = results/wmh/val_dice.png results/wmh/tra_dice.png \
			results/wmh/tra_loss.png \
		results/wmh/val_batch_dice.png \
		results/wmh/val_haussdorf.png results/wmh/hausdorff.png

HIST =  results/wmh/val_dice_hist.png results/wmh/tra_loss_hist.png \
		results/wmh/val_batch_dice_hist.png
PLT = $(GRAPH) $(HIST)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-wmh.tar.gz

all: pack

pack: report $(PACK)
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/WMH/train/in_npy data/WMH/val/in_npy: data/WMH
data/WMH: data/wmh
	rm -rf $@_tmp
	$(CC) $(CFLAGS) slice_wmh.py --source_dir $< --dest_dir $@_tmp --n_augment=0 --retain=10
	mv $@_tmp $@

data/WMH-pos/train/in_npy data/WMH-pos/val/in_npy: data/WMH-pos
data/WMH-pos: data/wmh
	rm -rf $@_tmp
	$(CC) $(CFLAGS) slice_wmh.py --source_dir $< --dest_dir $@_tmp --n_augment=4 --retain=10 --discard_negatives
	mv $@_tmp $@

data/wmh: data/wmh.lineage data/Amsterdam_GE3T.zip data/Singapore.zip data/Utrecht.zip
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp/*/* $@_tmp && rmdir $@_tmp/GE3T $@_tmp/Singapore $@_tmp/Utrecht
	rm -r $@_tmp/*/orig  # Do not care about that part
	rm -r $@_tmp/*/pre/3DT1.nii.gz  # Cannot align to the rest
	mv $@_tmp $@

# Training
results/wmh/gdl: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1)]"
results/wmh/gdl: data/WMH/train/in_npy data/WMH/val/in_npy
results/wmh/gdl: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True)]"

results/wmh/gdl_surface_w: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.0)]"
results/wmh/gdl_surface_w: data/WMH/train/in_npy data/WMH/val/in_npy
results/wmh/gdl_surface_w: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]"

results/wmh/gdl_surface_add: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.01)]"
results/wmh/gdl_surface_add: data/WMH/train/in_npy data/WMH/val/in_npy
results/wmh/gdl_surface_add: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"

results/wmh/gdl_surface_steal: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.01)]"
results/wmh/gdl_surface_steal: data/WMH/train/in_npy data/WMH/val/in_npy
results/wmh/gdl_surface_steal: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"

$(TRN):
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=8 --in_memory --l_rate=0.001 --schedule \
		--compute_haussdorf \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --modalities=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
results/wmh/val_batch_dice.png results/wmh/val_dice.png results/wmh/tra_dice.png: COLS = 1

results/wmh/val_haussdorf.png results/wmh/tra_haussdorf.png: COLS = 1
results/wmh/val_haussdorf.png results/wmh/tra_haussdorf.png: OPT = --ylim 0 7 --min

results/wmh/hausdorff.png: COLS = 1
results/wmh/hausdorff.png: OPT = --ylim 0 7 --min

results/wmh/tra_loss.png: COLS = 0
results/wmh/tra_loss.png: OPT = --min --ylim -5 5
$(GRAPH): plot.py $(TRN)

results/wmh/val_batch_dice_hist.png results/wmh/val_dice_hist.png: COLS = 1
results/wmh/tra_loss_hist.png: COLS = 0
$(HIST): hist.py $(TRN)

$(GRAPH) $(HIST):
	$(eval metric:=$(subst _hist,,$(@F)))  # Needed to use same recipe for both histogram and plots
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/WMH/val/flair data/WMH/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) $(DEBUG)
	# viewer -n 3 --img_source data/WMH/val/flair data/WMH/val/gt $(addsuffix /iter000/val, $^) --crop 10 \

report: $(TRN) $(HAUS)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_dice val_haussdorf hausdorff --axises 1

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=2 --grp_regex="$(G_RGX)" --gt_folder data/WMH/val/gt \
		--pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)

$(HAUS): $(TRN)
	$(CC) hd_over_time.py --pred_root $(@D) --gt_folder data/WMH/val/gt --save_folder $(@D) \
		--num_classes 2 --epochs $(EPC) $(DEBUG)