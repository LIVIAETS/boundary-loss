CC = python3.6
SHELL = /usr/bin/zsh

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5

NET = UNet
B_DATA = [('in_npy', torch.tensor, False), ('gt_npy', gt_transform, True)]

# TRN = results/isles/gdl results/isles/gdl_surface_w \
	# results/isles/gdl_surface_add results/isles/gdl_surface_steal

TRN = results/isles/gdl results/isles/gdl_surface_steal
HAUS = $(addsuffix /hausdorff.npy, $(TRN))

GRAPH = results/isles/val_dice.png results/isles/tra_dice.png \
			results/isles/tra_loss.png \
		results/isles/val_haussdorf.png results/isles/hausdorff.png

HIST =  results/isles/val_dice_hist.png results/isles/tra_loss_hist.png \
		results/isles/val_batch_dice_hist.png
PLT = $(GRAPH) $(HIST)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-isles.tar.gz

all: pack
# all: $(PLT)

pack: report $(PACK)
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available


# Extraction and slicing
data/ISLES/train/in_npy data/ISLES/val/in_npy: data/ISLES
data/ISLES: data/isles/TRAINING
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) slice_isles.py --source_dir $< --dest_dir $@_tmp --n_augment=0 --retain=20
	mv $@_tmp $@
data/ISLES/test: data/isles/TESTING

data/isles/TESTING data/isles/TRAINING: data/isles
data/isles: data/ISLES.lineage data/ISLES2018_Training.zip data/ISLES2018_Testing.zip
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	rm -r $@_tmp/__MACOSX
	rm -r $@_tmp/*/*/*CT_4DPWI*  # For space efficiency1
	mv $@_tmp $@

# Training
results/isles/gdl: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1)]"
results/isles/gdl: data/ISLES/train/in_npy data/ISLES/val/in_npy
results/isles/gdl: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True)]"

results/isles/gdl_surface_w: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.1)]"
results/isles/gdl_surface_w: data/ISLES/train/in_npy data/ISLES/val/in_npy
results/isles/gdl_surface_w: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]"

results/isles/gdl_surface_add: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.01)]" \
	--scheduler=AddWeightLoss --scheduler_params="{'to_add': [0, 0.01]}"
results/isles/gdl_surface_steal: OPT = --losses="[('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), \
	('SurfaceLoss', {'idc': [1]}, None, None, None, 0.01)]" \
	--scheduler=StealWeight --scheduler_params="{'to_steal': 0.01}"
results/isles/gdl_surface_add results/isles/gdl_surface_steal: data/ISLES/train/in_npy data/ISLES/val/in_npy
results/isles/gdl_surface_add results/isles/gdl_surface_steal: DATA = --folders="$(B_DATA)+[('gt_npy', gt_transform, True), \
	('gt_npy', dist_map_transform, False)]"


$(TRN):
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=4 --in_memory --l_rate=1e-3 --schedule \
		--compute_haussdorf \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --modalities=5 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
results/isles/val_batch_dice.png results/isles/val_dice.png results/isles/tra_dice.png: COLS = 1

results/isles/val_haussdorf.png results/isles/tra_haussdorf.png: COLS = 1
results/isles/val_haussdorf.png results/isles/tra_haussdorf.png: OPT = --ylim 0 7 --min

# results/isles/hausdorff.png:
results/isles/hausdorff.png: COLS = 1
results/isles/hausdorff.png: OPT = --ylim 0 7 --min

results/isles/tra_loss.png: COLS = 0
results/isles/tra_loss.png: OPT = --min --ylim -5 5
$(GRAPH): plot.py $(TRN)

results/isles/val_batch_dice_hist.png results/isles/val_dice_hist.png: COLS = 1
results/isles/tra_loss_hist.png: COLS = 0
$(HIST): hist.py $(TRN)

$(GRAPH) $(HIST):
	$(eval metric:=$(subst _hist,,$(@F)))  # Needed to use same recipe for both histogram and plots
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/ISLES/val/mtt data/ISLES/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^)

report: $(TRN) $(HAUS)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_dice val_haussdorf hausdorff --axises 1

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=2 --grp_regex="(case_\d+_\d+)_\d+" --gt_folder data/ISLES/val/gt \
		--pred_folders $(addsuffix /iter199/val, $^) $(DEBUG)
		# --pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)

$(HAUS): $(TRN)
	$(CC) hd_over_time.py --pred_root $(@D) --gt_folder data/ISLES/val/gt --save_folder $(@D) \
		--num_classes 2 --epochs $(EPC) $(DEBUG)