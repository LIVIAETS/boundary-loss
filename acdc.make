CC = python3.9
PP = PYTHONPATH="$(PYTHONPATH):."
SHELL = zsh


.PHONY: all geodist train plot view view_labels npy pack report weak

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
reset:=$(shell tput sgr0)

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/acdc

# CFLAGS = -O
# DEBUG = --debug
EPC = 100
BS = 8  # BS stands for Batch Size
K = 4  # K for class

G_RGX = (patient\d+_\d+_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
# NET = Dummy


TRN = $(RD)/ce \
	$(RD)/diceloss \
	$(RD)/boundary


GRAPH = $(RD)/val_dice.png $(RD)/tra_dice.png \
		$(RD)/tra_loss.png \
		$(RD)/val_3d_dsc.png
BOXPLOT = $(RD)/val_3d_dsc_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-acdc.tar.gz

all: pack

train: $(TRN)
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	$(info $(red)tar cf $@$(reset))
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available


# Data generation
data/ACDC-2D: OPT = --seed=0 --retain 25
data/ACDC-2D: data/acdc
	$(info $(yellow)$(CC) $(CFLAGS) preprocess/slice_acdc.py$(reset))
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$@_tmp $(OPT)
	mv $@_tmp $@

data/acdc: data/acdc.lineage data/acdc.zip
	$(info $(yellow)unzip data/acdc.zip$(reset))
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	rm $@_tmp/training/*/*_4d.nii.gz  # space optimization
	mv $@_tmp $@


data/ACDC-2D/train/img data/ACDC-2D/val/img: | data/ACDC-2D
data/ACDC-2D/train/gt data/ACDC-2D/val/gt: | data/ACDC-2D


# Trainings
$(RD)/ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/ce: data/ACDC-2D/train/gt data/ACDC-2D/val/gt
$(RD)/ce: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/diceloss: OPT = --losses="[('DiceLoss', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/diceloss: data/ACDC-2D/train/gt data/ACDC-2D/val/gt
$(RD)/diceloss: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/boundary: OPT = --losses="[('BoundaryLoss', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/boundary: data/ACDC-2D/train/gt data/ACDC-2D/val/gt
$(RD)/boundary: DATA = --folders="$(B_DATA)+[('gt', dist_map_transform, False)]"

# Template
$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 1 2 3 \
		--compute_3d_dice \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
$(RD)/val_3d_dsc.png $(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1 2 3
$(RD)/tra_loss.png: COLS = 0
$(RD)/tra_loss.png: OPT = --dynamic_third_axis
$(RD)/val_dice.png $(RD)/tra_loss.png $(RD)/val_3d_dsc.png: plot.py $(TRN)
$(RD)/tra_dice.png: plot.py $(TRN)

$(RD)/val_3d_dsc_boxplot.png: COLS = 1 2 3
$(RD)/val_3d_dsc_boxplot.png: moustache.py $(TRN)

$(RD)/%.png:
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT)


# Viewing
view: $(TRN)
	viewer/viewer.py -n 3 --img_source data/ACDC-2D/val/img data/ACDC-2D/val/gt \
		$(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) --no_contour -C $(K)

report: $(TRN)
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_3d_dsc val_dice --axises 3 $(DEBUG)