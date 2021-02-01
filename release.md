# 1.0 release

Differences:
* 3d computation of the distance maps, done at the pre-processing time
* Other losses implementation, for comparison
* Example of a multi-class setting (ACDC dataset)
* Improved readme and instructions
* Updated for latest python and pytorch releases
* Removed all remnants from our constrained-cnn works (shared codebase)
* Add colors to the makefile, for improved readability
* More flexible makefiles, that allow separate results folders (through the RD environment variable)
* submodule for the viewer, pointing to https://github.com/HKervadec/segmentation_viewer
* The training recipe now store commit-hash and diff, to track exactly the code version