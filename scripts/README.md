To simplify working with the project, this folder
contains examples of preprocess/train/serve scripts
that can be run almost without further arguments supplied,
for two different datasets.

See the README in chi-data for further information

### Data directory
Importantly, the Makefile in the project root allows you to specify
a `DATA_DIR` that will be mounted into the Docker containers. However,
specifying that each time is annoying -- and so the scripts in the
subfolders here automatically source `data-dir.sh` which exports
a valid `DATA_DIR`. You must therefore _modify_ that file to export
the full path to the location you want to mount as /data in the container.

By default, `/data` is mounted to `/data` in the container.
