#!/bin/bash

rm -rf valis-results-*
rm -rf valis-slides-*

python main.py \
    --cytomine_host FILL_ME
    --cytomine_public_key FILL_ME \
    --cytomine_private_key FILL_ME \
    --cytomine_id_project 551726806 \
    --cytomine_id_software 551728990 \
    --reference_image 551727960 \
    --all_images "551728445,551728438,551728431,551727960,551727800,551727635,551727479,551727304,551727156,551726992" \
    --image_ordering "created" \
    --align_toward_reference "true" \
    --image_crop "all" \
    --registration_type "non-rigid" \
    --compose_non_rigid "false" \
    --annotation_to_map "551735031,551734856" \
    --images_to_warp "551728445,551728438,551728431"