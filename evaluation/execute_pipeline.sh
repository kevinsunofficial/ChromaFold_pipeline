#!/bin/bash -l

for ((i=1; i<=3; i++))
do
    python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
                       --pred_dir /data/leslie/suny4/predictions/chromafold/ \
                       --paired --ct mycGCB_am_gfp_myc_gcb_thelp_sample mycGCB_am_gfp_myc_gcb_nothelp_sample \
                       --chrom $i --avg_stripe --topdom_cutoff 0 \
                       --kernel pearson --similar_window 10 --pattern smooth \
                       --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
                       --filters gene_type=protein_coding \
                       --out_dir ../testfile/ \
                       --fig_dir /data/leslie/suny4/figures/pipeline_test/
done