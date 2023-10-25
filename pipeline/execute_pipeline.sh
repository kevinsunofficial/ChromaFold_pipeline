#!/bin/bash -l

while getopts c:s:e: flag
do
    case "${flag}" in
        c) sample=${OPTARG};;
        s) start=${OPTARG};;
        e) end=${OPTARG};;
    esac
done

min=1
max=19

if [ "$start" -lt "$min" ] || [ "$end" -gt "$max" ]
then
    echo "Invalid chromosome range, please try again"
    exit 1
fi

echo "Current sample: $sample"
echo "Using chromosome $start to $end"

if [[ "$sample" == "mycGCB_am_gfp_myc_sample" ]]
then
    for (( i=$start; i<=$end; i++ ))
    do
        time python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
            --pred_dir /data/leslie/suny4/predictions/chromafold/ \
            --paired --ct mycGCB_am_gfp_myc_gcb_thelp_sample mycGCB_am_gfp_myc_gcb_nothelp_sample \
            --chrom $i --avg_stripe --kernel tad_diff \
            --min_dim 25 --max_dim 75 --num_dim 10 --close 10 \
            --filters gene_type=protein_coding \
            --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
            --num_plot 10 --out_dir /data/leslie/suny4/pipeline_result/mycGCB_am_gfp_myc_sample
    done
elif [[ "$sample" == "darko_ctcfhet_cb" ]]
then
    for (( i=$start; i<=$end; i++ ))
    do
        time python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
            --pred_dir /data/leslie/suny4/predictions/chromafold/ \
            --paired --ct darko_wt_cb darko_ctcfhet_cb \
            --chrom $i --avg_stripe --kernel tad_diff \
            --min_dim 25 --max_dim 75 --num_dim 10 --close 10 \
            --filters gene_type=protein_coding \
            --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
            --num_plot 10 --out_dir /data/leslie/suny4/pipeline_result/darko_ctcfhet_cb
    done
elif [[ "$sample" == "darko_arid1ahet_cb" ]]
then
    for (( i=$start; i<=$end; i++ ))
    do
        time python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
            --pred_dir /data/leslie/suny4/predictions/chromafold/ \
            --paired --ct darko_wt_cb darko_arid1ahet_cb \
            --chrom $i --avg_stripe --kernel tad_diff \
            --min_dim 25 --max_dim 75 --num_dim 10 --close 10 \
            --filters gene_type=protein_coding \
            --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
            --num_plot 10 --out_dir /data/leslie/suny4/pipeline_result/darko_arid1ahet_cb
    done
else
    echo "No valid commands match, please try again"
    exit 1
fi

echo "Done"