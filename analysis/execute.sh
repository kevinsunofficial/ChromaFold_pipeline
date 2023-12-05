#!/bin/bash -l

while getopts x:y:s:e: flag
do
    case "${flag}" in
        x) samplex=${OPTARG};;
        y) sampley=${OPTARG};;
        s) start=${OPTARG};;
        e) end=${OPTARG};;
    esac
done

min=1
max=20

if [[ $start -lt $min ]] || [[ $end -gt $max ]] || [[ $start -gt $end ]]
then
    echo "Invalid chromosome range, forcing ranges with min / max"
    if [[ $start -lt $min ]]
    then
        start=$min
    fi
    if [[ $end -gt $max ]]
    then
        end=$max
    fi
fi

if [[ $start == $end ]]
then
    echo "Using chromosome $start"
else
    echo "Using chromosome $start to $end"
fi

for (( i=$start; i<=$end; i++ ))
do
    chrom=$i
    if [[ $i == $max ]]
    then
        chrom="X"
    fi

    if [[ "$sampley" == "none" ]]
    then
        time python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
            --pred_dir /data/leslie/suny4/predictions/chromafold/ \
            --ct $samplex --chrom $chrom --min_dim 10 --max_dim 90 --num_dim 25 --close 10 \
            --filters gene_type=protein_coding \
            --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
            --gtf_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.gtf \
            --num_plot 10 --out_dir "/data/leslie/suny4/pipeline_result/${samplex}"
    else
        time python pipeline.py --input_dir /data/leslie/suny4/processed_input/ \
            --pred_dir /data/leslie/suny4/predictions/chromafold/ \
            --ct $samplex $sampley --chrom $chrom --min_dim 10 --max_dim 90 --num_dim 25 --close 10 \
            --filters gene_type=protein_coding \
            --db_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.db \
            --gtf_file /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation.gtf \
            --num_plot 10 --out_dir "/data/leslie/suny4/pipeline_result/${samplex}_${sampley}"
    fi
done

echo "Done"
