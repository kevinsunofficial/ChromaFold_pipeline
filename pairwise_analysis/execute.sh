#!/bin/bash -l

while getopts x:y:c:s:e:m: flag
do
    case "${flag}" in
        x) samplex=${OPTARG};;
        y) sampley=${OPTARG};;
        c) control=${OPTARG};;
        s) start=${OPTARG};;
        e) end=${OPTARG};;
        m) mode=${OPTARG};;
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
        echo "UNDEFINED"
        exit 1
    else
        if [[ "$control" == "none" ]]
        then
            time python main.py --root_dir /data/leslie/suny4/processed_input/ \
                --pred_dir /data/leslie/suny4/predictions/chromafold/ \
                --out_dir "/data/leslie/suny4/pipeline_result/${samplex}_${sampley}" \
                --annotation /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation \
                --ct $samplex $sampley --chrom $chrom --mode $mode\
                --filters gene_type=protein_coding
        else
            time python main.py --root_dir /data/leslie/suny4/processed_input/ \
                --pred_dir /data/leslie/suny4/predictions/chromafold/ \
                --out_dir "/data/leslie/suny4/pipeline_result/${samplex}_${sampley}_${control}" \
                --annotation /data/leslie/suny4/data/chrom_size/gencode.vM10.basic.annotation \
                --ct $samplex $sampley $control --chrom $chrom --mode $mode\
                --filters gene_type=protein_coding
        fi
        if [ $? != 0 ];
        then
            echo "Error"
            exit 1
        fi
    fi
done

echo "Done"
