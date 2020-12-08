#!/bin/bash
# parse args
ARGS=`getopt -o : -l "help,data_dir:,output_path:" -- "$@"`
eval set -- "${ARGS}"

declare -A run_map=()

HELP_STR="Usage: This script use with args\n 
--input_data\n
--output_dir"

while true;
do
    case "$1" in
        --data_dir)
            run_map["input"]="$2"
            shift 2
            ;;
        --output_path)
            run_map["output"]="$2"
            shift 2
            ;;
        --help)
            echo -e ${HELP_STR}
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error"
            exit 1
            ;;
    esac
done

python create_tfrecords.py --image_dir=${run_map['input']}/images/ --annotations_dir=${run_map['input']}/annotations/ --output=${run_map['output']} --num_shards=20
