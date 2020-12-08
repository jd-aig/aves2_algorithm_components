#!/bin/bash
# parse args
ARGS=`getopt -o : -l "help,output_path:,tfrecords_dir:,model_dir:" -- "$@"`
eval set -- "${ARGS}"

declare -A run_map=()

HELP_STR="Usage: This script use with args\n 
--output_dir\n
--tfrecords_dir\n
--model_dir"

while true;
do
    case "$1" in
        --output_path)
            run_map["output"]="$2"
            shift 2
            ;;
        --tfrecords_dir)
            run_map["tfrecords"]="$2"
            shift 2
            ;;
        --model_dir)
            run_map["model"]="$2"
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

export PYTHONPATH=$PYTHONPATH:/opt:/opt/slim:/opt/tensorflow-yolov3-master
python eval.py --logtostderr --checkpoint_dir=${run_map['model']} --eval_dir=${run_map['output']}/val --pipeline_config_path=${run_map['model']}/pipeline.config
mv ./result.json ${run_map['output']}/result.json
rm -rf ${run_map['output']}/val
