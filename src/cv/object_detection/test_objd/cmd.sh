ARGS=`getopt -o : -l "help,data_dir:,output_path:,model_dir:,min_score_thresh:" -- "$@"`
eval set -- "${ARGS}"

declare -A run_map=()

HELP_STR="Usage: This script use with args..."

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
        --model_dir)
            run_map["model"]="$2"
            shift 2
            ;;
        --min_score_thresh)
            run_map["min_score_thresh"]="$2"
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
python test.py --data_dir ${run_map['input']} --output_path ${run_map['output']} --model_dir ${run_map['model']} --min_score_thresh ${run_map["min_score_thresh"]}
