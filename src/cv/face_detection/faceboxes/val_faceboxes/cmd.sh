ARGS=`getopt -o : -l "help,output_path:,val_tfrecords:,model_dir:" -- "$@"`
eval set -- "${ARGS}"

declare -A run_map=()

HELP_STR="Usage: This script use with args..."

while true;
do
    case "$1" in
        --output_path)
            run_map["output"]="$2"
            shift 2
            ;;
        --val_tfrecords)
            run_map["val"]="$2"
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

python3 val.py --val_tfrecord ${run_map['val']} --model_dir ${run_map['model']} --output_path ${run_map['output']}
