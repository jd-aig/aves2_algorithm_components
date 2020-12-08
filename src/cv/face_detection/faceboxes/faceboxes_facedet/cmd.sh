ARGS=`getopt -o : -l "help,output_path:,train_tfrecords:,val_tfrecords:,pretrainmodel_dir:,learning_rate:,epochs:,batch_size:" -- "$@"`
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
        --train_tfrecords)
            run_map["train"]="$2"
            shift 2
            ;;
        --val_tfrecords)
            run_map["val"]="$2"
            shift 2
            ;;
        --pretrainmodel_dir)
            run_map["pretrainmodel"]="$2"
            shift 2
            ;;
        --learning_rate)
            run_map["learning_rate"]="$2"
            shift 2
            ;;
        --epochs)
            run_map["epochs"]="$2"
            shift 2
            ;;
        --batch_size)
            run_map["batch_size"]="$2"
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

python3 train.py --train_tfrecord ${run_map['train']} --val_tfrecord ${run_map['val']} --output_path ${run_map['output']} --batch_size ${run_map["batch_size"]} --epochs ${run_map["epochs"]}
mkdir ${run_map['output']}/tmp
python3 save.py --output_path ${run_map['output']}

mv ${run_map['output']}/*.index ${run_map['output']}/model.ckpt.index
mv ${run_map['output']}/*.meta ${run_map['output']}/model.ckpt.meta
mv ${run_map['output']}/*.data-00000-of-00001 ${run_map['output']}/model.ckpt.data-00000-of-00001
mv ${run_map['output']}/tmp/* ${run_map['output']}/saved_model

rm -rf ${run_map['output']}/event*
rm -rf ${run_map['output']}/eval/
rm -r ${run_map['output']}/tmp

python3 create_pb.py -s ${run_map['output']}/saved_model -o ${run_map['output']}/frozen_inference_graph.pb
