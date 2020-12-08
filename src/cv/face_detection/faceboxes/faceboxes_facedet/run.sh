#python3 train.py --train_tfrecord /export/luozhuang/softwares/components/FACE_DETECTION/create_tfrecord_facedetection/output_train/ --val_tfrecord /export/luozhuang/data_tf/faceboxes/val_tfrecords/ --output_path ./output/ --batch_size 16 --epochs 20

#create saved_model
#python3 save.py
#create inference pb 
#python3 create_pb.py -s export/run00/1559023097/ -o ./inference.pb

/bin/bash cmd.sh --train_tfrecords /export/luozhuang/softwares/components/FACE_DETECTION/create_tfrecord_facedetection/output_train/ --val_tfrecords /export/luozhuang/data_tf/faceboxes/val_tfrecords/ --output_path ./output/ --batch_size 16 --epochs 20
