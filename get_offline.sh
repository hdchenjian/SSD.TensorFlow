../tools/pb_to_cambricon/pb_to_cambricon.host --graph=model/ssd300_vgg16_mlu.pb \
                  --param_file=ssd_model_offline_param_file.txt \
                  --core_version="MLU270" \
                  --core_num=1 \
                  --save_pb=true
