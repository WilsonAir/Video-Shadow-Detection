import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json

data_path_list ={
# "ori+boundary+map" : "work_dirs/video_mask2former_edge_beitv2_adapter_large_896_80k_visha_ms/only_boundary_map_loss/20231128_115915.log.json",
# "ori+boundary+map2" : "work_dirs/video_mask2former_edge_beitv2_adapter_large_896_80k_visha_ms/boundary_map_loss_retrain/20231129_222029.log.json",
# "ori+boundary+area+map" : "work_dirs/video_mask2former_edge_contrast_beitv2_adapter_large_896_80k_visha_ms/only_boundaer_area_loss/20231130_200525.log.json",
# "ori+boundary+area+contrast" : "work_dirs/video_mask2former_edge_contrast_beitv2_adapter_large_896_80k_visha_ms/boundary_contrast_loss/20231130_235954.log.json",
# "ori+area+contrast" : "work_dirs/video_mask2former_edge_contrast_beitv2_adapter_large_896_80k_visha_ms/20231201_180337.log.json",
# "perpixel-spatial+structure" : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_temporal_vit/20231123_104343.log.json",
# "perpixel+structure_mse"    : "work_dirs/video_perpixel_structure_plus_beitv2_adapter_large_896_80k_visha_ms/perpixel_structure_mse_loss_pretrained_cocostuff_intergerate_and_versa/20231202_160832.log.json",
# "perpixel+structure"        : "work_dirs/video_perpixel_structure_plus_beitv2_adapter_large_896_80k_visha_ms/perpixel_structure_mse_loss_pretrained_cocostuff_only_intergerate/20231204_131148.log.json",
# "perpixel_from_cocostuff"   : "work_dirs/video_perpixel_structure_plus_beitv2_adapter_large_896_80k_visha_ms/perpixel_pretrained_from_cocostuff/20231202_122333.log.json",
# "perpixel_segformer"   : "work_dirs/video_perpixel_segformer_plus_beitv2_adapter_large_896_80k_visha_ms/segformer_ori/20231205_230122.log.json",
# "perpixel_segformer+abl"   : "work_dirs/video_perpixel_segformer_plus_beitv2_adapter_large_896_80k_visha_ms/segformer_abl/20231208_000539.log.json",
# "perpixel_segformer+structure"   : "work_dirs/video_perpixel_segformer_plus_beitv2_adapter_large_896_80k_visha_ms/segformer_structure/20231206_114347.log.json",
# "perpixel_label_decouple_bcel"   : "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/body_detail_bcel/20231205_171008.log.json",
# "perpixel_label_decouple_mse"   : "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/body_detail_mse/20231206_211344.log.json",
# "perpixel_label_decouple_bcel_inter_4_b8"   : "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/20231221_173651.log.json",
# "perpixel_label_decouple_mul_mse_inter_4_b2"   : "work_dirs/video_perpixel_decouple_mul_plus_beitv2_adapter_large_896_80k_visha_ms/20231221_220442.log.json",
# "perpixel_label_decouple_mul_mse_inter_5_b2"   : "work_dirs/video_perpixel_decouple_mul_plus_beitv2_adapter_large_896_80k_visha_ms/20231221_220442.log.json",
# "perpixel"                  : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_temporal_vit_saptial_vit/20231121_124052.log.json",
# "ori+boundary+branch" : "work_dirs/video_mask2former_edge_branch_beitv2_adapter_large_896_80k_visha_ms/20231128_233959.log.json",
# "ori" : "work_dirs/video_mask2former_beitv2_adapter_large_896_80k_visha_ms/frame_dis_5_spatial_vit_temporal_vit_m2f_resume_form_cocostuff/20231125_162522.log.json",
# "ori_perpixel_pretrained" : "work_dirs/video_mask2former_beitv2_adapter_large_896_80k_visha_ms/frame_dis_5_spatial_vit_temporal_vit_m2f_resume_from_perpixel/20231124_171652.log.json",
# "ori_without_temporal_vit" : "work_dirs/video_mask2former_beitv2_adapter_large_896_80k_visha_ms/frame_dis_5_spatial_temproal_seprate/20231026_165246_1.log.json",
# "ori-m2f" : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_temporal_vit_saptial_vit/20231121_124052.log.json",
# "ori-m2f-spatial" : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_temporal_vit/20231123_104343.log.json",
# "ori-m2f-temporal" : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_spatial/20231110_112842.log.json",
# "ori-m2f-temporal-spatial" : "work_dirs/video_perpixel_plus_beitv2_adapter_large_896_80k_visha_ms/ablation_no_intraction/20231107_143657.log.json"
    'interval_1': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_1/20231227_091700.log.json",
    # 'interval_2': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_2/20231226_225814.log.json",
    'interval_3': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_3/20231227_110840.log.json",
    # 'interval_4': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_4/20231228_093425.log.json",
    'interval_5': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_5/20231228_091850.log.json",
    # 'interval_6': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_6/20231228_220153.log.json",
    'interval_7': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_7/20231228_220511.log.json",
    # 'interval_8': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_8/20231229_105426.log.json",
    'interval_9': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_9/20231229_094552.log.json",
    # 'interval_10': "work_dirs/video_perpixel_decouple_plus_beitv2_adapter_large_896_80k_visha_ms/ablation/interval_10/20231229_105235.log.json",

}
# data_path_2 = "work_dirs/video_mask2former_edge_branch_beitv2_adapter_large_896_80k_visha_ms/20231128_233959.log.json"
def read_json(data_path, stage, key_1, key_2):
    iter, acc_seg = [], []
    with open(data_path, 'r') as f:
        content = f.readlines()
        i=1
        for line in content:
            if i == 1:
                i += 1
                continue
            if json.loads(line)["mode"] == stage:
                iter.append(i)
                acc_seg.append(json.loads(line)[key_2])
            i += 1
    return iter, acc_seg

max_val = 0
acc_list = {}
iter_list = {}
loss_list = {}
iter_2_list = {}
for item in data_path_list:
    iter_1, acc_seg_1 = read_json(data_path_list[item], "val", "iter", "mIoU")
    iter_2, loss_1 = read_json(data_path_list[item], "train", "iter", "loss")
    # iter_3, acc_seg_3 = read_json(data_path_3, "val", "iter", "mIoU")
    acc_list.update({item:acc_seg_1})
    loss_list.update({item:loss_1})
    iter_list.update({item:iter_1})
    iter_2_list.update({item:iter_2})
    max_value = max(acc_seg_1) if max(acc_seg_1) > max_val else max_val
print(max_value)
#设置横纵坐标的名称以及对应字体格式
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 10,
}
index = 0
# for i in range(len(acc_seg_2)):
#     if acc_seg_2[i] >= 80:
#         index = i
#         break
fig = plt.figure(dpi=100,figsize=(24,10))
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel('iteration', font)
ax.set_ylabel('acc_seg', font)
ax.set_title('Seg Acc', size=10)

for item in acc_list:
    ax.plot(iter_list[item][index:], acc_list[item][index:], label=item)
    # ax.plot(iter_2[index:], acc_seg_2[index:], label="model_2")
ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel('iteration', font)
ax.set_ylabel('loss_train', font)
ax.set_title('Train Loss', size=10)
plt.legend(loc='lower right')
for item in acc_list:
    ax.plot(iter_2_list[item][index:], loss_list[item][index:], label=item)
    # ax.plot(iter_2[index:], acc_seg_2[index:], label="model_2")
plt.legend(loc='upper right')
plt.show()
