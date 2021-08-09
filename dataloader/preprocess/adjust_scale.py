import os
import h5py
import numpy as np

data_dir = '/home/qianyu.fqy/data/3D'
ori_filename = 'pix3d_2894_chair.hdf5'
new_filename = 'pix3d_rescale_chair.hdf5'
vox_size = 32

pixel3d_path = os.path.join(data_dir, ori_filename)
if os.path.exists(pixel3d_path):
    pixel3d = h5py.File(pixel3d_path, 'r')
else:
    raise Exception("pixel3d not exist")


# pix3d scale (src) to shapenet scale (dst)
def rescale_pix3d_voxel(vxl_src,
                d_src=20.038700760193503, h_src=30.66620594333103, w_src=20.852107809260538,
                d_dst=15.18812246403541, h_dst=22.558465510881593, w_dst=15.704537071191442):
    assert d_src > d_dst
    assert h_src > h_dst
    assert w_src > w_dst
    d, h, w = vxl_src.shape

    d_pad = int(d * (d_src - d_dst) / float(d_dst))
    if d_pad % 2 == 1: d_pad = d_pad + 1
    h_pad = int(h * (h_src - h_dst) / float(h_dst))
    if h_pad % 2 == 1: h_pad = h_pad + 1
    w_pad = int(w * (w_src - w_dst) / float(w_dst))
    if w_pad % 2 == 1: w_pad = w_pad + 1

    vxl = np.zeros((d + d_pad, h + h_pad, w + w_pad), dtype=np.float32)
    vxl[int(d_pad / 2):int(d_pad / 2) + d, int(h_pad / 2):int(h_pad / 2) + h, int(w_pad / 2):int(w_pad / 2) + w] = vxl_src

    s_d = float(d_pad + d) / float(d)
    s_h = float(h_pad + h) / float(h)
    s_w = float(w_pad + w) / float(w)

    vxl_dst = np.zeros_like(vxl_src)
    for i in range(d):
        for j in range(h):
            for k in range(w):
                key_sum = np.sum(vxl[
                                 int(i * s_d):int((i + 1) * s_d),
                                 int(j * s_h):int((j + 1) * s_h),
                                 int(k * s_w):int((k + 1) * s_w)])
                vxl_dst[i, j, k] = 1 if key_sum >= 1.0 else 0

    return vxl_dst

    
new_voxels = []
sample_num = len(pixel3d['pixels'])
for sample_idx in range(sample_num):
    if sample_idx % 10 == 0:
        print(sample_idx, '/', sample_num)
    r_voxel = pixel3d['voxels'][sample_idx].astype(np.float32)
    r_voxel = np.reshape(r_voxel, [1, vox_size, vox_size, vox_size])
    r_voxel = r_voxel.transpose(0,1,3,2)[:,:,:,::-1].astype(np.float32)
    # adjust scale to pixel3d gt
    r_voxel = np.squeeze(r_voxel)
    r_voxel = rescale_pix3d_voxel(r_voxel)
    r_voxel = np.expand_dims(r_voxel, axis=0)
    new_voxels.append(r_voxel)

new_voxels = np.stack(new_voxels, axis=0)

f = h5py.File(os.path.join(data_dir, new_filename), 'w')
f.create_dataset('pixels', data=pixel3d['pixels'])
f.create_dataset('voxels', data=new_voxels)
f.close()
