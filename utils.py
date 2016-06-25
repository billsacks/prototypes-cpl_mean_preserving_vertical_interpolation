import numpy as np

def find_index_from_lat_lon(lat_array, lon_array, lat_target, lon_target):
    target_point = np.where(np.logical_and(abs(lat_array - lat_target) < 1.e-6,
                                           abs(lon_array - lon_target) < 1.e-6))
    if len(target_point[0]) < 1:
        raise ValueError('Target lat/lon not found for lat={:6.3f} lon={:6.3f}'.format(
            lat_target, lon_target))
    if len(target_point[0]) > 1:
        raise ValueError('Target lat/lon found for multiple points for lat={:6.3f} lon={:6.3f}'.format(
            lat_target, lon_target))
    return target_point

def get_l2x_data(cplhist, cplhist_index):
    topo = np.zeros(10)
    smb = np.zeros(10)
    for ec in range(10):
        topo_str = 'l2x_Sl_topo{:02d}'.format(ec+1)
        topo[ec] = cplhist.variables[topo_str][:][cplhist_index]
        smb_str = 'l2x_Flgl_qice{:02d}'.format(ec+1)
        smb[ec] = cplhist.variables[smb_str][:][cplhist_index]

    return (topo, smb)

