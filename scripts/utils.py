import os

def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if 'config.json' in pth_list:
            pth_list.remove('config.json')
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:float(x.split('-')[-1].split('=')[-1].split('.pth')[0]))
            return os.path.join(ckpt_path,pth_list[0])
        else:
            return None
    else:
        return None
    
def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:float(x.split('-')[1].split('=')[-1]))
            for pth_item in pth_list[retain:]:
                os.remove(os.path.join(ckpt_path,pth_item))

def dfs_remove_weight(ckpt_path=None,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  

if __name__ == "__main__":
    dfs_remove_weight()