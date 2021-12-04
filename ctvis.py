import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob


def _get_df(base_path='public-covid-data', folder='rp_im'):

    data_dict = {'FilePath': glob('{}/{}/*'.format(base_path, folder)),
                 'FileName': [p.split('/')[-1] for p in glob('{}/{}/*'.format(base_path, folder))]}
    data_df = pd.DataFrame(data_dict)
    return data_df


def get_df_all(base_path='public-covid-data'):
    """
    base_path以下のファイルパスをDataFrame(FilePathImage, FileName, FilePathMask)で返す．

    Parameters
    -----------------
        base_path(str):　データを保存しているフォルダのパス

    Return
    ------------------
        DataFrame:
            FilePathImage: イメージファイルのパス
            FileName: ファイル名
            FilePathMask: マスクファイルのパス
    """

    rp_im_df = _get_df(base_path, 'rp_im')
    rp_msk_df = _get_df(base_path, 'rp_msk')
    return rp_im_df.merge(rp_msk_df, on='FileName', suffixes=('Image', 'Mask'))


def load_nifti(path):
    """
    NIfTIファイルをNumPy Array(shape=(h, w, z))として読み込む
    
    Parameters
    ------------------
        path(str): NIfTIファイルのパス
    
    Return
    ------------------
        NumPy Array (shape=(h, w, z))
    """
    nifti = nib.load(path)
    data = nifti.get_fdata()
    return np.rollaxis(data, 1, 0)


def label_color(mask_volume,
                ggo_color=[255, 0, 0],
                consolidation_color=[0, 255, 0],
                effusion_color=[0, 0, 255]):
    """
    Maskデータ(h, w, z)をRGBのカラーデータ(h, w, z, 3)に変換する
    
    Parameters
    -------------------
        mask_volume(ndarray): (h, w, z)の値が0~3までのNumPy Array
        ggo_color(list): GGOラベル(ラベル=1)のRGB値
        consolidation_color(list): Consolidationラベル(ラベル=2)のRGB値
        effusion_color(list): Plural effusionラベル(ラベル=3)のRGB値
    
    Return
    -------------------
        mask_color(ndrray): MaskデータのRGBデータ(h, w, z, 3)
    """
    
    shp = mask_volume.shape
#     箱作成
    mask_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    mask_color[np.equal(mask_volume, 1)] = ggo_color
    mask_color[np.equal(mask_volume, 2)] = consolidation_color
    mask_color[np.equal(mask_volume, 3)] = effusion_color
    return mask_color


def hu_to_gray(volume):
    """
    CTデータをHUからグレースケール(0-255)に変換する
    
    Parameters
    --------------------
        volume(ndarray): CTデータ
    
    Return
    --------------------
        volume_rerange(ndarray):　0-255のグレースケール
    """

    maxhu = np.max(volume)
    minhu = np.min(volume)
    volume_rerange = (volume - minhu)/max((maxhu - minhu), 1e-3)
    volume_rerange = volume_rerange * 255

    volume_rerange = np.stack([volume_rerange, volume_rerange, volume_rerange], axis=-1)
    return volume_rerange.astype(np.uint8)


def overlay(gray_volume, mask_volume, mask_color, alpha=0.3):
    """
    グレースケールのCTとマスクデータを重ね合わせる(overlay)
    
    Parameters
    ---------------------
        gray_volume(ndarray): グレースケール(0-255)のCTデータ(shape=(h, w, z, 3))
        mask_volume(ndarray): マスクデータ(shape=(h, w, z))
        mask_color(ndarray): マスクデータのRGBデータ(shape=(h, w, z, 3))
        alpha(float): 0.0-1.0でマスクの透明度．　0に近いほど透明
    
    Return
    ---------------------
        overlayed(ndarray): CTとマスクのoverlayされたNumPy Array (shape=(h, w, z, 3))
    
    """

    # ラベルが付いているピクセルのみoverlayするためのフィルタ
    mask_filter = np.greater(mask_volume, 0)
    # RGBチャネルに合わせてstackしてチャネル数を3つにする
    mask_filter = np.stack([mask_filter, mask_filter, mask_filter], axis=-1)
    overlayed = np.where(mask_filter,
         ((1-alpha)*gray_volume + alpha*mask_color).astype(np.uint8),
         gray_volume)
    return overlayed


def vis_overlay(overlayed, original_volume, mask_volume, cols=5, display_num=25, figsize=(15, 15)):
    """
    CTのaxial画像を指定した枚数等間隔に抜き出して表示する．スライスのindex番号と各ラベルのHUの統計量も合わせて表示する．
    
    Parameters
    -----------------------
        overlayed(ndarray): 表示する対象のデータ(shape=(h, w, z, 3))
        original_volume(ndarray): 統計量を計算するための元のCTデータ(HU) (shape=(h, w, z))
        mask_volume(ndarray): マスクデータ(shape=(h, w, z))
        cols(int): 表示する列数
        display_num(int): 表示する枚数
        figsize(tuple): 表示する画像のサイズ
    """
    
    total_num = overlayed.shape[-2]
    interval = total_num / display_num
    rows = (display_num - 1) // cols + 1
    # 同じ画像を表示する意味はないので，intervalは最小でも1になるようにする
    if interval < 1:
        interval = 1
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(display_num):
        indx = int(i * interval)
        # out of indexを避ける為にbreak
        if indx >= total_num:
            break

        row_i = i//cols
        col_i = i%cols
        stats = _get_hu_stats(original_volume[:, :, indx], mask_volume[:, :, indx])
        # 表示する文字列を作成(for文で回してもいいですが，なにが表示されるかわかりにくくなるので，こういう場合は複数行に渡って書いてもいいと思います)
        title = 'slice #: {}'.format(indx)
        title += '\nggo mean:{:.0f}±{:.0f}'.format(stats['ggo_mean'], stats['ggo_std'])
        title += '\nconsli mean:{:.0f}±{:.0f}'.format(stats['consolidation_mean'], stats['consolidation_std'])
        title += '\neffusion mean:{:.0f}±{:.0f}'.format(stats['effusion_mean'], stats['effusion_std'])
        ax[row_i, col_i].imshow(overlayed[:, :, indx])
        ax[row_i, col_i].set_title(title)
        #画像のサイズは全て同じなので，軸を非表示にする
        ax[row_i, col_i].axis('off')
    fig.tight_layout()


#講義ではprivate functionにしてませんでしたが，この場合はprivate functionとして定義してもいい気がします
def _get_hu_stats(volume, mask_volume, label_dict={1: 'ggo_', 2: 'consolidation_', 3: 'effusion_'}):

    result = {}

    for label in range(1, 4):
        prefix = label_dict[label]
        roi_hu = volume[np.equal(mask_volume, label)]
        result[prefix+'mean'] = np.mean(roi_hu)
        result[prefix+'std'] = np.std(roi_hu)

    return result

