import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
from pathlib import Path
from natsort import natsorted
from pathlib import Path
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler




def parseArgs():
    
    parser = argparse.ArgumentParser()
    #Required arguments
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("main_col_str", type=str, help="String common to all main_plot columns")

    #--------------------優先度高め--------------------
    
    #カラムを識別するためのもの
    parser.add_argument("-sid", "--sample_id", type=str, default=None,
                        help="the column include sample_id. if None, the first column of dataset will be used.")
    parser.add_argument("-clsc", "--cluster_col", type=str, default=None,
                        help="the column include clusters. if None, the first column of sub_plot will be used.")
    
    #データフレームの前処理に関わるもの
    parser.add_argument("-thc", "--thd_col", type=list, default=None,
                        help="the columns list you want to threshold.")
    parser.add_argument("-th", "--threshold", type=list, default=None,
                        help="the threshold list you want to use with thd_col.")
    parser.add_argument("-flc", "--fillna_col", type=list, default=None,
                        help="the columns list you want to convert to missing values.")
    parser.add_argument("-msv", "--missing_value", type=list, default=None,
                        help="the value or string list you want to use with fillna_col.")
    parser.add_argument("-ctgc", "--ctg_col", type=list, default=None,
                        help="the columns that need to be label encoded. if None, all columns including categorical values in sub_plot will be used.")
    parser.add_argument("-mscl", "--main_scale", type=bool, default=False,
                        help="if true, main_plot columns will be standardized.")
    
    #描画するカラムやその順序、描画の見た目に関わるもの
    parser.add_argument("-spo", "--sub_plot_order", type=list, default=None
                        help="the columns name and order for sub_plot. if None, the others than sample_id and main_plot columns will be used.")
    parser.add_argument("-srtc", "--sort_col", type=list, default=None,
                        help="the columns you want to sort by. if None, sorting will be performed only on the cluster_col.")
    parser.add_argument("-cml", "--cmap_list", type=list, default=None,
                        help="the colors list you want to use with plotting.")
    parser.add_argument("-rsc", "--right_side_col", type=list, default=None,
                        help="the columns list you want to plot on right side in add_plot")
    
    #outputの設定
    parser.add_argument("-od", "--output_dir", type=str, default=None) #same as input_dir
    parser.add_argument("-on", "--output_name", type=str, default=None) #same as input_dir
    parser.add_argument("-of", "--output_format", type=str, default="png", choices=['png', 'jpg', 'pdf'])
    parser.add_argument("-dpi", "--dpi", type=float, default=100)
    
    #--------------------優先度低め--------------------
    
    #figure領域の設定
    parser.add_argument("-fgs", "--figsize", type=list, default=(8, 8)) #or tuple
    
    #master_gridのパラーメータ設定
    parser.add_argument("-smhr", "--sub_main_hratio", type=list, default=(3,7)) #or tuple
    parser.add_argument("-mwr", "--master_w_ratio", type=list, default=(7,3)) #or tuple
    parser.add_argument("-mhs", "--master_hspace", type=float, default=0.05)
    parser.add_argument("-mws", "--master_wspace", type=float, default=0.01)
    
    #sub_plot横のテキストのパラメータ設定
    parser.add_argument("-stxx", "--sub_text_x", type=float, default=0)
    parser.add_argument("-stxy", "--sub_text_y", type=float, default=0.1)
    parser.add_argument("-stxfs", "--sub_text_font_size", type=float, default=12)
    
    #cbar_gridのパラーメータ設定
    parser.add_argument("-cbhr", "--cbar_h_ratio", type=list, default=(5,5)) #or tuple
    parser.add_argument("-cbwr", "--cbar_w_ratio", type=list, default=(1,9))
    parser.add_argument("-cbws", "--cbar_wspace", type=float, default=0.2) #z-scoreの右側にspaceを作る
    
    #cbarのパラーメータ設定
    parser.add_argument("-cbtfs", "--cbar_tick_font_size", type=float, default=8)
    parser.add_argument("-cbt", "--cbar_title", type=str, default="Z-score")
    parser.add_argument("-cbtfs", "--cbar_title_font_size", type=float, default=8)
    parser.add_argument("-ctp", "--cbar_title_pad", type=float, default=8) #cbarのtitle上部のspace
    parser.add_argument("-cbkws", "--cbar_kws", type=dict, default={"ticks":[-2,-1,0,1,2]})
    parser.add_argument("-cbvmn", "--cbar_vmin", type=float, default=-2)
    parser.add_argument("-cbvmx", "--cbar_vmin", type=float, default=2)
    parser.add_argument("-cbctr", "--cbar_center", type=float, default=0)
    
    #add_gridのパラーメータ設定
    parser.add_argument("-adhr", "--add_h_ratio", type=list, default=(-0.25, 10)) #or tuple
    parser.add_argument("-adwr", "--add_w_ratio", type=list, default=(1,1)) #or tuple
    parser.add_argument("-adws", "--add_wspace", type=float, default=0.0) #左右列の中間にspaceを作る
    parser.add_argument("-adhs", "--add_hspace", type=float, default=0.8) #各add_heatmapの上下にspaceを作る
    
    #add_plotのパラメータ設定
    parser.add_argument("-adtfs", "--add_title_font_size", type=float, default=8)
    parser.add_argument("-adtp", "--add_title_pad", type=float, default=3.5) #add_plotのtitle上部のspace
    parser.add_argument("-adtr", "--add_tick_rotate", type=float, default=0)
    parser.add_argument("-adtcfs", "--add_tick_font_size", type=float, default=7)
    parser.add_argument("-adtcp", "--add_tick_pad", type=float, default=0) #add_plotのラベル右側のspace
    
    #分割線(vlines)のパラーメータ設定
    parser.add_argument("-vlw", "--vlines_width", type=float, default=0.8)
    parser.add_argument("-vlc", "--vlines_color", type=str, default="white")
    
    
    #Printing arguments to the command line
    args = parser.parse_args()

    print("Called with args:")
    print(f"{args}\n")

    return args


#全角表記を半角表記に変換する関数
def unic_norm(string):
    if str(string) != "nan": return unicodedata.normalize('NFKC', str(string))
    else: return string


#欠損値を伴うラベルエンコード用の関数
def label_encode(df, cols):
    
    le_dic = {}
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
        le_dic[col] = le
    
    return df, le_dic


#add_plotの描画時に使用する変数を作成する
#カラムのユニーク数が左右列で同じになるように大きい順に割り振り
#差が出た場合はダミーの空間を追加して調整する
def add_plot_adjuster(df, columns):

    #ユニーク数とカラム名のリスト
    uni_list = [[len(df[c].unique()), c] for c in columns]

    n_left = 0
    n_right = 0
    right = []
    left = []
    #逆順にソートしてn_uniの大きい順に試行(合計値を調整しやすい)
    for li in sorted(uni_list)[::-1]:

        #right_side_colの指定がある場合
        if right_side_col != None:
            if li[1] in right_side_col:
                n_right += li[0]
                right.append([li[0], li[1]])
            elif n_left > n_right:
                n_right += li[0]
                right.append([li[0], li[1]])
            else:
                n_left += li[0]
                left.append([li[0], li[1]])

        #right_side_colの指定がない場合
        else:
            if n_left > n_right:
                n_right += li[0]
                right.append([li[0], li[1]])
            else:
                n_left += li[0]
                left.append([li[0], li[1]])

    #グリッド内で使用する高さの比
    left_ratio = [n[0] for n in sorted(left)]
    right_ratio = [n[0] for n in sorted(right)]

    #左右のunique合計値に差がある場合は、少ない方にダミーの空間を作る
    if n_right > n_left: left_ratio.append(n_right - n_left)
    elif n_right < n_left: right_ratio.append(n_left - n_right)
    
    #add_plotのカラム名(描画順)
    add_plot = [li[1] for li in sorted(left) + sorted(right)]
    
    return add_plot, left_ratio, right_ratio



