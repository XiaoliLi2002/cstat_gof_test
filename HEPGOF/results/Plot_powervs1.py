from Plot_1valpha import create_prefix
from load_data import *

# 设置全局样式
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def format_params(params):
    """将参数字典转换为可读字符串"""
    parts = []
    for k, v in params.items():
        if k == "strue_snull_relation" or k=="beta":  # 跳过关系参数
            continue
        # 智能格式化数值
        if isinstance(v, float):
            if v.is_integer():
                val_str = str(int(v))
            else:
                val_str = f"{v:.2f}".rstrip('0').rstrip('.')
        elif isinstance(v, list):
            val_str = ", ".join([f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else str(x) for x in v])
        else:
            val_str = str(v)
        parts.append(f"{k}={val_str}")
    return ", ".join(parts)

def plot_powervs1_single(target_params, save_prefix=None,test='one-sided'):
    full_df = load_data(target_params,test=test)
    if full_df.empty:
        print("没有找到匹配的数据")
        return

    methods = ["Chisq", "Oracle", "Plug_in", "Uncond", "Cond", "Modified"]
    colors = sns.color_palette(n_colors=len(methods))  # 固定颜色方案

    # Power 图表 =====================================================
    null_df = full_df[full_df["is_null"]].sort_values("alpha")
    min_alpha = null_df["alpha"].min()
    max_alpha = null_df["alpha"].max()
    # 合并数据
    merged = pd.merge(
        full_df[full_df["is_null"]].groupby("alpha")[methods].mean(),
        full_df[~full_df["is_null"]].groupby("alpha")[methods].mean(),
        left_index=True, right_index=True,
        suffixes=('_null', '_alt')
    )

    for method, color in zip(methods, colors):
        # 准备数据
        x = merged[f"{method}_null"].values
        y = merged[f"{method}_alt"].values
        alpha_values = merged.index.values

        # 生成平滑曲线（按原始顺序）
        if len(x) > 3:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]

            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            spl = make_interp_spline(x_sorted, y_sorted, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x, y

        # 绘制主曲线
        plt.plot(x_smooth, y_smooth,
                 linewidth=2, color=color,
                 label=method, zorder=2)

        # 每隔6个原始点添加标记
        mask = (np.arange(len(x)) % 6 == 0)  # 修改这里调整间隔
        plt.scatter(x[mask], y[mask],
                    color=color, edgecolor='white',
                    s=60, zorder=3, linewidth=0.8,
                    marker='s')

    # 绘制参考线
    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line, 'k--',
             alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    plt.xlabel("Type I Error Rate")
    plt.ylabel("Power")
    plt.title(f"beta={target_params['beta']}", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)


# 使用示例
if __name__ == "__main__":
    # 指定需要固定的参数
    test = 'one-sided'  # one-sided or two-sided
    beta1=[1.0,2.5,5.0,10.0]
    target_params = [{
        "n": 50,
        "beta": [i, 1.],  # 自动匹配beta参数
        "strue": "brokenpowerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 10000.0,
    } for i in beta1]

    params_str = format_params(target_params[0])
    plt.figure(figsize=(12, 9))
    for i in range(len(beta1)):
        plt.subplot(2,2,i+1)
        plt.suptitle(f"Power vs Type I Error\n({params_str})", fontsize=14)
        plot_powervs1_single(target_params[i],test=test)
    plt.tight_layout()
    final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figure/{test}/{final_prefix}_powervstype1_all.png", dpi=300, bbox_inches='tight')
