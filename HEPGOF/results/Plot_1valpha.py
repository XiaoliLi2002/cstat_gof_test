from load_data import *

# 设置全局样式
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def create_prefix(params):
    parts = []
    for key in params:  # 保持参数原始顺序
        if key=="beta":
            continue
        value = params[key]

        # 处理不同值类型
        if isinstance(value, list):
            val_str = "_".join(map(str, value))
        elif isinstance(value, float) and value.is_integer():
            val_str = str(int(value))
        else:
            val_str = str(value)

        # 移除特殊字符并组合
        clean_str = f"{key}{val_str}".replace(" ", "").lower()
        parts.append(clean_str)
    return "_".join(parts)

def format_params(params):
    """将参数字典转换为可读字符串"""
    parts = []
    for k, v in params.items():
        if k == "strue_snull_relation" or k=="beta" or k=="strength":  # 跳过关系参数
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

def plot_1vsalpha_single(target_params, save_prefix=None, test='one-sided'):
    full_df = load_data(target_params,test=test)
    if full_df.empty:
        print("没有找到匹配的数据")
        return

    methods = ["Chisq", "Oracle", "Plug_in", "Uncond", "Cond", "Modified"]
    colors = sns.color_palette(n_colors=len(methods))  # 固定颜色方案

    # Type I Error 图表 ==============================================
    null_df = full_df[full_df["is_null"]].sort_values("alpha")
    min_alpha = null_df["alpha"].min()
    max_alpha = null_df["alpha"].max()

    for method, color in zip(methods, colors):
        # 准备数据
        grouped = null_df.groupby("alpha", as_index=False)[method].mean()
        x = grouped["alpha"].values
        y = grouped[method].values

        # 生成平滑曲线
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x, y

        # 绘制主曲线
        plt.plot(x_smooth, y_smooth,
                 linewidth=2, color=color,
                 label=method, zorder=2)

        # 每隔6个数据点添加标记
        mask = (grouped.index % 6 == 0)  # 修改这里调整间隔
        plt.scatter(x[mask], y[mask],
                    color=color, edgecolor='white',
                    s=60, zorder=3, linewidth=0.8,
                    marker='o')

    # 绘制动态参考线
    plt.xlabel("Nominal significance level (α)")
    plt.ylabel("Type I Error Rate")
    plt.title(f"beta={target_params['beta']}", fontsize=14, pad=20)
    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line,
             'k--', alpha=0.5, linewidth=1,
             label='y=x', zorder=1)
    plt.grid(True, alpha=0.3)


# 使用示例
if __name__ == "__main__":
    # 指定需要固定的参数
    test='one-sided' # one-sided or two-sided
    beta1=[1.0,2.5,5.0,10.0]
    target_params = [{
        "n": 50,
        "beta": [i, 1.],  # 自动匹配beta参数
        "strue": "powerlaw",
        "snull": "powerlaw",
        "strength": 2,
        "iters": 10000.0,
    } for i in beta1]

    params_str = format_params(target_params[0])
    plt.figure(figsize=(12, 9))
    for i in range(len(beta1)):
        plt.subplot(2,2,i+1)
        plt.suptitle(f"Type I Error vs α\n({params_str})", fontsize=14)
        plot_1vsalpha_single(target_params[i], test=test)
    plt.tight_layout()
    final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figure/{test}/{final_prefix}_type1vsalpha_all.png", dpi=300, bbox_inches='tight')

