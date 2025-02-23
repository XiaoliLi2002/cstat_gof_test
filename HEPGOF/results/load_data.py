import re
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
import seaborn as sns

# 设置全局样式
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def parse_filename(filename):
    """使用正则表达式解析文件名"""
    pattern = r"results_n(\d+)_beta([\d._]+)_strue(\w+)_snull(\w+)_strength(\d+)_iters([\d.]+)\.xlsx"
    match = re.fullmatch(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式不匹配: {filename}")

    # 解析beta参数（支持整数和浮点数）
    beta_str = match.group(2).replace("_", " ")  # 将分隔符统一转为空格
    beta_values = list(map(float, beta_str.split()))

    return {
        "n": int(match.group(1)),
        "beta": beta_values,
        "strue": match.group(3),
        "snull": match.group(4),
        "strength": int(match.group(5)),
        "iters": float(match.group(6))
    }


def load_data(target_params,test="one-sided"):
    """加载符合目标参数的数据"""
    if test == "one-sided": # one-sided or two-sided
        data_dir = "data/onesided"
    else:
        data_dir = "data/twosided"
    files = Path(data_dir).glob("results_*.xlsx")
    rows = []

    for file_path in files:
        try:
            params = parse_filename(file_path.name)
        except ValueError:
            continue

        # 检查参数匹配
        match = all(
            params.get(k) == v
            for k, v in target_params.items()
            if k != "strue"  and k!="strength"# 特殊处理关系条件
        )

        if match:
            df = pd.read_excel(file_path)
            # 添加关系标记列
            df["is_null"] = (params["strue"] == params["snull"])
            if params["strue"]==target_params["snull"] or (params["strue"]==target_params["strue"] and params["strength"]==target_params["strength"]):
                rows.append(df)

    return pd.concat(rows) if rows else pd.DataFrame()


def plot_separate_figures(target_params, save_prefix=None):

    def create_prefix(params):
        parts = []
        for key in params:  # 保持参数原始顺序
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

    # 使用自定义前缀或自动生成
    final_prefix = save_prefix if save_prefix else create_prefix(target_params)

    """绘制并保存两张优化后的图表"""
    # 加载数据
    full_df = load_data(target_params)
    if full_df.empty:
        print("没有找到匹配的数据")
        return

    methods = ["Chisq", "Oracle", "Plug_in", "Uncond", "Cond", "Modified"]
    colors = sns.color_palette(n_colors=len(methods))  # 固定颜色方案

    # Type I Error 图表 ==============================================
    plt.figure(figsize=(8, 6))
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
    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line,
             'k--', alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    plt.xlabel("Nominal significance level (α)")
    plt.ylabel("Type I Error Rate")
    plt.title("Type I Error vs α")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figure/{final_prefix}_type1_error.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Power 图表 =====================================================
    plt.figure(figsize=(8, 6))
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
    plt.plot(ref_line, ref_line, 'k--',
             alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    plt.xlabel("Type I Error Rate")
    plt.ylabel("Power")
    plt.title("Power vs Type I Error")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figure/{final_prefix}_power_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 指定需要固定的参数
    target_params = {
        "n": 50,
        "beta": [1.0, 1.],  # 自动匹配beta参数
        "strue": "brokenpowerlaw",
        "snull": "powerlaw",
        "strength": 2,
        "iters": 10000.0,
    }
    plot_separate_figures(target_params)