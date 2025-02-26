import multiprocessing
import time
import os
from typing import Set


# 文件名生成函数保持不变
def generate_filename(data_type: str) -> str:
    """生成对应数据类型的文件路径"""
    base_dir = "results/data"
    if data_type == "onesided":
        return os.path.join(base_dir, "pvalues", "onesided", "pvalues_onesided.csv")
    elif data_type == "twosided":
        return os.path.join(base_dir, "pvalues", "twosided", "pvalues_twosided.csv")
    elif data_type == "time":
        return os.path.join(base_dir, "time", "execution_time.csv")
    else:
        raise ValueError(f"无效的数据类型: {data_type}")


# 测试函数保持不变
def main(iteration: int) -> dict:
    time.sleep(1)  # 保持模拟耗时操作

    p = 3  # 保持示例数据
    return {
        "pvalue_onesided": [0.01 * (iteration + 1) for _ in range(p)],
        "pvalue_twosided": [0.02 * (iteration + 1) for _ in range(p)],
        "time": [0.1 * (iteration + 1) for _ in range(p)],
        "iteration": iteration
    }


def get_existing_iterations() -> Set[int]:
    """获取所有结果文件中已存在的迭代号"""
    existing = set()
    for data_type in ['onesided', 'twosided', 'time']:
        path = generate_filename(data_type)
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if parts and parts[0].isdigit():
                        existing.add(int(parts[0]))
    return existing


def save_single_result(result: dict):
    """安全保存单个迭代的结果"""
    try:
        # 准备数据行（包含迭代号）
        onesided_line = f"{result['iteration']}," + ",".join(f"{x:.6f}" for x in result["pvalue_onesided"]) + "\n"
        twosided_line = f"{result['iteration']}," + ",".join(f"{x:.6f}" for x in result["pvalue_twosided"]) + "\n"
        time_line = f"{result['iteration']}," + ",".join(f"{x:.4f}" for x in result["time"]) + "\n"

        # 原子化写入操作
        for data_type, content in [('onesided', onesided_line),
                                   ('twosided', twosided_line),
                                   ('time', time_line)]:
            path = generate_filename(data_type)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', buffering=1) as f:  # 行缓冲模式
                f.write(content)
    except Exception as e:
        print(f"保存迭代 {result['iteration']} 失败: {str(e)}")


def process_iteration(iteration: int):
    """处理单个迭代的完整流程"""
    try:
        # 跳过已存在的迭代（二次检查）
        if iteration in existing_iterations:
            return None

        result = main(iteration)
        save_single_result(result)
        print(f"完成迭代 {iteration}")
        return iteration
    except Exception as e:
        print(f"迭代 {iteration} 出错: {str(e)}")
        return None


if __name__ == '__main__':
    # 初始化设置
    TOTAL_ITERATIONS = 1000
    WORKERS = multiprocessing.cpu_count()
    existing_iterations = get_existing_iterations()

    # 准备任务列表
    all_iterations = set(range(TOTAL_ITERATIONS))
    remaining_iterations = sorted(list(all_iterations - existing_iterations))
    print(f"发现 {len(existing_iterations)} 个已完成的迭代，剩余 {len(remaining_iterations)} 个待处理")


    # 进度跟踪函数
    def track_progress(start_time, total, completed):
        if completed == 0:
            return
        elapsed = time.time() - start_time
        avg = elapsed / completed
        remaining = avg * (total - completed)
        print(f"进度: {completed}/{total} | 已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s")


    # 并行处理
    with multiprocessing.Pool(processes=WORKERS) as pool:
        start_time = time.time()
        processed_count = 0

        # 使用异步迭代器
        results = pool.imap_unordered(process_iteration, remaining_iterations)

        # 处理结果
        for i, result in enumerate(results, 1):
            if result is not None:
                processed_count += 1

            # 每10次或最后次打印进度
            if i % 10 == 0 or i == len(remaining_iterations):
                track_progress(start_time, len(remaining_iterations), i)

        print(f"\n所有迭代处理完成，有效完成数: {processed_count}")
        print(f"总耗时: {time.time() - start_time:.1f}秒")