import multiprocessing
import time


# 假设这是你的原始测试函数
def main(iteration):
    # 这里替换成你的实际计算逻辑
    time.sleep(1)  # 模拟耗时操作
    method1_p = 0.05  # 示例值
    method2_p = 0.04  # 示例值
    return {
        "method1": method1_p,
        "method2": method2_p,
        "iteration": iteration
    }


def process_iteration(iteration):
    """ 包装函数用于处理单个迭代 """
    try:
        result = main(iteration)
        print(f"完成迭代 {iteration}")
        return result
    except Exception as e:
        print(f"迭代 {iteration} 出错: {str(e)}")
        return None


if __name__ == '__main__':
    # 配置参数
    TOTAL_ITERATIONS = 1000
    WORKERS = multiprocessing.cpu_count()  # 使用全部CPU核心

    # 创建进程池
    with multiprocessing.Pool(processes=WORKERS) as pool:
        # 使用imap_unordered获取结果（完成顺序可能乱序但更快）
        start_time = time.time()

        results = []
        for i, result in enumerate(pool.imap_unordered(process_iteration, range(TOTAL_ITERATIONS))):
            if result:
                results.append(result)
            # 打印进度
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / (i + 1) * (TOTAL_ITERATIONS - i - 1)
                print(f"进度: {i + 1}/{TOTAL_ITERATIONS} | 已用时间: {elapsed:.2f}s | 剩余时间: {remaining:.2f}s")

    # 保存最终结果（这里可以替换成你的保存逻辑）
    print(f"\n所有 {len(results)} 次迭代完成")
    print("示例结果:", results[0])