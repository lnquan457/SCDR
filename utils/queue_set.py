from multiprocessing import Queue


class QueueSet:
    def __init__(self):
        # 训练集用
        self.eval_data_queue = Queue()
        self.eval_result_queue = Queue()

        # 测试集用
        self.test_eval_data_queue = Queue()
        self.test_eval_result_queue = Queue()
