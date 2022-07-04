from multiprocessing import Queue


class EvalQueueSet:
    def __init__(self):
        # 训练集用
        self.eval_data_queue = Queue()
        self.eval_result_queue = Queue()

        # 测试集用
        self.test_eval_data_queue = Queue()
        self.test_eval_result_queue = Queue()


class StreamDataQueueSet:
    def __init__(self):
        self.data_queue = Queue()
        self.start_flag_queue = Queue()
        self.stop_flag_queue = Queue()

    def clear(self):
        self.data_queue.close()
        self.start_flag_queue.close()
        self.stop_flag_queue.close()


class ModelUpdateQueueSet:
    def __init__(self):
        self.data_queue = Queue()
        self.re_embedding_flag_queue = Queue()

    def clear(self):
        self.data_queue.close()
        self.re_embedding_flag_queue.close()
