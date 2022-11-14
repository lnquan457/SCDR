from multiprocessing import Queue, Value
from threading import Semaphore


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

    STOP = 0
    SAVE = 1
    DATA_STREAM_END = 2
    UPDATE = 3

    def __init__(self):
        self.training_data_queue = Queue()
        self.raw_data_queue = Queue()
        self.embedding_queue = Queue()
        self.flag_queue = Queue()
        self.WAITING_UPDATED_DATA = Value("b", False)
        self.INITIALIZING = Value("b", False)
        self.MODEL_UPDATING = Value("b", False)

    def clear(self):
        self.training_data_queue.close()
        self.embedding_queue.close()
        self.flag_queue.close()


# 用于对新数据进行处理时的数据传输
class DataProcessorQueueSet:

    def __init__(self):
        self.data_queue = Queue()

    def clear(self):
        self.data_queue.close()


# 用于传输各阶段用时数据
class CalTimeQueueSet:
    def __init__(self):
        self.knn_approx_queue = Queue()
        self.knn_update_queue = Queue()
        self.model_initial_queue = Queue()
        self.model_update_queue = Queue()
        self.training_data_sample_queue = Queue()

    def close(self):
        self.knn_approx_queue.close()
        self.knn_update_queue.close()
        self.model_update_queue.close()
        self.model_initial_queue.close()
        self.training_data_sample_queue.close()
