import os

import h5py
import numpy as np
import pandas

from do_abalation_eval import build_metric_tool
from utils.metrics_tool import cal_global_position_change

if __name__ == '__main__':
    situation = "PD"
    dataset_list = ["arem", "basketball", "shuttle", "HAR_2", "covid_twi", "mnist_fla"]
    metric_list = ["Trust", "Continuity", "Neighbor Hit", "KA(10)", "Shepard Goodness", "DEMaP", "Position Change"]
    batch_num = 1000
    inc_time = 5
    eval_k = 10

    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    save_dir = r"D:\Projects\流数据\Evaluation\消融实验\质量约束"
    final_res = np.zeros(shape=(len(dataset_list), len(metric_list), inc_time + 1, 2))

    for i, dataset in enumerate(dataset_list):
        print("Processing {}".format(dataset))
        dataset_dir = os.path.join(save_dir, dataset)
        dataset_res_dir = os.path.join(dataset_dir, os.listdir(dataset_dir)[0])
        embedding_dir = os.path.join(dataset_res_dir, "embeddings")
        total_res = np.zeros(shape=(len(metric_list), inc_time + 1, 2))

        with h5py.File(os.path.join(data_dir, "{}.h5".format(dataset)), "r") as hf:
            x = np.array(hf['x'], dtype=float)
            y = np.array(hf['y'], dtype=int)

        initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)),
                                                 allow_pickle=True)
        initial_data = x[initial_indices]
        initial_label = y[initial_indices]

        stream_data = x[after_indices]
        stream_label = y[after_indices]

        pre_n_samples = initial_data.shape[0]
        initial_data_num = pre_n_samples
        stream_idx = 0
        total_data = initial_data
        total_label = initial_label
        metric_log_file = open(os.path.join(dataset_res_dir, "metric_log.txt"), "w")
        pre_initial_embeddings = None
        pre_new_embeddings = None

        for j in range(inc_time + 1):
            print("Eval Step {}".format(j))
            total_embeddings = np.load(os.path.join(embedding_dir, "embedding_{}.npy".format(j)))
            cur_initial_embeddings = total_embeddings[:initial_data_num]

            if j > 0:
                cur_new_embeddings = total_embeddings[-batch_num:] if j > 0 else None
                cur_new_data_indices = np.arange(pre_n_samples, pre_n_samples + batch_num).astype(int)
                cur_new_data = stream_data[stream_idx:stream_idx + batch_num]
                cur_new_label = stream_label[stream_idx:stream_idx + batch_num]
                total_data = np.concatenate([total_data, cur_new_data], axis=0)
                total_label = np.concatenate([total_label, cur_new_label])

            metric_tool = build_metric_tool(dataset, total_data, total_label, eval_k)
            metric_tool.subset_indices = np.arange(pre_n_samples).astype(int)
            initial_metric_res = metric_tool.cal_simplified_metrics(eval_k, total_embeddings, knn_k=eval_k, clear=False, cal_demap=False)
            initial_metric_res = list(initial_metric_res)
            initial_pc = cal_global_position_change(cur_initial_embeddings, pre_initial_embeddings) if j > 0 else 0
            initial_metric_res.append(initial_pc)

            if j > 0:
                metric_tool.subset_indices = cur_new_data_indices
                new_data_metric_res = metric_tool.cal_simplified_metrics(eval_k, total_embeddings, knn_k=eval_k, clear=False, cal_demap=False)
                new_data_metric_res = list(new_data_metric_res)
                new_data_pc = cal_global_position_change(total_embeddings[pre_n_samples:pre_n_samples+pre_new_embeddings.shape[0]],
                                                         pre_new_embeddings) if j > 1 else 0
                new_data_metric_res.append(new_data_pc)
            else:
                new_data_metric_res = np.ones(len(metric_list)).tolist()

            # initial_metric_res = np.ones(6).tolist()
            # initial_metric_res.append(0)
            # new_data_metric_res = np.ones(6).tolist()
            # new_data_metric_res.append(0)

            metric_log_file.write("Step {}\n".format(j))
            output = "Initial: "
            for k, metric_name in enumerate(metric_list):
                total_res[k, j, 0] = initial_metric_res[k]
                output += "%s: %.4f " % (metric_name, initial_metric_res[k])
            print(output)
            metric_log_file.write(output + "\n")

            output = "New: "
            for k, metric_name in enumerate(metric_list):
                total_res[k, j, 1] = new_data_metric_res[k]
                output += "%s: %.4f " % (metric_name, new_data_metric_res[k])
            print(output)
            metric_log_file.write(output + "\n")

            if j > 0:
                pre_new_embeddings = total_embeddings[pre_n_samples:]
                stream_idx += batch_num
                pre_n_samples += batch_num
            pre_initial_embeddings = cur_initial_embeddings

        for j, metric_name in enumerate(metric_list):
            excel_save_path = os.path.join(dataset_dir, "{}.xlsx".format(metric_name))
            res_dict = {'Type': ['Initial', 'New']}
            for k in range(inc_time + 1):
                res_dict['Step {}'.format(k)] = total_res[j, k]
            res_dict['Avg'] = np.mean(total_res[j], axis=0)
            df = pandas.DataFrame(res_dict)
            df.to_excel(excel_save_path)

        final_res[i] = total_res

    for i, metric_name in enumerate(metric_list):
        excel_save_path = os.path.join(save_dir, "{}.xlsx".format(metric_name))
        res_dict = {'Type': ['Initial', 'New']}
        tmp_mean = []
        for k in range(inc_time + 1):
            res_dict['Step {}'.format(k)] = np.mean(final_res[:, i, k], axis=0)
            tmp_mean.append(np.mean(final_res[:, i, k], axis=0))
        res_dict['Avg'] = np.mean(np.array(tmp_mean), axis=0)
        df = pandas.DataFrame(res_dict)
        df.to_excel(excel_save_path)