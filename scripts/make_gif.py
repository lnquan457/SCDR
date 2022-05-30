import imageio
import os
import numpy as np
import cv2


def create_gif(image_list, gif_name, total_frames):
    frames = []
    img_index = 0
    cur_img_step = int(image_list[img_index].split("/")[-1].split("_")[-1].split(".")[0])
    for i in range(1, total_frames + 1):
        if i > cur_img_step:
            img_index += 1
            cur_img_step = int(image_list[img_index].split("/")[-1].split("_")[-1].split(".")[0])
        frames.append(imageio.imread(image_list[img_index]))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF')  # duration:秒
    print("GIF成功保存到", gif_name)
    return


def create_video(image_list, save_path, total_frames, size, fps=30):
    # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    video_writer = cv2.VideoWriter(save_path, -1, fps, size)
    img_index = 0
    cur_img_step = int(image_list[img_index].split("\\")[-1].split("_")[-1].split(".")[0])

    for i in range(1, total_frames + 1):
        if i > cur_img_step:
            img_index += 1
            cur_img_step = int(image_list[img_index].split("/")[-1].split("_")[-1].split(".")[0])

        img = cv2.imread(image_list[img_index])
        if img is None:
            print(image_list[img_index] + " is not exists!")
            continue
        video_writer.write(img)

    video_writer.release()
    print("视频成功保存到", save_path)


def make():
    root_dir = '../results/初步实验/stream_rate'
    dataset_list = ["isolet_subset", "food"]
    # multi or single
    # for sub_dir_1 in os.listdir(root_dir):
    for sub_dir_1 in ['single cluster']:
        # stream rate
        # for sub_dir_2 in os.listdir(os.path.join(root_dir, sub_dir_1)):
        for sub_dir_2 in ["r50", "r200", "r2"]:
            # method
            for method in os.listdir(os.path.join(root_dir, sub_dir_1, sub_dir_2)):
                # dataset
                # for ds in dataset_list:
                method_dir = os.path.join(root_dir, sub_dir_1, sub_dir_2, method)
                for ds in os.listdir(method_dir):
                    pre_dir = os.path.join(method_dir, ds)
                    process_single_dir(pre_dir)
                    # save_path = os.path.join(pre_dir, "stream.mp4")
                    # create_video(img_list, save_path, total_frames, size=(5158, 5126))


def process_single_dir(base_dir):
    print("正在处理：", base_dir)
    # img_dir = os.path.join(base_dir, os.listdir(base_dir)[0], 'imgs')
    img_dir = os.path.join(base_dir, 'imgs')
    step_list = list(map(lambda x: int(x.split("_")[-1].split(".")[0]), os.listdir(img_dir)))
    re_indices = np.argsort(step_list)
    total_frames = np.max(step_list)
    img_list = np.array([os.path.join(img_dir, item) for item in os.listdir(img_dir)])[re_indices]
    # save_path = os.path.join(os.path.join(base_dir, os.listdir(base_dir)[0]), "stream.gif")
    # save_path = os.path.join(base_dir, "stream.gif")
    # create_gif(img_list, save_path, total_frames)
    save_path = os.path.join(base_dir, "stream.mp4")
    create_video(img_list, save_path, total_frames, size=(5467, 5306))


if __name__ == '__main__':
    # make()
    process_single_dir(r"..\results\RTSCDR\food\final_vc_test")
