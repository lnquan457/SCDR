import argparse
import matplotlib.pyplot as plt
import numpy as np

ax = plt.gca()
[ax.spines[i].set_visible(False) for i in ["top", "right"]]


def minute2float(data):
    """

    :param data: 字符串格式的数据，h:m
    :return: 浮点值的小时数据
    """
    def transform(d):
        h, m = d.split(":")
        return float(h) + float(m) / 60

    ret = []
    for day_items in data:
        day_ret = []
        for item in day_items:
            sta, end = item
            day_ret.append([transform(sta), transform(end)])
        ret.append(day_ret)
    return ret


def gatt(period, dates):
    """
    周一至周六工作时长甘特图绘制
    :param dates: 日期列表，y轴
    :param period: 周一至周六每天的签到时间和签退时间
    :return:
    """
    hour_data = minute2float(period)
    total = 0
    for i in range(len(period)):
        day = dates[i]
        for j in range(len(period[i])):
            sta_time, end_time = hour_data[i][j]

            duration = end_time - sta_time
            total += duration
            sta_text, end_text = period[i][j]

            plt.barh(day, duration, left=sta_time, height=0.1, color="green")
            if not sta_text == end_text:
                plt.text(sta_time + 0.1, i + 0.2, '%s ~ %s, %.1f h' % (sta_text, end_text, duration), color="black", size=10)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="date.txt")

    periods = []
    with open("date.txt") as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            # print(idx, line)
            text = line.replace("\n", "")
            split_periods = text.split(";")
            day_data = []
            for item in split_periods:
                if len(item) == 0:
                    day_data.append(["8:00", "8:00"])
                else:
                    sta, end = item.split(" ")
                    day_data.append([sta, end])

            periods.append(day_data)

    y = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat"]
    # t = [[["9:58", "22:20"]], [["8:42", "12:10"], ["14:31", "22:12"]], [["9:55", "22:16"]], [["13:53", "22:15"]], [["13:47", "22:14"]],
    #      [["13:30", "22:15"]]]

    total_hour = gatt(periods, y)
    plt.title("Weekly Working Info - %.1fh" % total_hour, pad=20)
    plt.xlabel("Hour")
    plt.ylabel("Date")
    plt.xticks(np.arange(8, 24, 2))
    plt.savefig("pic.jpg")
    plt.show()
