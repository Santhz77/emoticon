import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
# x = np.loadtxt('result.csv',delimiter=',')
#
# print(x)

def plot_the_simple_data(x, y, hr_x, hr_y):
    plt.plot_date(dates, x)
    plt.plot(x,y)
    plt.plot(hr_x, hr_y)
    plt.title('depth variation')
    plt.xlabel('Time elasped')
    plt.ylabel('depth')

    plt.show()


def plot_the_data(x,y, hr_x,hr_y):

    # plt.plot_date(dates, x)
    # plt.plot(x,y)
    # plt.plot(hr_x, hr_y)
    # plt.title('depth variation')
    # plt.xlabel('Time elasped')
    # plt.ylabel('depth')
    #
    # plt.show()


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('heart rate', color=color)
    ax1.plot(hr_x, hr_y, color=color, alpha=0.4)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('depth of the pit', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


with open('result_20_19.csv', 'r', encoding='utf=8') as f:
    reader = csv.reader(f)
    data = []

    for val in reader:
        data.append(val)

    hr_time = data[0]
    hr = data[1]
    time_hr_x = [] # this represnts the experiment time that has elasped.


    for tx in hr_time:
        time_hr_x.append(int(tx.replace('\'', '')))



    time_depth_change = data[2]
    depth_val = data[3]
    new_x = []
    for time_x in time_depth_change:
        new_x.append(int(time_x.replace('\'', '')))

    depth = []

    print(len(depth_val))
    print(len(time_hr_x))



    plot_the_data(new_x,depth_val,time_hr_x , hr)



    # for i in range(0,len(new_x)):
    #     for j in range(0,len(time_hr_x)):
    #         if(time_hr_x[j] > = new_x[i]):
    #             print(time_hr_x[i])
    #         else:
    #             depth.append(0)
    # print(depth)


    # for i in range(0,len(time_hr_x)):
    #     flag = False
    #     for j in range(0,len(new_x)):
    #         if(time_hr_x[i] >= new_x[j]):
    #             print(depth_val[j])