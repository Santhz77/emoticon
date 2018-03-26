import numpy as np
import matplotlib.pyplot as plt
import csv
import re



filename = 'user_study_responses.csv'



def get_interested_data(file_reader):
    interested_data = []

    for data in reader:
        if (int(data['Are you afraid of heights ?']) > 3):
            interested_data.append(data)



def plot_age_distribution(reader):
    age_data = []
    for line in reader:
        age_data.append(int(line['Age :']))

    data = np.unique(age_data, return_counts=True)
    plt.bar(data[0], data[1])
    plt.title('Age group')
    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.show()


def plot_self_reorted_heights(reader):
    self_reported_fear = []
    tick_label = [ 'Not at all', 'Slightly' , 'Somewhat', 'Moderately', 'Quite a bit', 'Very much',  'An extreme amount']
    for line in reader:
        self_reported_fear.append(int(line['Are you afraid of heights ?']))

    data = np.unique(self_reported_fear, return_counts=True)

    plt.bar(data[0], data[1] , color='SkyBlue' ,  tick_label = tick_label)
    plt.title('Are you afraid of heights ?')
    plt.ylabel('Count')

    # plt.text()


    percent_of_goal =[]
    for i in range(0 , len(data[1])):
        percent_of_goal.append(round(data[1][i]/sum(data[1]) * 100 , 2))

    print(percent_of_goal)

    plt.show()


def plot_calm(reader,text ,show = False):
    x= []
    print(text)
    for line in reader:
        x.append(int(line[text]))
    print(x)

    tick_label = [ 'Not at all', 'Slightly' , 'Somewhat', 'Moderately', 'Quite a bit', 'Very much',  'An extreme amount']

    data = np.unique(x, return_counts=True)
    new_tick_label = []
    for i in range(0,len(data[0])):
        new_tick_label.append(tick_label[data[0][i] - 1])


    plt.figure(figsize=(10, 8), dpi=80)
    # Create a new subplot from a grid of 1x1
    plt.subplot(111)



    plt.bar(data[0], data[1], color='kgbrymc' , tick_label = new_tick_label )

    emotion  = re.search(r"\[([A-Za-z0-9_]+)\]", text)

    title =  str(emotion.group(0)) + ' | Color : '  +  str(emotion.group(1)) + '.'

    plt.title(title)
    plt.ylabel('count of people')

    plt.savefig('images/' + str(emotion.group(1)), dpi=72)

    if(show):
        plt.show()


def get_header(reader):
    header =[]
    # create header
    for line in reader:
        for head in line:
            header.append(head)
        break;
    return header

def get_source(reader, text):
    source = []
    print(text)
    for lines in reader:
        print(lines)

def get_file_contents():

    return reader

#interested_data = get_interested_data(reader)




with open(filename, 'r') as theFile:
    reader = csv.DictReader(theFile)

    my_data =[]
    for line in reader:
        my_data.append(line)

    # to plot the age
    # plot_age_distribution(reader)

    # plot_self_reorted_heights(reader)


    header = get_header(my_data)
    for head in header:
        k = str(head)
        if(k[0:5] == 'Which'):
            print(k)
            plot_calm(my_data, k)



    # header = get_header(my_data)
    # for head in header:
    #     k = str(head)
    #     if (k[0:5] == 'Scene'):
    #
    #         emotion = re.search(r"\[([A-Za-z0-9_]+)\]", k)
    #
    #         if(str(emotion.group(1)) == 'Terror'):
    #
    #             mean_val = []
    #             importnat = []
    #             for line in my_data:
    #                 mean_val.append(int(line[str(k)]))
    #                 if(int(line[str(k)]) >=4):
    #                     importnat.append(int(line[str(k)]))
    #
    #             print(k[0:7], str(emotion.group(1)) )
    #             # print(len(mean_val) , len(importnat))
    #             ratio = (len(importnat) / len(mean_val) )* 100
    #             # print(max(importnat,key=importnat.count))
    #             # print(np.mean(importnat))
    #             print(ratio)
    #             print('--------------------------')












