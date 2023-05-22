import random
import pandas as pd
import math
import random

def train_task():
    f = open('trials/cnceleb/fix_cnceleb_trials.txt', 'r')
    # f = open('trials/cnceleb/trials.lst', 'r')
    cnceleb_trails = f.read()
    f.close()

    trail_list = cnceleb_trails.split('\n')

    enrolls = []
    for index  in trail_list:
        if index.split(' ')[0] not in enrolls:
            enrolls.append(index.split(' ')[0])
    enrolls = enrolls[:-1]
    select_enroll = random.sample(enrolls, len(enrolls)//3)

    f = open('trials/cnceleb/train_task/cnceleb_split.txt', 'w')
    for trail in trail_list:
        if trail.split(' ')[0] in select_enroll:
            f.write(trail + '\n')
    f.close()

def meta_task():
    f = open('trials/cnceleb/fix_cnceleb_trials.txt', 'r')
    # f = open('trials/cnceleb/trials.lst', 'r')
    cnceleb_trails = f.read()
    f.close()

    dict = {}

    trail_list = cnceleb_trails.split('\n')
    trail_list = trail_list[:-1]
    for trail in trail_list:
        trail_type = trail.split(' ')[1].split('-')[1]
        if trail_type in dict.keys():
            dict[trail_type].append(trail)
        else:
            dict[trail_type] = [trail]

    for meta in dict.keys():
        f = open('trials/cnceleb/meta_task/cnceleb_' + meta + '.txt', 'w')
        for tra in dict[meta]:
            f.write(tra + '\n')
        f.close()

def file_list():
    f = open('trials/cnceleb/fix_cnceleb_trials.txt', 'r')
    # f = open('trials/cnceleb/trials.lst', 'r')
    cnceleb_trails = f.read()
    f.close()

    all_file = []

    trail_list = cnceleb_trails.split('\n')
    trail_list = trail_list[:-1]
    for trail in trail_list:
        enroll = trail.split(' ')[0].replace('wav', 'pkl')
        # enroll = 'enroll/' + trail.split(' ')[0] + '.pkl'
        test = trail.split(' ')[1].replace('wav', 'pkl')
        if enroll not in all_file:
            all_file.append(enroll)
        if test not in all_file:
            all_file.append(test)
    # all_file = list(set(all_file))
    
    f = open('trials/cnceleb/cnceleb_files.txt', 'w')
    for fi in all_file:
        f.write(fi + '\n')
    f.close()

def train_file_list():
    f = open('trials/cnceleb/train_task/cnceleb_split.txt', 'r')
    cnceleb_trails = f.read()
    f.close()

    all_file = []

    trail_list = cnceleb_trails.split('\n')
    for trail in trail_list:
        if trail == '':
            continue
        enroll = trail.split(' ')[0].replace('wav', 'pkl')
        # enroll = 'enroll/' + trail.split(' ')[0] + '.pkl'
        test = trail.split(' ')[1].replace('wav', 'pkl')
        if enroll not in all_file:
            all_file.append(enroll)
        if test not in all_file:
            all_file.append(test)
    # all_file = list(set(all_file))
    
    f = open('trials/cnceleb/train_task/split_cnceleb_files.txt', 'w')
    for fi in all_file:
        f.write(fi + '\n')
    f.close()

def fix_trial():
    f = open('trials/cnceleb/train_task/cnceleb_split.txt', 'r')
    cnceleb_trails = f.read()
    f.close()

    all_trials = []

    trail_list = cnceleb_trails.split('\n')
    for trail in trail_list:
        if trail == '':
            continue
        enroll = 'enroll/' + trail.split(' ')[0] + '.wav'
        test = trail.split(' ')[1]
        ground_truth = trail.split(' ')[2]
        all_trials.append(enroll + ' ' + test + ' ' + ground_truth + '\n')
    
    f = open('trials/cnceleb/fix_cnceleb_trials.txt', 'w')
    for fi in all_trials:
        f.write(fi)
    f.close()

def create_train_trial():
    cnceleb_DB = pd.read_json('filelists/Cnceleb/cnceleb_test.json')
    cnceleb_DB['speaker_id'] = cnceleb_DB['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    cnceleb_DB['temp'] = cnceleb_DB['filename'].apply(lambda x: x.split('/')[-2])
    enroll_DB = cnceleb_DB[cnceleb_DB['temp']=='enroll']
    test_DB = cnceleb_DB[cnceleb_DB['temp']=='test']
    speaker_list = list(set(test_DB['speaker_id']))

    spk_dict = {}
    for speaker in speaker_list:
        temp = test_DB[test_DB['speaker_id']==speaker]
        if list(set(temp['filename'])) != []:
            files = list(set(temp['filename']))
            files = random.sample(files, math.ceil(len(files)/4))
            spk_dict[speaker] = files

    trials = []

    for _, row in enroll_DB.iterrows():
        spk = row['speaker_id']
        enroll = row['filename']
        p_tests = spk_dict[spk]
        for p_test in p_tests:
            enroll_write = enroll.split('/')[-2] + '/' + enroll.split('/')[-1].replace('pkl', 'wav') + ' '
            test_write = p_test.split('/')[-2] + '/' + p_test.split('/')[-1].replace('pkl', 'wav') + ' '
            ground_truth = '1'
            trials.append(enroll_write + test_write + ground_truth)

        tests = []
        for temp_spk in spk_dict.keys():
            if temp_spk != spk:
                tests.extend(spk_dict[temp_spk])
            else:
                print('a')
        n_tests = random.sample(tests, math.ceil(len(tests)/25)-len(p_tests))

        for n_test in n_tests:
            enroll_write = enroll.split('/')[-2] + '/' + enroll.split('/')[-1].replace('pkl', 'wav') + ' '
            test_write = n_test.split('/')[-2] + '/' + n_test.split('/')[-1].replace('pkl', 'wav') + ' '
            ground_truth = '0'
            trials.append(enroll_write + test_write + ground_truth)

    f = open('trials/cnceleb/train_task/new_split_cnceleb_trails.txt', 'w')
    for fi in trials:
        f.write(fi + '\n')
    f.close()


def create_train_file_list():
    f = open('trials/cnceleb/train_task/new_split_cnceleb_trails.txt', 'r')
    cnceleb_trails = f.read()
    f.close()

    all_file = []

    trail_list = cnceleb_trails.split('\n')
    for trail in trail_list:
        if trail == '':
            continue
        enroll = trail.split(' ')[0].replace('wav', 'pkl')
        # enroll = 'enroll/' + trail.split(' ')[0] + '.pkl'
        test = trail.split(' ')[1].replace('wav', 'pkl')
        if enroll not in all_file:
            all_file.append(enroll)
        if test not in all_file:
            all_file.append(test)
    # all_file = list(set(all_file))
    
    f = open('trials/cnceleb/train_task/new_split_cnceleb_files.txt', 'w')
    for fi in all_file:
        f.write(fi + '\n')
    f.close()






# train_task()
# meta_task()
# file_list()
# train_file_list()
# fix_trial()
# create_train_trial()
# create_train_file_list()
