import sqlite3
import numpy as np
import pickle
import sys

def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    """
    Create progress bar for large task
    """
    percent = round(progress / float(total) * 100, 2)
    buf = "{0}|{1}| {2}{3}/{4} {5}%".format(lbar_prefix, ('#' * round(percent)).ljust(100, '-'),
        rbar_prefix, progress, total, percent)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()


def print_data():
    """
    Print the data (top lines) in the .txt file
    """
    with open("./data/mxm/mxm_dataset_train.txt", 'r') as file:
        for i, line in enumerate(file.readlines()):
            print(line)
            if i >= 50:
                break


if __name__=='__main__':
    train_file = "../../data/mxm/mxm_dataset_train.txt"

    # The dictionary that stores BOW ["MSD_ID": np.array(BagOfWords)]
    bow = dict()
    tracks = dict()

    # Get the blank vector for each bag of words vector
    # For coding convenience, we do it seperately from loading data
    vector = None
    with open(train_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == '%':
                words = line.strip()[1:].split(',')
                vector = np.zeros(len(words), dtype=int)

    # Load data from training file. Create bag of words vector for each MSD_ID.
    # Assumption: No MSD_ID duplications
    with open(train_file, 'r') as file:
        lines = file.readlines()
        total = len(lines)
        # Use enumerate for getting both index and value for the progress bar
        count = 0
        for i, line in enumerate(lines):
            # Create vector copy, make sure original blank vector not changed
            vec = vector.copy()
            # Ignore comments, blanks and the line of words
            if line == '' or line.strip() == '':
                continue
            if line[0] in ('#', '%'):
                continue
            line_parts = line.strip().split(',')
            # Get IDs
            MSD_ID = line_parts[0]
            MXM_ID = line_parts[1]
            # Create BOW vector for each MSD_ID
            for word_cnt in line_parts[2:]:
                word_id, cnt = word_cnt.split(':')
                # Use word_id-1 due to index displacement
                vec[int(word_id)-1] = int(cnt)
            bow[MSD_ID] = vec
            tracks[count] = MSD_ID
            # Report Progress Bar
            count += 1
            report_progress(i, total)

    # Dump the BOW dict into Pickle
    # with open('lyrics.pickle', 'wb') as file:
    #     pickle.dump(bow, file)

    with open('../../data/mxm/match.pickle', 'wb') as file:
        pickle.dump(tracks, file)

    # Test Loading
    # with open('lyrics.pickle', 'rb') as file:
    #     bow = pickle.load(file)
    #     print("BOW vectors #: {}".format(len(bow))) # 210519
