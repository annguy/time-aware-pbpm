"""
This script allows the users to build dictionaries to encode and decode the activities.
This script also forms divisors that are important to predict an average right time for certain time steps.

The implementation is inspired by the works of Tax, Niek, Ilya Verenich, Marcello La Rosa, and Marlon Dumas.
“Predictive Business Process Monitoring with LSTM Neural Networks.” In Advanced Information Systems Engineering. 2017

"""

# Importing Libraries
from collections import Counter
import numpy as np
import copy
import time
from datetime import datetime


class Preprocess:
    """
    Preprocess Class

    ...

    This class takes numpy arrays as input and returns Dictionary Maps of ActivitiesID and with ASCII coded string.
    Also it forms time divisors which are important to predict an average right time over the entire data set.
    The implementation is inspired by the works of Tax, Niek, Ilya Verenich, Marcello La Rosa, and Marlon Dumas.
    “Predictive Business Process Monitoring with LSTM Neural Networks.” In Advanced Information Systems Engineering. 2017


    Attributes
    --------------

        divisor: float
            average time between current and first events
        divisor2: float
            average time between current and first events
        divisor3 :float
            remaining time divisor
        lines: list
            these are all the activity seq
        timeseqs: list
            time sequences (differences between two events)
        timeseqs2:list
            time sequences (differences between the current and first)
        numlines:int
            Count of the number of cases
        ascii_offset: int
            Offset ascii value to encode to string
        timeseqs3: list
            Midnight time training_set
        timeseqs3: list
            midnight time test_set
        timeseqs4: list
            day of the week training_set
        fold3: list
            ActivityIDs of TestSet
        fold3_c: list
            CaseIds of TestSet
        fold3_t: list
            differences between two events test_set
        fold3_t2: list
            differences between the current and first of test_set
        fold3_t3: list
            Midnight time test_set
        fold3_t4: list
            day of the week test_set
        char_indices : dict
            ascii coded characters of the unique activities to integer indices
        indices_char: dict
            integer indices to ascii coded characters of the unique activities
        target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
        target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities
        lines: list
             ActivityIDs
        lines_t: list
            differences between two events
        lines_t2: list
            differences between the current and first of test_set
        lines_t3 : list
            Midnight time
        lines_t4 : list
            Day of the week



   Methods

    ---------------
    divisor_cal(self, spamreader):
        Takes a numpy array as input and returns the time divisors
    dict_cal(self,):
       This function forms dictionary values only on the basis of only the Training set i.e. first 2/3rd of the Data.
    training_set():
        Creates the sequences structure of data (Timesteps X Actvities X Features) for modelling purposes from event log.
        Also genertes the label sequences both from first 2/3rd of the data.
    test_set():
        Returns the last 1/3rd of the data as test_set

    """

    def __init__(self, divisor=0,divisor2=0,
                 divisor3=0,caseids=[],lines=[],
                 timeseqs=[],timeseqs2=[],timeseqs3=[],
                 timeseqs3_test=[],timeseqs4=[],numlines=0,
                 ascii_offset=161,maxlen=0,fold3=0,
                 fold3_c='',fold3_t=0,fold3_t2=0,fold3_t3=0,fold3_t4=0):


        """""
        Parameters
        ------------------        
        divisor: float
            average time between current and first events
        divisor2: float
            average time between current and first events
        divisor3 :float
            remaining time divisor
        lines: list
            these are all the activity seq
        timeseqs: list
            time sequences (differences between two events)
        timeseqs2:list
            time sequences (differences between the current and first)
        numlines:int
            Count of the number of cases
        ascii_offset: int
            Offset ascii value to encode to string
        timeseqs3: list
            Midnight time training_set
        timeseqs3: list
            midnight time test_set
        timeseqs4: list
            day of the week training_set
        fold3: list
            ActivityIDs of TestSet
        fold3_c: list
            CaseIds of TestSet
        fold3_t: list
            differences between two events test_set
        fold3_t2: list
            differences between the current and first of test_set
        fold3_t3: list
            Midnight time test_set
        fold3_t4: list
            day of the week test_set
        """""
        self.divisor = divisor
        self.divisor2 = divisor2
        self.divisor3 = divisor3
        self.caseids = caseids
        self.lines = lines  # these are all the activity seq
        self.timeseqs = timeseqs  # time sequences (differences between two events)
        self.timeseqs2 = timeseqs2  # time sequences (differences between the current and first)
        self.timeseqs3 = timeseqs3
        self.timeseqs3_test = timeseqs3_test
        self.timeseqs4 = timeseqs4
        self.numlines = numlines
        self.ascii_offset = ascii_offset
        self.maxlen = maxlen

        # Test Set
        self.fold3 = fold3
        self.fold3_c = fold3_c
        self.fold3_t = fold3_t
        self.fold3_t2 = fold3_t2
        self.fold3_t3 = fold3_t3
        self.fold3_t4 = fold3_t4

    # Calculates the time divisors
    def divisor_cal(self, spamreader):

        """Takes a numpy array as input and returns the time divisors

        Parameters

        --------------
        spamreader: obj
            Structured eventlog as like a numpy array obj.

        *****Helper Variables******
        line: char
            helps to store each encoded activity after iteration
        times: list
            helps to store each time difference between two activities after iteration
        times2: list
            helps to store time difference between starting and current activity after each iteration
        casestarttime: float
            helper variable
        lasteventtime: float
            helper variable
        lastcase: char
            helper variable

        Returns
        ---------------
        self.divisor: float
            average time between current and first events
        self.divisor2: float
            average time between current and first events

        """
        # helper variables
        line = ''
        firstLine = True
        times = []
        times2 = []
        casestarttime = None
        lasteventtime = None
        lastcase = ''
        self.spamreader = spamreader

        for row in spamreader:  # the rows are "CaseID,ActivityID,CompleteTimestamp"
            t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # creates a datetime object from row[2]
            if row[0] != lastcase:  # 'lastcase' is to save the last executed case for the loop
                self.caseids.append(row[0])
                casestarttime = t
                lasteventtime = t
                lastcase = row[0]
                if not firstLine:
                    self.lines.append(line)
                    self.timeseqs.append(times)
                    self.timeseqs2.append(times2)
                    self.timeseqs3.append(times3)
                    self.timeseqs3_test.append(times3_test)
                    self.timeseqs4.append(times4)
                line = ''
                times = []
                times2 = []
                times3 = []
                times4 = []
                times3_test = []
                self.numlines += 1
            line += chr(int(row[1]) + self.ascii_offset)  # AsciiEncoding for the Events
            timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(lasteventtime))
            timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(casestarttime))
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
            timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
            timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
            timediff3 = timesincemidnight.seconds  # this leaves only time even occured after midnight
            timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()  # day of the week
            times.append(timediff)
            times2.append(timediff2)
            times3.append(timediff3)
            times3_test.append(datetime.fromtimestamp(time.mktime(t)))
            times4.append(timediff4)
            lasteventtime = t
            firstLine = False

        # add last case
        self.lines.append(line)
        self.timeseqs.append(times)
        self.timeseqs2.append(times2)
        self.timeseqs3.append(times3)
        self.timeseqs3_test.append(times3_test)
        self.timeseqs4.append(times4)
        self.numlines += 1

        ########################################
        self.divisor = np.mean([item for sublist in self.timeseqs for item in sublist])  # average time between events
        print('divisor: {}'.format(self.divisor))
        self.divisor2 = np.mean(
            [item for sublist in self.timeseqs2 for item in sublist])  # average time between current and first events
        print('divisor2: {}'.format(self.divisor2))
        self.divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x) - 1] - y, x))),
                                         self.timeseqs2)))  # time average for remaining time
        print('divisor3: {}'.format(self.divisor3))

        return self.divisor, self.divisor2, self.divisor3

    def dict_cal(self, ):
        """This function forms dictionary values only on the basis of only the Training set i.e. first 2/3rd of the Data.



        ****Helper Variables*****
        elems_per_fold: int
            one third of number of cases.


        Training Set(2/3):
        fold1: list
            Activity list of first one third of cases.
        fold1_t: list
            Time differences between two events of first one third of cases.
        fold1_t2: list
            Time differences between first and current activity of first one third of cases.

        fold2: list
            Activity list of second one third of cases.
        fold2_t: list
            Time differences between two events of second one third of cases.
        fold2_t2: list
            Time differences between first and current activity of second one third of cases.


        Test Set(1/3):
        self.fold3_t: list
            differences between two events test_set
        self.fold3_t2: list
            differences between the current and first of test_set
        self.fold3_t3: list
            Midnight time test_set
        self.fold3_t4: list
            day of the week test_set
         self.maxlen
            maximum length of cases

        Returns
        ---------------
        self.char_indices : dict
            ascii coded characters of the unique activities to integer indices
        self.indices_char: dict
            integer indices to ascii coded characters of the unique activities
        self.target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
        self.target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities
        self.maxlen
            maximum length of cases
        self.chars:list
            ascii coded characters of the activities.
        self.target_chars: list
            ascii coded characters of the target activities.
           (target includes one excess activity '!' case end)

        """

        # separate Data into 3 parts
        elems_per_fold = int(round(self.numlines / 3))
        fold1 = self.lines[:elems_per_fold]  # Events
        fold1_t = self.timeseqs[:elems_per_fold]  # differences between two events
        fold1_t2 = self.timeseqs2[:elems_per_fold]  # differences between the current and first
        fold1_t3 = self.timeseqs3[:elems_per_fold]
        fold1_t4 = self.timeseqs4[:elems_per_fold]

        fold2 = self.lines[elems_per_fold:2 * elems_per_fold]
        fold2_t = self.timeseqs[elems_per_fold:2 * elems_per_fold]
        fold2_t2 = self.timeseqs2[elems_per_fold:2 * elems_per_fold]
        fold2_t3 = self.timeseqs3[elems_per_fold:2 * elems_per_fold]
        fold2_t4 = self.timeseqs4[elems_per_fold:2 * elems_per_fold]

        self.fold3 = self.lines[2 * elems_per_fold:]
        self.fold3_c = self.caseids[2 * elems_per_fold:]
        self.fold3_t = self.timeseqs[2 * elems_per_fold:]
        self.fold3_t2 = self.timeseqs2[2 * elems_per_fold:]
        self.fold3_t3 = self.timeseqs3_test[2 * elems_per_fold:]

        # leave away fold3 for now
        self.lines = fold1 + fold2
        self.lines_t = fold1_t + fold2_t
        self.lines_t2 = fold1_t2 + fold2_t2
        self.lines_t3 = fold1_t3 + fold2_t3
        self.lines_t4 = fold1_t4 + fold2_t4

        self.lines = list(map(lambda x: x + '!', self.lines))  # put delimiter symbol

        self.maxlen = max(map(lambda x: len(x), self.lines))  # find maximum line size
        print('maxlen', self.maxlen)

        # next lines here to get all possible characters for events and annotate them with numbers
        self.chars = map(lambda x: set(x), self.lines)
        self.chars = list(set().union(*self.chars))
        self.chars.sort()
        self.target_chars = copy.copy(self.chars)
        print(self.target_chars)
        self.chars.remove('!')
        print('total chars: {}, target chars: {}'.format(len(self.chars), len(self.target_chars)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars, start=1))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars, start=1))
        self.target_char_indices = dict((c, i) for i, c in enumerate(self.target_chars, start=1))
        self.target_indices_char = dict((i, c) for i, c in enumerate(self.target_chars, start=1))
        print('charecter_indices', self.char_indices)

        return self.maxlen, self.chars, self.target_chars, self.char_indices, self.indices_char, self.target_char_indices, self.target_indices_char

    def training_set(self, num_features, step=1):
        """This function generates training_set data an labels.
        The dimension of the data is Timesteps x Actvities x Features

        Parameters
        -------------
        num_features: int
            number of features
        step: int
            number of activities appended at each prefix step

        ******Helper Variables*****
        softness: int
            tuning parameter(zero softness by default)
        sentences :list
            list of ActivityIDs
        next_chars : list
            list of next ActivityIDs
        sentences_t : list
            time differences between two events
        sentences_t2 : list
            time difference between current and first event
        sentences_t3 : list
            midnight time of the current activity
        sentences_t4 : list
            day of the week
        next_chars_t : list
            time differences between next two events
        next_chars_t2 :list
            time difference between next and first event
        next_chars_t3 : list
        next_chars_t4 :list
        lines: char
            ActivityID
        lines_t: datetime
            time differences between two events
        lines_t2: datetime
            time differences between the current and first event
        lines_t3 : datetime
            Midnight time
        lines_t4 : int
            Day of the week
        self.lines: list
             ActivityIDs
        self.lines_t: list
            list of differences between two events
        self.lines_t2: list
            list of differences between the current and first event of training_set
        self.lines_t3 : list
            list midnight time
        self.lines_t4 : list
            lsit of day of the week


        Returns
        -----------
        X: numpy.ndarray
           training_set data left padded with zeros (samples, time_steps, features)
        y_a: numpy.ndarray
            training_set labels for next activity one-hot encoded (samples, unique_activities)
        y_t: numpy.ndarray
            training_set labels for time until next event scaled by avg time between events (samples, )
        dts: numpy.ndarray
            time deltas for each sample left padded not scaled in seconds (samples, time_steps)


        """

        softness = 0
        sentences = []
        next_chars = []
        sentences_t = []
        sentences_t2 = []
        sentences_t3 = []
        sentences_t4 = []
        next_chars_t = []
        next_chars_t2 = []
        next_chars_t3 = []
        next_chars_t4 = []
        for line, line_t, line_t2, line_t3, line_t4 in zip(self.lines, self.lines_t, self.lines_t2, self.lines_t3,
                                                           self.lines_t4):
            for i in range(0, len(line), step):
                if i == 0:
                    continue

                # we add iteratively, first symbol of the line, then two first, three...

                sentences.append(line[0: i])
                sentences_t.append(line_t[0:i])
                sentences_t2.append(line_t2[0:i])
                sentences_t3.append(line_t3[0:i])
                sentences_t4.append(line_t4[0:i])
                next_chars.append(line[i])
                if i == len(line) - 1:  # special case to deal time of end character
                    next_chars_t.append(0)
                    next_chars_t2.append(0)
                    next_chars_t3.append(0)
                    next_chars_t4.append(0)
                else:
                    next_chars_t.append(line_t[i])
                    next_chars_t2.append(line_t2[i])
                    next_chars_t3.append(line_t3[i])
                    next_chars_t4.append(line_t4[i])
        print('nb sequences:', len(sentences))
        num_features=len(self.chars)+5
        print('num features: {}'.format(num_features))
        X = np.zeros((len(sentences), self.maxlen, num_features), dtype=np.float32)  # Model input is X
        dts = np.zeros((len(sentences), self.maxlen), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(self.target_chars)),
                       dtype=np.float32)  # labels for the activity-->One hot Encoded
        y_t = np.zeros((len(sentences)), dtype=np.float32)  # labels for the time

        for i, sentence in enumerate(sentences):
            leftpad = self.maxlen - len(sentence)
            next_t = next_chars_t[i]
            sentence_t = sentences_t[i]
            sentence_t2 = sentences_t2[i]
            sentence_t3 = sentences_t3[i]
            sentence_t4 = sentences_t4[i]

            for t, char in enumerate(sentence):
                multiset_abstraction = Counter(sentence[:t + 1])
                # print(multiset_abstraction)
                for c in self.chars:
                    if c == char:  # this will encode present events to the right places
                        X[i, t + leftpad, self.char_indices[c]] = 1
                X[i, t + leftpad, len(self.chars)] = t + 1
                X[i, t + leftpad, len(self.chars)+1] = sentence_t[t] / self.divisor
                X[i, t + leftpad, len(self.chars)+2] = sentence_t2[t] / self.divisor2
                X[i, t + leftpad, len(self.chars)+3] = sentence_t3[t] / 86400
                X[i, t + leftpad, len(self.chars)+4] = sentence_t4[t] / 7
                dts[i,t+leftpad] = sentence_t[t]
            for c in self.target_chars:
                # print(i)
                if c == next_chars[i]:
                    y_a[i, self.target_char_indices[c]-1] = 1 - softness
                else:
                    y_a[i, self.target_char_indices[c]-1] = softness / (len(self.target_chars)-1)

            y_t[i] = next_t / self.divisor


        return X, y_a, y_t, dts

    def test_set(self,):
        """Returns the test_set


        *****Helper Variable****
        self.fold3_t: list
            differences between two events test_set
        self.fold3_t2: list
            differences between the current and first of test_set
        self.fold3_t3: list
            Midnight time test_set
        self.fold3_t4: list
            day of the week test_set

        Returns
        -------------
        self.lines: list
            these are all the activity seq
        self.caseids: list
            caseid of the test sets
        self.lines_t:list
            time sequences (differences between the current and first)
        self.lines_t2: list
            time from case start
        self.lines_t3: list
            time features for midnight time


        """

        self.lines = self.fold3
        self.caseids = self.fold3_c
        self.lines_t = self.fold3_t
        self.lines_t2 = self.fold3_t2
        self.lines_t3 = self.fold3_t3

        return self.lines, self.caseids, self.lines_t, self.lines_t2, self.lines_t3
