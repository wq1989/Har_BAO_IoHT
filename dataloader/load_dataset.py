# load dataset
import os

from numpy import dstack
from numpy import vstack
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_WISDM_signals(config, data_dir):

    # data_dict = loadmat(file)
    all_data_list = []
    all_labels = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            
            data_dict = loadmat(file_path)
            data = data_dict['F'][0] 
            X_data = np.zeros((data.shape[0], data[0].shape[0], data[0].shape[1]))
            
            for i in range(data.shape[0]):
                var_count = data[i].shape[-1]
                #print(i, X_train_mat[i])
                X_data[i, :, :var_count] = data[i]
                
            lbl_list = [os.path.splitext(filename)[0]] * data.shape[0]
            
            all_data_list.append(X_data)
            all_labels.extend(lbl_list)
            
            config.LOGGER_FILE.info(f"Loaded {lbl_list[0]} with size {X_data.shape}")
            # print(f"Loaded {lbl_list[0]} with size {X_data.shape}")
            
    all_data = np.vstack(all_data_list)

    
    
    return all_data, all_labels


def load_WISDM(config, dataset_path):
    
    data_dir1 = os.sep.join([dataset_path, 'acc_gravity'])
    data_dir2 = os.sep.join([dataset_path, 'acc_body'])
    
    all_data_grav, all_labels_grav = load_WISDM_signals(config, data_dir1)
    all_data_body, all_labels_body = load_WISDM_signals(config, data_dir2)

    concat_all_data = np.dstack((all_data_grav, all_data_body))
    # loaded label are identical, we will use one of them
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(all_labels_grav)
    config.label_encoder = le
    config.LOGGER_FILE.info(f'Labels encoded: {le.classes_}')
    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(concat_all_data, labels, test_size=0.3, shuffle = True, random_state=42)
    
    return X_train, y_train, X_test, y_test

class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)
    
# standardize data
def scale_data(X_train, X_test, X_val):
    # remove overlap: the data is split into windows of 128 time
    # steps, with a 50% overlap. To do it properly we must first
    # remove the duplicated before fitting the StandardScaler()
    cut = int(X_train.shape[1] / 2)
    longX = X_train[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flatTestX = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    flatValidationX = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
    # standardize
    s = StandardScaler()
    # s = MinMaxScaler()
    # fit on training data
    s.fit(longX)
    # apply to training, test and validation data
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    flatValidationX = s.transform(flatValidationX)
    # reshape
    flatTrainX = flatTrainX.reshape((X_train.shape))
    flatTestX = flatTestX.reshape((X_test.shape)) 
    flatValidationX = flatValidationX.reshape((X_val.shape))

    return flatTrainX, flatTestX, flatValidationX

# summarize the balance of classes in an output variable column
def class_breakdown(config, data):
	# convert the numpy array into a dataframe
	df = pd.DataFrame(data)
	# group data by the class value and calculate the number of rows
	counts = df.groupby(0).size()
	# retrieve raw rows
	counts = counts.values
	# summarize
	for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		if config.DATA_NAME == 'wisdm':
			config.LOGGER_FILE.info('Class=%s, total=%d, percentage=%.3f' % (config.label_encoder.classes_[i], counts[i], percent))
		else:
			config.LOGGER_FILE.info('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))
  
# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
 
# load a dataset group, such as train or test
def load_dataset(config, group, prefix='', verbose = 0):
	filepath = os.sep.join([prefix, group, 'Inertial Signals/'])
	# load all 9 files as a single array
	filenames = list()

	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
 
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(os.sep.join([prefix, group, 'y_'+group+'.txt'])) #prefix + group + '/y_'+group+'.txt')
 
	if verbose > 0:
		print(f'{group} Dataset')
		class_breakdown(config, y)
    
	return X, y

def prep_data(config, verbose=1, validation_split=None):
    # config_info['data_folder_raw'] + 'UCI HAR Dataset/'
    if config.DATA_NAME == 'UCI_HAR':
        X_train, Y_train = load_dataset(config, 'train', config.DATASET_DIR, verbose = 1)
        # load all test
        X_test, Y_test = load_dataset(config, 'test', config.DATASET_DIR, verbose = 1)
        # in UCI-HAR indexes of labels starts from 1  -> we need to start from 0
        #  or we will get false results
        Y_train, Y_test  = Y_train.ravel().astype(int) - 1, Y_test.ravel().astype(int) - 1
        
    elif config.DATA_NAME == 'wisdm':
        X_train, Y_train, X_test, Y_test  = load_WISDM(config, config.DATASET_DIR)
        config.LOGGER_FILE.info(f'Train: \n')
        class_breakdown(config, Y_train)
        config.LOGGER_FILE.info(f'Test: \n')
        class_breakdown(config, Y_test)
        
    if verbose > 1:
		# summarize combined class breakdown
        print('Both')
        combined = vstack((Y_train, Y_test))
        class_breakdown(config, combined)
        
    if config.REDUCE_DATA != None:
        X_train = X_train[:config.REDUCE_DATA]
        Y_train = Y_train[:config.REDUCE_DATA]

    [no_signals_train, no_steps_train, no_components_train] = np.shape(X_train)
    [no_signals_test, no_steps_test, no_components_test] = np.shape(X_test)

    if verbose > 0:
        config.LOGGER.info("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
        config.LOGGER.info("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test,no_steps_test,no_components_test))
        # print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
        # print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test,no_steps_test,no_components_test))
        
   # Validation
    if validation_split:
        index = int(X_train.shape[0] * validation_split)
        X_val = X_train[: index, :, :]
        Y_val = Y_train[: index]
        X_train = X_train[index:, :, :]
        Y_train = Y_train[index:]
        if verbose > 0:
            # print("The train data set has been splitted in a validation set ({} samples) and a train set ({} samples)".format(
            #         np.shape(Y_val)[0], np.shape(Y_train)[0]))
            config.LOGGER.info("The train data set has been splitted in a validation set ({} samples) and a train set ({} samples)".format(
                    np.shape(Y_val)[0], np.shape(Y_train)[0]))

        return X_train, Y_train, X_test, Y_test, X_val, Y_val


    return X_train, Y_train, X_test, Y_test 

def normalize_data(X_train, X_test, X_val):
    
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)
    X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
    X_val = (X_val - X_train_mean) / (X_train_std + 1e-8)
    return X_train, X_test, X_val

def load(config, batch_size=64):

    # load all train and test
    x_train, y_train, x_test, y_test, x_val, y_val = prep_data(config, verbose=1, validation_split=config.VAL_SPLIT)

    # Scale data
    x_train, x_test, x_val = scale_data(x_train, x_test, x_val)

    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    val_set = data_loader(x_val, y_val, transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader