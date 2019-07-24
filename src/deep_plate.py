import imresize
import tf_utils as utils
import tf_cnn as cnn

def main():
    generateResizedImages = False
    reshuffleImages = False

    if generateResizedImages:
        imresize.generate_resized_images(256)

    if reshuffleImages:
        sourceFolders = ['E:\\Projects\\Hach2019\\Data\\ImageDataSetS256\\Plate',
            'E:\\Projects\\Hach2019\\Data\\ImageDataSetS256\\NonePlate']
        h5Target = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetS256\\plate256.h5'
        dataset = utils.partition_dataset(sourceFolders, 0.8, 0.0, 0.2)
        utils.save_dataset_h5(target=h5Target, dataset=dataset)
        
        print("dataset shapes:")
        print("train_set_x: " + str(dataset["train_set_x"].shape))
        print("train_set_y: " + str(dataset["train_set_y"].shape))
        print("cv_set_x: " + str(dataset["cv_set_x"].shape))
        print("cv_set_y: " + str(dataset["cv_set_y"].shape))
        print("test_set_x: " + str(dataset["test_set_x"].shape))
        print("test_set_y: " + str(dataset["test_set_y"].shape))    
        print("classes: " + str(dataset["classes"].shape))

    h5Source = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetS256\\plate256.h5'
    tr_set_x, tr_set_y, cv_set_x, cv_set_y, ts_set_x, ts_set_y, classes = utils.load_dataset_h5(source=h5Source)
    
    print("loaded data shapes:")
    print("tr_set_x: " + str(tr_set_x.shape))
    print("tr_set_y: " + str(tr_set_y.shape))
    print("ts_set_x: " + str(ts_set_x.shape))
    print("ts_set_y: " + str(ts_set_y.shape))
    print("cv_set_x: " + str(cv_set_x.shape))
    print("cv_set_y: " + str(cv_set_y.shape))
    print("classes: " + str(classes.shape))


    X_train = tr_set_x/255.
    X_test = ts_set_x/255.
    Y_train = tr_set_y
    Y_test = ts_set_y
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}

    cnn.model(X_train, Y_train, X_test, Y_test)

# Main entry to program
if __name__ == '__main__':
    main()

