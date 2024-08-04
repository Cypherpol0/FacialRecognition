import pathlib
import os
import sys
from torch.utils.data import DataLoader
import torch
import random
import multiprocessing

PACKAGE_PARENT = pathlib.Path.cwd().parent 
SCR_DIR = os.path.join(PACKAGE_PARENT, 'IMAGES_IDVPROJECT')
sys.path.append(SCR_DIR)

from config.loc_config import TRAIN_DATA_LOC, TEST_DATA_LOC, PRED_DATA_LOC, ANNOT_LOC, MODEL_SAVE_LOC, REPORT_SAVE_LOC
from config.data_config import INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL, BATCH_SIZE, NUM_WORKERS
from preprocessing.utils import build_annotation_dataframe, check_annot_dataframe, transform_bilinear
from model.dataset_model import CustomDataset, create_validation_dataset
import model.cnn_model as cnn_model
import model.modelling_config as modelling_config
import model.custom_loss_function as custom_loss_function
from postprocessing.utils import save_model_with_timestamp, save_csv_with_timestamp, calculate_model_performance

import importlib
import model.dataset_model
import model.cnn_model as cnn_model
import model.custom_loss_function as custom_loss_function
import config.loc_config
import postprocessing.utils
importlib.reload(model.dataset_model)
importlib.reload(cnn_model)
importlib.reload(modelling_config)
importlib.reload(custom_loss_function)
importlib.reload(config.loc_config)
importlib.reload(postprocessing.utils)
from model.dataset_model import CustomDataset, create_validation_dataset
import model.modelling_config as modelling_config
import config.loc_config
from postprocessing.utils import save_model_with_timestamp, save_csv_with_timestamp

print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())

if __name__ == '__main__':
    train_df = build_annotation_dataframe(image_location = TRAIN_DATA_LOC, annot_location = ANNOT_LOC, output_csv_name = 'train.csv')
    test_df = build_annotation_dataframe(image_location = TEST_DATA_LOC, annot_location = ANNOT_LOC, output_csv_name = 'test.csv')
    class_names = list(train_df['class_name'].unique())
    print(class_names)
    print(check_annot_dataframe(train_df))
    print(check_annot_dataframe(test_df))

    image_transform = transform_bilinear(INPUT_WIDTH, INPUT_HEIGHT)
    main_dataset = CustomDataset(annot_df = train_df, transform=image_transform)
    train_dataset, validation_dataset = create_validation_dataset(main_dataset, validation_proportion=0.2)
    print('Train set size: ', len(train_dataset))
    print('Validation set size: ', len(validation_dataset))

    test_dataset = CustomDataset(annot_df = test_df, transform=image_transform)
    print('Test set size: ', len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS)
    val_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

    # initiation
    model = cnn_model.MyCnnModel()
    device = modelling_config.get_default_device()
    modelling_config.model_prep_and_summary(model, device)
    criterion = modelling_config.default_loss()
    optimizer = modelling_config.default_optimizer(model = model)
    num_epochs = 10

    # get training results

    # Add the following line to ensure proper initialization of multiprocessing
    trained_model, train_result_dict = cnn_model.train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs)
    multiprocessing.freeze_support()
    #save_model_with_timestamp(trained_model, MODEL_SAVE_LOC)
    cnn_model.visualize_training(train_result_dict)
    save_model_with_timestamp(trained_model, MODEL_SAVE_LOC)

    save_csv_with_timestamp(train_result_dict, REPORT_SAVE_LOC)

    #testing
    trained_model_list = os.listdir(MODEL_SAVE_LOC)
    MODEL_10_EPOCH_PATH = os.path.join(MODEL_SAVE_LOC, trained_model_list[0])
    MODEL_10_EPOCH = cnn_model.MyCnnModel()
    device = modelling_config.get_default_device()
    print(MODEL_10_EPOCH_PATH)
    MODEL_10_EPOCH.load_state_dict(torch.load(MODEL_10_EPOCH_PATH))

    # check accuracy on test set
    y_pred, y_true = cnn_model.infer(model = MODEL_10_EPOCH, device = device, data_loader = test_loader)
    confusion_matrix, class_metrics, overall_metrics = calculate_model_performance(y_pred, y_true, class_names = class_names)
    print(confusion_matrix)
    print(class_metrics)
    print(overall_metrics)

    image_list = os.listdir(PRED_DATA_LOC)
    random_image = random.choice(image_list)
    random_image_path = os.path.join(PRED_DATA_LOC, random_image)
    print(random_image_path)

    predicted_class_index = cnn_model.infer_single_image(
        model=MODEL_10_EPOCH, 
        device=device, 
        image_path=random_image_path, 
        transform=image_transform)
    print(class_names[predicted_class_index])
'''
confusion_matrix

my_custom_loss = custom_loss_function.MyCustomLoss(device, confusion_matrix, 'fn')
my_custom_loss._fn_cost_matrix

my_custom_loss._fp_cost_matrix

# initiation
model = cnn_model.MyCnnModel()
device = cnn_model.get_default_device()
cnn_model.model_prep_and_summary(model, device)
criterion = custom_loss_function.MyCustomLoss(device, confusion_matrix, 'fn')
optimizer = cnn_model.default_optimizer(model = model)
num_epochs = 2

# get training results
trained_model, train_result_dict = cnn_model.train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs)
cnn_model.visualize_training(train_result_dict)

# saving model and report
save_model_with_timestamp(trained_model, MODEL_SAVE_LOC)
save_csv_with_timestamp(train_result_dict, REPORT_SAVE_LOC)'''