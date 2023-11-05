from trainer import Trainer

CSV_PATH = r"E:\\study\\bachelor\\temp-files\\data\\Obtain data\\dataframes\\3_min_training_map.csv"

CHECKPOINT_LOCATION = r"E:\study\bachelor\temp-files\data\models_weights\VGG16CustomHead_not_freezed_last_epoch.pt"
SAVE_CHECKPOINT_PATH = r"E:\study\bachelor\temp-files\data\models_weights"
MODEL_NAME = "VGG16CustomHead_not_freezed"

if __name__ == '__main__':

    trainer = Trainer(CSV_PATH)
    
    trainer.train(
        checkpoint_location=CHECKPOINT_LOCATION,
        save_checkpoint_path=SAVE_CHECKPOINT_PATH, 
        model_name=MODEL_NAME)