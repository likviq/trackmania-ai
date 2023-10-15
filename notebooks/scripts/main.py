from trainer import Trainer

CSV_PATH = r"E:\\study\\bachelor\\temp-files\\data\\Obtain data\\dataframes\\3_min_training_map.csv"

if __name__ == '__main__':

    trainer = Trainer(CSV_PATH)

    trainer.train()