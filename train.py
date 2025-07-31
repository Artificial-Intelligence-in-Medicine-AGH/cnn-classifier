import argparse


from TrainingManager import TrainingManager


checkpoint_name = None


def parse_run_arguments() -> None:
    global checkpoint_name

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_checkpoint', action="store", default=None)
    args = parser.parse_args()

    checkpoint_name = args.load_checkpoint  


def main():
    trainer = TrainingManager(checkpoint_name=checkpoint_name)
    trainer.training_loop()


if __name__ == "__main__":
    parse_run_arguments()
    main()
