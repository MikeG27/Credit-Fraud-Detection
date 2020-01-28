import os


def make_dataset(bash_command):
    os.system(bash_command)


if __name__ == '__main__':
    bash_command = "kaggle datasets download --unzip creditcardfraud --path ../data/raw "
    make_dataset(bash_command)
