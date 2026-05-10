from utils.path_util import get_project_root,get_abs_path,DATA_DIR,STORAGE_DIR,LOGS_DIR,CONFIG_DIR


def test_path():
    print("项目根目录:", get_project_root())
    print("知识库目录:", DATA_DIR)
    # .....


if __name__ == '__main__':
    test_path()