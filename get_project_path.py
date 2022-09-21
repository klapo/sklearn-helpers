import os


def project_paths():
    dir_helpers = os.path.dirname(os.path.realpath(__file__))
    dir_project = os.path.abspath(os.path.join(dir_helpers, '..'))
    dir_data = os.path.abspath(os.path.join(dir_project, 'data'))
        
    return dir_helpers, dir_project, dir_data