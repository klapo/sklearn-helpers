import os


def project_paths():
    """Helper function for using locally developed python
    functions in a jupyter notebook/conda environment without
    building a package. Also useful for pointing towards other
    common folders like "data".

    Returns
    ----------
    dir_helpers : os.path
        Path to this file.
    dir_project : os.path
        Path to the project (assumes this file is in a subdirectory)
    dir_data : os.path
        Path to data (assumed to be a subdirectory in the project
        folder labeled "data")
    """
    dir_helpers = os.path.dirname(os.path.realpath(__file__))
    dir_project = os.path.abspath(os.path.join(dir_helpers, '..'))
    dir_data = os.path.abspath(os.path.join(dir_project, 'data'))

    return dir_helpers, dir_project, dir_data
