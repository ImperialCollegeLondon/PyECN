from numpy import ndarray, loadtxt

def read_LUT(fname: str) -> ndarray:
    """Read a look-up table.

    Args:
        fname: the csv file containing the look-up table

    Returns:
        ndarray: the look-up table data
    """
    return loadtxt(fname, skiprows=1, delimiter=",")
