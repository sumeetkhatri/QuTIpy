import random
import string
import timeit
from typing import Callable, List


def get_execution_time(func: Callable, repeat=5,
                       number=1000000) -> List[float]:
    """Returns the time take to execute a function

    Records the timimg of a function to benchmark performance in a pythonic way.

    Parameters
    ----------
    func : Callable
        Function to get performance timing for.

    Returns
    -------
    List[float]
        List of float values representing the seconds it took to run the function {number} times for each {repeat}
    """
    N = 10
    name = 'k'
    name += ''.join(random.choices(string.ascii_uppercase +
                    string.digits, k=N))

    global_variable = {**globals(), name: func}
    time = timeit.repeat(
        stmt=f'{name}()',
        globals=global_variable,
        repeat=repeat,
        number=number)
    return time
