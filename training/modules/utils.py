import numpy as np

def merge(lst):
    """
    Merge identical adjacent elements in a list.
    """

    # check if the list is empty
    if not lst:
        return [], []

    # start with the first element
    merged_list = [lst[0]]
    count_list = [1]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            count_list[-1] += 1  # increment count for the current sequence
        elif lst[i] != lst[i - 1]:
            merged_list.append(lst[i])
            count_list.append(1)  # reset count for the new element

    return merged_list, count_list