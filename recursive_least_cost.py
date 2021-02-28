# Python program that finds the shortest path of a list of arrays
# via dynamic progrmming from:
# "Lecture Notes on Optimal Control: Optimization and system theory"

import numpy as np


# recursive function that find the minimum path using dynamic programming
# ind is index of the array to start from
# arr is the input list of arrays
# L is the length of the list of arrays
def find_min_cost_path(ind, arr, L):
    # return 0 and None with ind = L
    if ind == L-1:
        return np.zeros(arr[ind].shape), None

    # returns cost and path 
    ret_cost, ret_path = find_min_cost_path(ind+1,arr,L)

    # expand axis for broadcast
    ret_cost = ret_cost[np.newaxis, :]

    # compute cost values with broadcast and save in dummy variable 
    min_arr = np.add(np.square(np.subtract(arr[ind+1],arr[ind][:, np.newaxis])), ret_cost)
    min_cost = np.amin(min_arr, axis=1)

    # indexes positions in list j+1
    min_index = np.argmin(min_arr, axis=1)

    # try to return index of minimum cost paths
    arr_holder = []
    if ret_path is not None:
        arr_holder.append(min_index)
        arr_holder = arr_holder + ret_path
    else:
        arr_holder.append(min_index)

    min_path_index = arr_holder

    return min_cost, min_path_index

def main():

    ##### test 1 #####
    
    # create arrays
    a = np.asarray([1,2,3])
    b = np.asarray([8,7,6,5,4])
    c = np.asarray([12,11,10])

    # put array in a list as a numpy object array
    abc = [a, b, c]
    abc = np.asarray(abc, dtype='object')
    
    # get length of array of arrays 
    L = len(abc)

    # find mini cost
    total_min_cost, min_path_index = find_min_cost_path(0,abc,L)
    
    # get min cost
    min_cost = np.amin(total_min_cost)
    min_cost_index = np.argmin(total_min_cost)

    # get index of path and element of path
    path = []
    ext_arr = []
    path.append(min_cost_index)
    ext_arr.append(abc[0][path[-1]])
    for i in range(len(min_path_index)):
        path.append(min_path_index[i][path[-1]])
        ext_arr.append(abc[i+1][path[-1]])

    # print path with min cost
    print(ext_arr)

    ##### test 2 #####

    # create arrays
    a = np.asarray([1,2,3])
    b = np.asarray([4,5,6])
    c = np.asarray([7,8,9])
    d = np.asarray([10,11,12])
    e = np.asarray([13,14,15])

    # put array in a list as a numpy object array
    abcde = [a, b, c, d, e]
    abcde = np.asarray(abcde, dtype='object')

    # get length of array of arrays 
    L = len(abcde)

    # find min cost and mini index
    total_min_cost, min_path_index = find_min_cost_path(0,abcde,L)

    # get min cost
    min_cost = np.amin(total_min_cost)
    min_cost_index = np.argmin(total_min_cost)

    # get index of path and element of path
    path = []
    ext_arr = []
    path.append(min_cost_index)
    ext_arr.append(abcde[0][path[-1]])
    for i in range(len(min_path_index)):
        path.append(min_path_index[i][path[-1]])
        ext_arr.append(abcde[i+1][path[-1]])

    # print path with min cost
    print(ext_arr)

if __name__ == "__main__":
    main()
