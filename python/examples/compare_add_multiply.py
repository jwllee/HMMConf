import numpy as np
import time


if __name__ == '__main__':
    n = 1000000
    nums = np.random.rand(n)
    nums = np.concatenate((nums, np.random.rand(n)))

    start = time.time()
    sum_ = nums.sum()
    end = time.time()

    print('Summing to {:.2f} took {:.2f}s'.format(sum_, end - start))

    start = time.time()
    multiply = nums.prod()
    end = time.time()

    print('Multiplication to {:.2f} took {:.2f}s'.format(multiply, end - start))
