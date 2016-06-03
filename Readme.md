# Parallel Latent Factor Model in Recommender Systems
## GPU Accelerated Matrix Factorization
### github (recommend to read this): https://github.com/lx9t01/cuda_SVD

### Project Description

The core idea of using matrix factorization in recommender System is to map user rating space to item space, via several implicit multi-dimensional singular value decomposition factor matrices to describe the relationship between user and items. Normally, the rating matrix between users and items is sparse, but we can train the decomposed matrix w.r.t the values we have, and predict the rating for those users who have not rated the item yet, therefore provide solid and favorable recommendation. 

### Program Details
This cuda accelerated matrix factorization project consists of a CPU control code and a GPU part of code. The GPU part has 3 kernels, they are responsible for parallelized 
* Matrix Multiplication
* Computing the RMS error between model prediction and true result
* Updating the P and Q matrix 
Data are streamed.

### Output
Time statistics of CPU and GPU based code, and two CSV files which has a predicted full matrix of rating from users to items. 

### Installation

The program relies on Eigen library installed for CPU part of the code, and GPU based cuda library should be installed in default folder as well. 

### Usage

After make the executable file, to run the program, simply type 
```
./cuda_SVD <path to training datafile>
```
to run the program in default settings, or type
```
./cuda_SVD <path to training datafile> <number of users> <number of items> <number of dimensions f>
```
to set the number of users and items manually. 

The training file provided is 
_u1.base_
this is a small portion of the netflix contest data file

_Parameters in the program_

| parameter        | value          | comments  |
| ------------- |:-------------:|:----- |
| BATCH_SIZE      | 1000 | defines batch size |
| PARSE_THROUGH      | 1      |   a toggle of whether use CPU to training iteratively (0) or one-time (1)|
|  () | 0.02      |  The threshold to determine when the result converges in iterative mode |
|  blocks | 64      |  number of blocks per grid |
|  threadsPerBlock | 64     |  number of threads per block |
|  num_users | 943      |  1-indexed number of users |
|  num_items | 1682      |  1-indexed number of items |
|  num_f | 30      |  dimension of latent factor |
|  num_f | 30      |  dimension of latent factor |
|  gamma | 0.001      |  step size, applies to both CPU and GPU code |
|  lamda | 0.0005      |  regularization term parameter, applies to both CPU and GPU code |



### History
*ORIGINAL METHOD: *
The project in the beginning is designed to handle medium number of data, and training on the data iteratively until the trained model is converged. 

However, this method applies to CPU code but not the GPU code. As the training goes on, we expect to shuffle the training data every time to mimic the random sampling of data. Doing so in GPU memory will cause unexpected mess and therefore The iterative training process only applies to CPU running. Copying the data back to CPU, shuffle and copy back will greatly undermine performance. 

The CPU iterative running results are as follows: 

num_users = 943;  num_items = 1682;  num_f = 30;

const float gamma = 0.01;  const float lamda = 0.005;

the last few deltas are listed below: 
delta_new: -9.74536e-05
stop condition: 0.216542
delta_new: 0.000672281
stop condition: -6.89847
delta_new: 5.57303e-06   (STOPPED here)

Total time to run classify on CPU: 10385.511719 (s)
After training, the training error RMS: 0.344782, 
which means in average, the trained model of SVD will geenrate 
a rating matrix that has a difference of 0.35 stars compared 
with existing data. 

It took Mako CPU almost 3 hours to train. 


*UPDATED METHOD: *
While iterative training could yield an accurate result in CPU demo, it’s not suitable for GPU acceleration for several reasons: 
* GPUs are not very good at shuffling the training data (at least I am unable to perform this in any kernel)
* In most real-world cases, data are in huge amount. Therefore, it makes sense for GPU or CPU training process to just parse through the data and finish the training, because they are likely to be converged in that case. 
* I expand the training data by simply copying the existing data several more times and concat them in the bottom of file, and it helps the CPU and GPU code to converge. 

### Results
Note: 
* because GPU training is updated inside the kernel, I did not print out RMS error every time the P and Q are updated, but just print out them in the end. They are similar to CPU RMS error. 
* Because the training is not complete, the cvs files containing the rating data are not accurate enough, but can only provide a trend of rating. To obtain real rating recommendation, we need more data to training more times. 

1. Original non-expanded training file: (100000 ratings)
Total time to run decomposition on CPU: 1.524624 (s)
Total time to run decomposition on GPU: 0.496859 (s)
RMS CPU: 3.69954
GPU RMS: 3.69957

2. Expanded training file: (560000 ratings)
Total time to run decomposition on CPU: 8.359917 (s)
Total time to run decomposition on GPU: 0.896845 (s)
RMS CPU: 1.28598
GPU RMS: 1.31158

As the training data got big, the submission attachment can only contain this number of data… 

### Performance Analysis
1. The GPU has significant time advantage over CPU (3x - 10x);
2. Because the training is not completed compared with iterative model, The RMS error is relatively big.
3. If the training data set got bigger, the accuracy will improve significantly.  

### Resources
1. Netflix Prize Data Set
http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a
2. Matrix Factorization Techniques For Recommender Systems
https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf



