## What?

This repository has examples for calculating matrices infinity norms with OpenMP-based threads.

PLEASE NOTE: This code is only for educational purposes and can only operate on random generated matrices.

## How to use

If you are on linux, you can just go and use cmake, however, on mac you'd need to run gcc by hands:

```shell script
.build$ gcc-9 -Wall -fopenmp ../matrix_norm_omp.c -o matrix_norm_omp
``` 

## Running

```
.build$ ./matrix_norm_omp 1600 2
                            |  |
                            |  └-- Number of partitions(threads)
                            └----- Square matrix size
1600;2;5852720
 |   |   |
 |   |   └-- Time spent in calculation only(microseconds)
 |   └------ Number of threads
 └---------- Size of the matrix
```