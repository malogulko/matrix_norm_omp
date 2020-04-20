//
// Created by Alex Malogulko on 13/01/2020.
//
//
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "utils.c"

struct rowSumPartitionReq {
    double *matrix_a;
    double *matrix_b;
    double *row_sum_vector;
    double *inf_norm;
    omp_lock_t *mutex_inf_norm;
    int size;
    int partition_num; // partition number (0..1,2,n)
    int partition_rows; // number of rows in the partition
};

void *ijk_row_sum_partition(void *input) {
    struct rowSumPartitionReq *info = (struct rowSumPartitionReq *) input;
    int partition_start = info->partition_num * info->partition_rows;
    int next_partition_start = (info->partition_num + 1) * info->partition_rows;
    double local_max_sum = 0;
    for (int i = partition_start; i < next_partition_start; i++) {
        double row_sum = 0;
        for (int j = 0; j < info->size; j++) {
            for (int k = 0; k < info->size; k++) {
                row_sum += *(info->matrix_a + i * info->size + k) * *(info->matrix_b + j * info->size + k);
            }
        }
        *(info->row_sum_vector + i) = row_sum;
        if (local_max_sum < row_sum) {
            local_max_sum = row_sum;
        }
    }
    omp_set_lock(info->mutex_inf_norm);
    if (local_max_sum > *info->inf_norm) {
        *info->inf_norm = local_max_sum;
    }
    omp_unset_lock(info->mutex_inf_norm);
    return NULL;
}

void ijk_row_sum_partitioned(double *matrix_a,
                             double *matrix_b,
                             double *row_sum_vector,
                             double *inf_norm,
                             int matrix_size,
                             int num_partitions) {
    check_partition(matrix_size, num_partitions);
    int partition_size = matrix_size / num_partitions;

    struct rowSumPartitionReq *reqs = malloc(num_partitions * sizeof(*reqs));

    omp_lock_t *mutex_inf_norm = malloc(sizeof(omp_lock_t));
    omp_init_lock(mutex_inf_norm);

    for (int p_num = 0; p_num < num_partitions; p_num++) {
        struct rowSumPartitionReq req = {
                .matrix_a = matrix_a,
                .matrix_b = matrix_b,
                .row_sum_vector = row_sum_vector,
                .inf_norm = inf_norm,
                .mutex_inf_norm = mutex_inf_norm,
                .size = matrix_size,
                .partition_num = p_num,
                .partition_rows = partition_size
        };
        *(reqs + sizeof(*reqs) * p_num) = req;
    }

    #pragma omp parallel for
    for (int p_num = 0; p_num < num_partitions; p_num++) {
        ijk_row_sum_partition(reqs + sizeof(*reqs) * p_num);
    }

    omp_destroy_lock(mutex_inf_norm);
    free(mutex_inf_norm);
}
/**
 * 4x4 matrix represented in memory as:
 *
 * 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
 *
 * Matrices A and C are stored in row-wise format:
 *
 * 01 02 03 04
 * 05 06 07 08
 * 09 10 11 12
 * 13 14 15 16
 *
 * At the same time, matrix B blocks stored in column-wise format:
 *
 * 01 05 09 13
 * 02 06 10 14
 * 03 07 11 15
 * 04 08 12 16
 *
 */
int main(int argc, char *argv[]) {
    int size, num_partitions;
    struct timespec start, end;
    // Uncomment this if you want the matrices to be actually random
    //srand(time(0));
    parse_args(argc, argv, &size, &num_partitions);
    double *matrix_a = matrix_malloc(size);
    double *matrix_b = matrix_malloc(size);
    double *inf_norm = malloc(sizeof(double));
    // Initialize info objs
    struct matrixInfo matrix_a_info = {.size = size, .mxPtr = matrix_a};
    struct matrixInfo matrix_b_info = {.size = size, .mxPtr = matrix_b};
    struct matrixInfo *matrices_to_randomize = malloc(2 * sizeof(*matrices_to_randomize));
    
    *matrices_to_randomize = matrix_a_info;
    *(matrices_to_randomize + sizeof(*matrices_to_randomize)) = matrix_b_info;

    #pragma omp parallel for
    for (int i = 0; i < 2; ++i) {
        random_matrix(matrices_to_randomize + i * sizeof(*matrices_to_randomize));
    }
    free(matrices_to_randomize);

    double *row_sum_vector = vector_malloc(size); // result vector with max

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    ijk_row_sum_partitioned(matrix_a, matrix_b, row_sum_vector, inf_norm, size, num_partitions);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000; // microseconds
    printf("%d;%d;%llu\n", size, num_partitions, delta_us);
    return 0;
}
