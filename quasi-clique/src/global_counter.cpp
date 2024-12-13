#include "../include/global_counter.h"

std::atomic<int> iteration_count{0};
const int MAX_ITERATIONS = 1000000;

void resetIterationCount()
{
    iteration_count = 0;
}