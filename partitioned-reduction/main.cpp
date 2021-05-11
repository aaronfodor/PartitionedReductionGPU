#include "cuda_functions.h"

#include <vector>
#include <iostream>


std::vector<float> create_random_array(int arraySize){
    srand(0);

    std::vector<float> array;
    for (int i = 0; i < arraySize; i++){
        array.push_back(rand());
        // for testing, to easily debug what is the maximum value
        //array.push_back(float(i));
    }

    std::cout << "Generated array: ";
    for (int i = 0; i < arraySize; i++){
        std::cout << array[i] << ",";
    }
    std::cout << std::endl;

    return array;
}

float task_partitioned_reduction(std::vector<float> input_array){
    float maxValue = CUDA_FUNCTIONS::partitioned_reduction(input_array.data(), input_array.size());
    return maxValue;
}

int main(int argc, char **argv){

    int arraySize = 0;

    if(argc < 2){
        std::cout << "Input argument not found. Please type the required array size (int):" << std::endl;
        std::cin >> arraySize;
    }
    else{
        char* numChars = argv[1];
        arraySize = atoi(numChars);
    }

    if(arraySize < 1){
        std::cout << "Required array size must be greater than 0." << std::endl;
        return 0;
    }

    std::cout << "Required array size: " << arraySize << std::endl;

    std::vector<float> input_array = create_random_array(arraySize);
    float maxValue = task_partitioned_reduction(input_array);
    std::cout << "Max value: " << maxValue << std::endl;
    return 0;
}
