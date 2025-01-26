#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <type_traits>

const int NUM_LAYERS = 5;
const int HIDDEN_LAYER_SIZE = 256;
const int INPUT_SIZE = 256;

template<typename Func>
void parallel_for(int start, int end, int num_threads, Func func) {
    std::vector<std::thread> threads;
    int total = end - start;
    int chunk_size = (total + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int thread_start = start + i * chunk_size;
        int thread_end = std::min(end, thread_start + chunk_size);
        if (thread_start >= thread_end) break;
        threads.emplace_back([=, &func] {  // Keep lambda capture for C++14
            for (int j = thread_start; j < thread_end; ++j) {
                func(j);
            }
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

long double sigmoid(long double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

long double randomNumberInRange(long double min, long double max) {
    thread_local std::mt19937 generator([]{
        std::random_device rd;
        return rd();
    }());
    std::uniform_real_distribution<long double> distribution(min, max);
    return distribution(generator);
}

class NeuralNetwork {
private:
    struct Layer {
        std::vector<std::vector<long double>> weights;
        std::vector<long double> biases;
    };

    std::vector<Layer> layers;
    int input_size;
    int num_threads;

public:
    NeuralNetwork(int input_size, const std::vector<int>& layer_sizes)
        : input_size(input_size), 
          num_threads(std::thread::hardware_concurrency()) {
        int prev_size = input_size;
        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            int current_size = layer_sizes[i];
            Layer layer;
            
            // Initialize weights
            layer.weights.resize(current_size, std::vector<long double>(prev_size));
            parallel_for(0, current_size, num_threads, [&](int j) {
                for (int k = 0; k < prev_size; ++k) {
                    layer.weights[j][k] = randomNumberInRange(-0.5, 0.5);
                }
            });
            
            // Initialize biases
            layer.biases.resize(current_size);
            parallel_for(0, current_size, num_threads, [&](int j) {
                layer.biases[j] = randomNumberInRange(-0.5, 0.5);
            });
            
            layers.push_back(layer);
            prev_size = current_size;
        }
    }

    std::vector<long double> forward(const std::vector<long double>& input) {
        std::vector<long double> activations = input;
        
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            std::vector<long double> new_activations(layer.biases.size());
            
            parallel_for(0, layer.biases.size(), num_threads, [&](int j) {
                long double sum = 0.0;
                for (size_t k = 0; k < activations.size(); ++k) {
                    sum += activations[k] * layer.weights[j][k];
                }
                sum += layer.biases[j];
                new_activations[j] = sigmoid(sum);
            });
            
            activations = std::move(new_activations);
        }
        
        return activations;
    }
};

int main() {
    std::vector<int> layer_sizes(NUM_LAYERS, HIDDEN_LAYER_SIZE);
    NeuralNetwork network(INPUT_SIZE, layer_sizes);

    std::vector<long double> input(INPUT_SIZE);
    const int num_threads = std::thread::hardware_concurrency();
    parallel_for(0, INPUT_SIZE, num_threads, [&](int i) {
        input[i] = randomNumberInRange(-0.5, 0.5);
    });

    std::vector<long double> output = network.forward(input);
    return 0;
}