#include "potts.h"

#include "npy.hpp" // https://github.com/llohse/libnpy

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <mutex>
#include <thread>

using namespace std;
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

// configuration
unsigned int q;
unsigned int L;
float T;
unsigned int batch_size;

// shared state
mutex todo_lock;
mutex samples_lock;
vector<potts_t> samples;
int n_todo;

void work()
{
    while (true)
    {
        // acquire share of work
        todo_lock.lock();
        int n = n_todo;
        n_todo -= batch_size;
        todo_lock.unlock();
        if (n < 0)
            return;
        if (n > batch_size)
            n = batch_size;

        // generate samples
        vector<potts_t> batch = generate_samples(q, L, T, batch_size);

        // add batch to dataset
        samples_lock.lock();
        samples.insert(samples.end(), batch.begin(), batch.end());
        samples_lock.unlock();
    }
}

int main(int argc, char* argv[]) // usage: <name>.exe <N_samples> <q> <L> <T> <threads> <batch_size> <output_path> 
{
    if (argc < 7)
        cout << "not enough arguments" << endl;

    unsigned int N = n_todo = stoi(argv[1]);
    q = stoi(argv[2]);
    L = stoi(argv[3]);
    T = stof(argv[4]);
    unsigned int n_threads = stoi(argv[5]);
    batch_size = stoi(argv[6]);
    string target_file(argv[7]);

    if (n_threads < 1)
        n_threads = 1;
    cout << "generating " << N << " samples of q = " << q << ", L = " << L << ", T = " << T << endl;
    cout << "using " << n_threads << " threads with batches of " << batch_size << " samples" << endl;
     
    srand(time(0));

    auto t_start = high_resolution_clock::now();
    
    // spawn (n_thread - 1) extra threads
    vector<thread> threads;
    for (int t = 0; t < n_threads - 1; t++)
        threads.push_back(thread(work));

    work(); // work on the main thread as well

    // join all other threads
    for (thread& worker : threads)
        worker.join();

    auto t_end = high_resolution_clock::now();

    const vector<long unsigned> shape{ N, L, L };
    npy::SaveArrayAsNumpy(target_file, false, shape.size(), shape.data(), samples);

    auto ms_int = duration_cast<milliseconds>(t_end - t_start);
    cout << "took " << ms_int.count() << " ms" << endl;

    return 0;
}
