#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <vector>
#include <time.h>
#include "util.h"
#include "CSettings.h"
using namespace thrust;
using namespace thrust::placeholders;
//Define a type so it can use either double or float, depending on what turns out to be better
#define weight_type thrust::complex<double>
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains all the public functions required for a network to function. Is a base for other networks
//
//****************************************************************************************************


//Contains the methods
class NetworkBase {
	//*********************
	//Class Variables
	//*********************

public:
	//*********************
	//Run The Network
	//*********************
	virtual device_vector<weight_type> runNetwork(weight_type* in) = 0;
	virtual void InitializeRun() = 0;
	//***************************
	//Train the Network
	//***************************
	//Initilialize the network for training
	virtual void InitializeTraining() = 0;
	//Run a round of training
	virtual void StartTraining(weight_type* in, weight_type* out) = 0;
	//Apply the error to the network
	virtual void ApplyError() = 0;

public:

	//*********************
	//Visualization
	//*********************

	//Only used for dubug. Outputs a simple example of what the network looks like
	virtual void VisualizeNetwork() = 0;

	virtual ostream& OutputNetwork(ostream &os) = 0;

	//***************************
	//Modify Structure Of Neuron
	//***************************
	virtual void addNeuron(int numberNeuronsToAdd) = 0;

	//Add a new weight between neurons
	virtual void addWeight(int numberWeightsToAdd) = 0;

public:
	virtual void ResetSequence() = 0;

	//Copies the information stored on the GPU into main memory
	virtual void CopyToHost() = 0;

	//Copies the information stored in Main Memory into GPU Memory
	virtual void CopyToDevice() = 0;

	//Copies information stored on the GPU memory into the Main Memory
	//Removes the GPU Memory copies
	virtual void cleanNetwork()= 0;

	//Empty all of the GPU memory currently used
	//Primarily used when finished training or running the network
	virtual void emptyGPUMemory() = 0;
};

//*********************
//Contains Functions to empty a vector
//*********************
namespace clear_vector{
	//Function to free memory from GPU
	template<class T> void free(T &V) {
		V.clear();
		V.shrink_to_fit();
	}

	template void free<thrust::device_vector<int> >(thrust::device_vector<int>& V);
	template void free<thrust::device_vector<double> >(
		thrust::device_vector<double>& V);
}