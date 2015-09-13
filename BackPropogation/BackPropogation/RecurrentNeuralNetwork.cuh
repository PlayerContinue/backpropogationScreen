#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <vector>
#include "util.h"
#include "CSettings.h"
using namespace thrust;
using namespace thrust::placeholders;
//Define a type so it can use either double or float, depending on what turns out to be better
#define weight_type double
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains all the functions and methods to train and alter a ReccurentNeuralNetwork
//Initial Version utilizes Long Short Term Memory
//
//****************************************************************************************************


//Contains the methods
class RecurrentNeuralNetwork {
	//*********************
	//Class Variables
	//*********************
private:
	
	vector<long> positionOfLastWeightToNode;
	long numberOfNodes; //The number of nodes currently in the system which can be linked to
	long numberNonWeights; //Keeps track of the number of non-weights before an actual weight appears
	long input_weights;
	//Stores the weights between neurons
	host_vector<weight_type> weights;
	host_vector<weight_type> output_values;
	//Contains whuch weight is connected to which neuron
	host_vector<int> mapTo;
	host_vector<int> mapFrom;
	
	
	CSettings settings;
public:
	//*********************
	//Constructors
	//*********************
	//Default Constructor
	//Creates a network with 1 input and 1 output
	RecurrentNeuralNetwork();

	//Constructor which asks for a settings object 
	//The settings object contains all the information required to perform a function
	RecurrentNeuralNetwork(CSettings& settings);
	//*********************
	//Initialization
	//*********************
private:
	//Initialize the network from the settings object if possible
	void initialize_network();

public:
	//*********************
	//Run The Network
	//*********************
	host_vector<weight_type> runNetwork(weight_type* in);

	//*********************
	//Visualization
	//*********************
	//Only used for dubug. Outputs a simple example of what the network looks like
	void VisualizeNetwork();

	//***************************
	//Modify Structure Of Neuron
	//***************************
	void addNeuron(int numberNeuronsToAdd);

	//Add a new weight between neurons
	void addWeight(int numberWeightsToAdd);
private:
	//Decide which node the new weight should be attached to 
	int decideNodeToAttachTo();
	//Decide which node the new weight should be attached from
	int decideNodeToAttachFrom();

	//Get a new weight
	weight_type getNewWeight();

};