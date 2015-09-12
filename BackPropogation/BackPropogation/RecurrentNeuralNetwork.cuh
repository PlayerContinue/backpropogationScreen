#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "CSettings.h"
using namespace thrust;

//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains all the functions and methods to train and alter a ReccurentNeuralNetwork
//Initial Version utilizes Long Short Term Memory
//
//****************************************************************************************************


//Contains the methods
class RecurrentNeuralNetwork {
	//Define a type so it can use either double or float, depending on what turns out to be better
	typedef double weight_type;
	//Stores the weights between neurons
	host_vector<weight_type> weights;
	//Contains whuch weight is connected to which neuron
	host_vector<int> map;
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
	RecurrentNeuralNetwork(CSettings settings);
	//*********************
	//Initialization
	//*********************
private:
	//Initialize the network from the settings object if possible
	void initialize_network();


};