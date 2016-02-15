/*
Programmer: David Greenberg
Reason: Contains the functions for training a recurrent network in a Hessian-Free Optimization function

*/

#include "TrainerBase.cuh"
#include "Functors.cuh"

class RNNTrainer :public TrainerBase{
private:
	thrust::device_vector<WEIGHT_TYPE> device_error;
	thrust::host_vector<WEIGHT_TYPE> host_error;
	thrust::device_vector<WEIGHT_TYPE> device_weight_error;//Contains the error of the weights
	thrust::host_vector<WEIGHT_TYPE> host_weight_error;
	thrust::device_vector<WEIGHT_TYPE> device_input;//Input from the previous run
	thrust::host_vector<WEIGHT_TYPE> host_input;//Input from the previous run
	TopologyLayerData layer_data;
public:
	RNNTrainer();

	//*********************************************
	//Create Training Enviornment
	//
	//*********************************************
	//Create the training enviornment before training starts
	bool createTrainingEnviornment(TopologyBase&, NSettings);

	//*********************************************
	//Run Training
	//Train the network on the given input, output pair
	//*********************************************
	void train(thrust::host_vector<WEIGHT_TYPE> input, thrust::host_vector<WEIGHT_TYPE> output);
	void train(thrust::device_vector<WEIGHT_TYPE> input, thrust::device_vector<WEIGHT_TYPE> output);
	
	//Run the network forward to find an output
	void forwardRun();

	//Find the gradiant values to apply
	void findGradiant(thrust::device_vector<WEIGHT_TYPE>::iterator start_target);

	//Apply the error
	void applyGradiant();

	//*********************************************
	//Clean Up Topology
	//Remove the current values
	//*********************************************
	//Empty the topology entirely
	bool cleanTrainer();

	//Clean the network for a run
	bool emptyTrainer();

	//*********************************************
	//Checkpoint
	//Create a Checkpoint of the Training Enviornment
	//*********************************************
	std::ostream& createCheckpoint(std::ostream&);


};