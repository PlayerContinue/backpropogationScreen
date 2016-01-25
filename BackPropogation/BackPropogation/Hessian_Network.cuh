#ifndef _X_INCLUDE_LONG_TERM_
#define _X_INCLUDE_LONG_TERM_
#include "LongTermShortTermNetwork.cuh"
#endif


class Hessian_Network : public LongTermShortTermNetwork {
private:
	thrust::device_vector<weight_type> alphas;
	thrust::device_vector<weight_type> hessian;
public:
	//Default Constructor
	//Creates a network with 1 input and 1 output
	Hessian_Network() :LongTermShortTermNetwork(){
	};

	//Constructor which asks for a settings object 
	//The settings object contains all the information required to perform a function
	Hessian_Network(CSettings& settings):LongTermShortTermNetwork(settings){
	};

	//Create a LongTermShortTermNetwork from a checkpoint
	Hessian_Network(CSettings& settings, bool checkpoint) :LongTermShortTermNetwork(settings,checkpoint){
	};

	//***************************
	//Train the Network
	//***************************
	void InitializeHessianNetwork();
	void InitializeTraining();
	void ApplyError();
	void StartTraining(weight_type** in, weight_type** out);
	
	void TrainingRun(weight_type** in, weight_type** out);
private:

	
	//Find the alpha to multiply the delta by
	void findAlpha();

	

	//Find the hessian free matrix
	void findHessianFreeMatrixForward();

	void findHessianFreeMatrixBackward();

	//Find the deltas of the weights
	void findWeightDeltas();

	void ResetSequence(){
		LongTermShortTermNetwork::ResetSequence();
		thrust::fill(this->alphas.begin(), this->alphas.end(),(weight_type)0);
		thrust::fill(this->hessian.begin(), this->hessian.end(), (weight_type)0);
	};
	//Resets all values except the weights, to-from, and other position information
	void ResetAllSequence(){
		LongTermShortTermNetwork::ResetAllSequence();
		thrust::fill(this->alphas.begin(), this->alphas.end(), (weight_type)0);
		thrust::fill(this->hessian.begin(), this->hessian.end(), (weight_type)0);
	}

	void emptyGPUMemory(){
		LongTermShortTermNetwork::emptyGPUMemory();
		clear_vector::free(this->alphas);
		clear_vector::free(this->hessian);
	}
};