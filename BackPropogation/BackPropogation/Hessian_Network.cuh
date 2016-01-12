#ifndef _X_INCLUDE_LONG_TERM_
#define _X_INCLUDE_LONG_TERM_
#include "LongTermShortTermNetwork.cuh"
#endif


class Hessian_Network : public LongTermShortTermNetwork {
private:
	thrust::device_vector<weight_type> alphas;

public:
	//Default Constructor
	//Creates a network with 1 input and 1 output
	Hessian_Network() :LongTermShortTermNetwork(){};

	//Constructor which asks for a settings object 
	//The settings object contains all the information required to perform a function
	Hessian_Network(CSettings& settings):LongTermShortTermNetwork(settings){};

	//Create a LongTermShortTermNetwork from a checkpoint
	Hessian_Network(CSettings& settings, bool checkpoint) :LongTermShortTermNetwork(settings,checkpoint){};

	//***************************
	//Train the Network
	//***************************

	void ApplyError();
	void StartTraining(weight_type** in, weight_type** out);
	void InitializeHessianNetwork();
	void TrainingRun(weight_type** in, weight_type** out);
private:

	
	//Find the alpha to multiply the delta by
	void findAlpha();

	

	//Find the hessian free matrix
	void findHessianFreeMatrix();

	

};