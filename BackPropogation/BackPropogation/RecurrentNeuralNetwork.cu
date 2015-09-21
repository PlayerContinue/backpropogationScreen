#include "RecurrentNeuralNetwork.cuh"

RecurrentNeuralNetwork::RecurrentNeuralNetwork(){
	this->settings = CSettings();
	RecurrentNeuralNetwork(this->settings);
}

RecurrentNeuralNetwork::RecurrentNeuralNetwork(CSettings& settings){
	this->settings = settings;
	this->initialize_network();
}

void RecurrentNeuralNetwork::initialize_network(){
	this->weights = host_vector<weight_type>();
	this->mapTo = host_vector<int>();
	this->mapFrom = host_vector<int>();
	this->output_values = host_vector<weight_type>();
	positionOfLastWeightToNode = vector<long>();
	//The initial size of the network will consist of only the input and output layers
	//This is an approximation of a feedforward network
	//Add the total number of nodes in the input layer as the first set of nodes
	//This set will change when running the system
	for (int i = 0; i < this->settings.i_input; i++){
		//Set the first output_values as the input
		this->output_values.push_back(0);
	}

	//Create a container for current value of each node
	for (int i = 0; i < this->settings.i_output; i++){
		this->output_values.push_back(0);
	}
	this->numberNonWeights = this->settings.i_input + this->settings.i_output;

	//Seed the random
	srand((unsigned)(time(NULL)));

	//This set will also change over time and contains all the output values
	for (int i = 0; i < this->settings.i_output; i++){
		if (i > 0){
			this->positionOfLastWeightToNode.push_back(this->positionOfLastWeightToNode[i - 1]);
		}else{
			this->positionOfLastWeightToNode.push_back(-1);
		}
		
		//Map a weight from each of the input to each of the outputs
		for (int j = 0; j < this->settings.i_input; j++){
			//Each one starts with a random value as the output weight
			this->weights.push_back(RandomClamped());
			this->mapFrom.push_back(j);
			this->mapTo.push_back(i);
			//Increment the position of the weights
			this->positionOfLastWeightToNode[i]++;
		}
	}
	this->numberOfNodes = this->settings.i_output;
	this->input_weights = this->weights.size();
}


//***************************
//Modify Structure Of Neuron
//***************************
int RecurrentNeuralNetwork::decideNodeToAttachTo(){
	vector<int> notFullyConnected = vector<int>();
	//Find how many nodes are not fully connected
	for (int k = this->settings.i_output; k<this->numberOfNodes; k++){
		if (this->positionOfLastWeightToNode[k] - (k > this->settings.i_output ? this->positionOfLastWeightToNode[k-1] : -1) < (this->numberOfNodes + this->settings.i_input - this->settings.i_output)){
			notFullyConnected.push_back(k);
		}
	}

	if (notFullyConnected.size() > 0){
		//Return a random number in the set of not fully connected nodes
		//It's the number contained in the positionOfLastWeightToNode which is this->settings.i_input less than its actual position
		return notFullyConnected[RandInt(0, notFullyConnected.size()-1)];
	}
	else{
		//All nodes are fully connected and no new weights can be added
		return -1;
	}
}

int RecurrentNeuralNetwork::decideNodeToAttachFrom(int attachTo){
	vector<int> notConnectedTo = vector<int>();
	bool containsValue = false;
	int start = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo-1] : 0);
	int end = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo] : this->positionOfLastWeightToNode[attachTo]);
	for (unsigned int k = this->numberNonWeights; k < this->output_values.size(); k++){
		for (int i = start; i<=end; i++){
			if (this->mapFrom[i] == k){
				//The value is already contained in the system
				containsValue = true;
				break;
			}
		}

		if (!containsValue){
			notConnectedTo.push_back(k);
		}
		containsValue = false;


	}


	
	if (notConnectedTo.size() != 0){
		//It's the number contained in the positionOfLastWeightToNode which is this->settings.i_input less than its actual position
		return notConnectedTo[RandInt(0, notConnectedTo.size() - 1)];
	}
	else{
		return -1;
	}

}





weight_type RecurrentNeuralNetwork::getNewWeight(){
	return RandomClamped();
}



void RecurrentNeuralNetwork::addWeight(int numberWeightsToAdd){
	int decideTo;//Where the input should go to
	int decideFrom;//Where the input should come from
	int positionToAdd;//The node to add in positionOfLastWeightToNode
	host_vector<weight_type>::iterator itWeights;
	host_vector<int>::iterator itMapFrom;
	host_vector<int>::iterator itMapTo;
	for (int i = 0; i < numberWeightsToAdd; i++){
		//A new weight should be added between existing neurons
		//Add one weight from any current node to the new nodes
		decideTo = this->decideNodeToAttachTo();
		if (decideTo != -1){//At least one node isn't completely connected to every other node
			decideFrom = this->decideNodeToAttachFrom(decideTo);
			if (decideFrom != -1){//The node which was chosen has at least one node not connected to it
				positionToAdd = (this->positionOfLastWeightToNode[decideTo]) + 1;
				decideTo += this->settings.i_input;
				itWeights = this->weights.begin() + positionToAdd;
				itMapFrom = this->mapFrom.begin() + positionToAdd;
				itMapTo = this->mapTo.begin() + positionToAdd;

				//Add the new weight into the position
				this->weights.insert(itWeights, this->getNewWeight());
				this->mapTo.insert(itMapTo, decideTo);
				this->mapFrom.insert(itMapFrom, decideFrom);

				//Increment the position for any following the current node
				for (unsigned int j = decideTo - this->settings.i_input; j < this->positionOfLastWeightToNode.size(); j++){
					this->positionOfLastWeightToNode[j] += 1;
				}
				this->input_weights++;
			}
		}

	}

	//Add the new weight positions to the output weight list
	for (int i = 0; i < this->settings.i_output; i++){
		this->positionOfLastWeightToNode[i] += numberWeightsToAdd;
	}
}

void RecurrentNeuralNetwork::addNeuron(int numberNeuronsToAdd){
	int addNewNeuron = 0;//Count the number of neurons to add. Multiple insertions at one time are easier than a single insertion

	//Add the new nodes defined by the numberofNodesToAdd
	for (int i = 0; i < numberNeuronsToAdd; i++){
		if (this->numberOfNodes == this->settings.i_output){
			//There are currently only the input/output nodes
			//In order to not need to delete any weights (which would cost quite a bit of time, we add in X new nodes, where X is the number of nodes in the output
			for (int j = 0; j < this->settings.i_output; j++){
				//Add the new node
				this->output_values.push_back(0);
				
				
				for (int k = (this->settings.i_input*j); k < ((this->settings.i_input*j)) + this->settings.i_input; k++){
					//We need to move all those inputs/outputs weights to the single new nodes
					//Move all weights between the current input/output pair to be input -> new node -> output
					this->mapTo[k] = this->output_values.size() - 1;
				}
				this->positionOfLastWeightToNode.push_back(this->positionOfLastWeightToNode[j]);
				
			}
			//Add new weights from the new nodes to the output nodes
			for (int j = this->settings.i_input; j < this->numberNonWeights; j++){
				for (unsigned int k = this->numberNonWeights; k < this->output_values.size(); k++){
					//Create a new weight from the current node to the weight
					//Create a new weight
					this->weights.push_back(RandomClamped());
					//Set a new pointer from the one new node to each of the output nodes
					this->mapFrom.push_back(k);
					//Map the new weights to the output
					this->mapTo.push_back(j);
				}
				this->positionOfLastWeightToNode[j - this->settings.i_input] = this->weights.size() - 1;
			}
			this->numberOfNodes += this->settings.i_output;

		}
		else if (true){
			//A new neuron is added
			addNewNeuron++;
		}



	}

	if (addNewNeuron > 0){
		host_vector<weight_type>::iterator it = this->weights.begin() + this->input_weights;
		host_vector<int>::iterator itInt = this->mapFrom.begin() + this->input_weights;
		host_vector<int>::iterator itInt2 = this->mapTo.begin() + this->input_weights;
		int output_size = this->output_values.size();
		int total_nodes_weights_before_output_added = 0;//Count the number of weights which are added before the to output nodes are found
		//Insert any new Neurons which were chosen to be created
		//Create connection from input to new node
		for (int i = addNewNeuron-1; i > -1; i--){
			//Add the new neuron
			this->output_values.push_back(0);
			for (int j = this->settings.i_input - 1; j > -1; j--){
				//Insert the connections to the input
				it = this->weights.insert(it, getNewWeight());
				itInt = this->mapFrom.insert(itInt, j);
				itInt2 = this->mapTo.insert(itInt2,  output_size + i);
				this->input_weights++;//Increase input_weights end position
				total_nodes_weights_before_output_added++;//Increment the number of weights added
			}

			this->positionOfLastWeightToNode.push_back(this->input_weights - 1);
			
		}


		it = this->weights.begin() + this->input_weights + this->numberOfNodes - this->settings.i_output;
		itInt = this->mapFrom.begin() + this->input_weights + this->numberOfNodes - this->settings.i_output;
		itInt2 = this->mapTo.begin() + this->input_weights + this->numberOfNodes - this->settings.i_output;
		//Create connection from new node to output node
		for (int i = 0; i < this->settings.i_output; i++){
			for (int j = addNewNeuron - 1; j > -1; j--){
				//Insert a connection to the output
				it = this->weights.insert(it, getNewWeight());
				itInt = this->mapFrom.insert(itInt, j + this->numberOfNodes + this->settings.i_input);
				itInt2 = this->mapTo.insert(itInt2, i + this->settings.i_input);
			}
			it += addNewNeuron + numberOfNodes - this->settings.i_output;
			itInt += addNewNeuron + numberOfNodes - this->settings.i_output;
			itInt2 += addNewNeuron + numberOfNodes - this->settings.i_output;
		}
		this->numberOfNodes += addNewNeuron;
		this->positionOfLastWeightToNode[0] = this->input_weights + (this->numberOfNodes - this->settings.i_output-1);
		//Increment the stored position of the last weight
		for (int i = 1; i < this->settings.i_output; i++){
			this->positionOfLastWeightToNode[i] = this->positionOfLastWeightToNode[i-1] + this->numberOfNodes - this->settings.i_output;
		}
	}


}



//*********************
//Run The Network
//*********************
//Multiply two values
template <typename T>
struct multiply : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return (thrust::get<0>(x) * thrust::get<1>(x));
	}

};

//Perform a sigmoid function
template <typename T>
struct sigmoid_functor : public thrust::unary_function < T, T > {
	sigmoid_functor(){};

	__host__ __device__
		T operator()(const T &x) const{
		T z = thrust::exp(((T) -1) * x);
		return (T)1 / ((T)1 + z);
	}

};

thrust::device_vector<weight_type> RecurrentNeuralNetwork::runNetwork(weight_type* in){
	//Sum all the input values
	device_vector<weight_type> GPUOutput_values = this->output_values;//Copy the output_nodes
	device_vector<weight_type> GPUPreviousOutput_Values = this->output_values;
	device_vector<int> GPUMapFrom = this->mapFrom;//Copy the map from
	device_vector<int> GPUMapTo = this->mapTo; //Copy the mapTo
	device_vector<weight_type> GPUWeights = this->weights;

	//Copy the input into the GPU memory
	for (int i = 0; i < this->settings.i_input; i++){
		GPUOutput_values[i] = (weight_type)in[i];
	}
	this->sumNetworkValues(GPUOutput_values,//Copy the output_nodes
		GPUPreviousOutput_Values,
		GPUMapFrom,//Copy the map from
		GPUMapTo, //Copy the mapTo
		GPUWeights, 1);

	//Free the used memory
	clear_vector::free(GPUMapFrom);
	clear_vector::free(GPUMapTo);
	clear_vector::free(GPUWeights);
	//Return either of these two as the output
	if (this->settings.i_recurrent_flip_flop % 2 == 0){
		clear_vector::free(GPUOutput_values);
		return GPUPreviousOutput_Values;
	}
	else{
		clear_vector::free(GPUPreviousOutput_Values);
		return GPUOutput_values;
	}

}
//*********************
//Training the Network
//*********************
template <typename T>
struct find_error : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return thrust::pow((thrust::get<0>(x) - thrust::get<1>(x)),(T)2);
	}

};


void RecurrentNeuralNetwork::LongShortTermMemoryTraining(device_vector<weight_type> in, weight_type* out){

}

void RecurrentNeuralNetwork::InitializeRealTimeRecurrentTraining(){
	//Store all the values in the device
	//Will later add option for too little memory
	this->host_deltas = host_vector<weight_type>(this->weights.size());
	//Copy the information to the device
	this->CopyToDevice();
	this->total_error = 0;
}

void RecurrentNeuralNetwork::RealTimeRecurrentLearningTraining(weight_type* in, weight_type* out){
	this->total_error = this->RealTimeRecurrentLearningTraining(in, out, this->total_error, this->GPUMapTo, this->GPUMapFrom, this->GPUWeights, this->GPUOutput_values, this->GPUPreviousOutput_Values,
		this->device_deltas);
}

//Incomplete
//Need more info about how it works
weight_type RecurrentNeuralNetwork::RealTimeRecurrentLearningTraining(
	weight_type* in, 
	weight_type* out, 
	weight_type total_error,
	thrust::device_vector<int> &GPUMapTo,
	thrust::device_vector<int> &GPUMapFrom, 
	thrust::device_vector<weight_type> &GPUWeights, 
	thrust::device_vector<weight_type> &GPUOutput_values, 
	thrust::device_vector<weight_type> &GPUPreviousOutput_Values,
	thrust::device_vector<weight_type> &GPU_Deltas){
	
	device_vector<weight_type> output = device_vector<weight_type>(this->settings.i_output);
	
	//Copy the desired output into GPU memory
	for (int i = 0; i < this->settings.i_output; i++){
		output[i] = (weight_type)out[i];
	}

	//Copy the input into the GPU memory
	for (int i = 0; i < this->settings.i_input; i++){
		GPUOutput_values[i] = (weight_type)in[i];
	}

	
	//The GPU Sum is now set
	this->sumNetworkValues(GPUOutput_values,//Copy the output_nodes
		GPUOutput_values,
		GPUMapFrom,//Copy the map from
		GPUMapTo, //Copy the mapTo
		GPUWeights, 1);
	
	//Get the sum of the error
	weight_type current_total = thrust::reduce(thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		GPUOutput_values.begin() + this->settings.i_input,
		output.begin())),
		find_error<weight_type>()
		),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		GPUOutput_values.begin() + this->settings.i_input + this->settings.i_output,
		output.end())),
		find_error<weight_type>()
		)
		);	

	thrust::fill(GPUPreviousOutput_Values.begin(), GPUPreviousOutput_Values.end(), (weight_type) 0);
	thrust::transform(GPUOutput_values.begin(), GPUOutput_values.begin() + this->settings.i_output, output.begin(), GPUPreviousOutput_Values.begin(), _2/_1);

	//Add the new change to the deltas to the current delta
	thrust::transform(GPUWeights.begin(), GPUWeights.end(),
		GPU_Deltas.begin(),GPU_Deltas.begin(),
		((((weight_type)(this->settings.d_alpha)) * current_total) / _1));
	weight_type temp = thrust::reduce(GPU_Deltas.begin(), GPU_Deltas.end());
	return current_total;

}

void RecurrentNeuralNetwork::RealTimeRecurrentLearningApplyError(){

	thrust::transform(this->GPUWeights.begin(), this->GPUWeights.end(), this->device_deltas.begin(), this->GPUWeights.begin(), _1 + _2);

}

//*********************
//Hessian Free
//*********************

void RecurrentNeuralNetwork::InitializeHessianFreeOptimizationTraining(){

	//Store all the values in the device
	//Will later add option for too little memory
	this->host_deltas = host_vector<weight_type>(this->GPUOutput_values.size());
	//Copy the information to the device
	this->CopyToDevice();
	//Fill the intial previous output as 0
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
}
void RecurrentNeuralNetwork::HessianFreeOptimizationTraining(weight_type* in, weight_type* out){

	device_vector<weight_type> output = device_vector<weight_type>(this->settings.i_output);
	device_vector<weight_type> GPUOutput_Values_Copy(this->GPUOutput_values);//Contains a copy of the current GPU weights
	//Copy the desired output into GPU memory
	for (int i = 0; i < this->settings.i_output; i++){
		output[i] = (weight_type)out[i];
	}

	//Copy the input into the GPU memory
	for (int i = 0; i < this->settings.i_input; i++){
		this->GPUOutput_values[i] = (weight_type)in[i];
	}
	


	//The GPU Sum is now set
	this->sumNetworkValues(this->GPUOutput_values, 
		GPUOutput_Values_Copy,//Copy the output_nodes
		this->GPUMapFrom,//Copy the map from
		this->GPUMapTo, //Copy the mapTo
		this->GPUWeights, 1);

	thrust::transform(this->GPUOutput_values.begin() + this->settings.i_input, GPUOutput_values.end(), this->GPUPreviousOutput_Values.begin() + this->settings.i_input, this->device_deltas.begin(), (_1 - _2) / (weight_type)this->settings.d_alpha);

	//Get the next iteration of values
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Difference between target and current value
	//Store the current round of values into the GPUPreviousOutput_Values
	//Essentailly delta f(x_n)
	thrust::transform(this->GPUOutput_values.begin() + this->settings.i_input, this->GPUOutput_values.begin() + this->settings.i_input + this->settings.i_output, output.begin(), this->GPUPreviousOutput_Values.begin(), _2 - _1);

	clear_vector::free(output);
	clear_vector::free(output);
}
void RecurrentNeuralNetwork::HessianFreeOptimizationApplyError(){
	//Apply the found delta to all of the values 
	//Essentially, add the delta
	thrust::transform(this->GPUWeights.begin(), this->GPUWeights.end(), 
		thrust::make_permutation_iterator(this->device_deltas.begin(),this->GPUMapFrom.begin()), this->GPUWeights.begin(), _1 + _2);
}

//*********************
//Perform Functionality
//*********************

void RecurrentNeuralNetwork::sumNetworkValues(device_vector<weight_type> &GPUOutput_values,//Copy the output_nodes
	device_vector<weight_type> &GPUPreviousOutput_Values,
	device_vector<int> &GPUMapFrom,//Copy the map from
	device_vector<int> &GPUMapTo, //Copy the mapTo
	device_vector<weight_type> &GPUWeights, int number_of_rounds){

	for (int i = 0; i < number_of_rounds; i++){
		if (i % 2 != 0){//Store the results in the previous Output
			//Reduce the input into the sum for each neuron
			thrust::reduce_by_key(
				GPUMapTo.begin(),
				GPUMapTo.end(),
				//Transform by multiplying the weight by the previous output
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				GPUWeights.begin(),
				make_permutation_iterator(
				GPUOutput_values.begin(),
				GPUMapFrom.begin()
				))
				),
				multiply<weight_type>()),
				thrust::make_discard_iterator(),
				GPUPreviousOutput_Values.begin()
				);
			//Transform the output using the sigmoid function
			thrust::transform(GPUPreviousOutput_Values.begin(), GPUPreviousOutput_Values.end(), GPUPreviousOutput_Values.begin(), sigmoid_functor<weight_type>());
		}
		else{//Store in current output
			//Reduce the input into the sum for each neuron
			thrust::reduce_by_key(
				GPUMapTo.begin(),
				GPUMapTo.end(),
				//Transform by multiplying the weight by the previous output
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(GPUPreviousOutput_Values.begin(),
				make_permutation_iterator(
				GPUOutput_values.begin(),
				GPUMapFrom.begin()
				))
				),
				multiply<weight_type>()),
				thrust::make_discard_iterator(),
				GPUOutput_values.begin()
				);

			//Transform the output using the sigmoid function
			thrust::transform(GPUOutput_values.begin(), GPUOutput_values.end(), GPUOutput_values.begin(), sigmoid_functor<weight_type>());
		}
	}

}

void RecurrentNeuralNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(),this->GPUOutput_values.end(), 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
}


void RecurrentNeuralNetwork::CopyToHost(){
	//Copy the device memory to local
	thrust::copy(this->GPUMapFrom.begin(), this->GPUMapFrom.end(), this->mapFrom.begin());
	thrust::copy(this->GPUMapTo.begin(), this->GPUMapTo.end(), this->mapTo.begin());
	thrust::copy(this->GPUWeights.begin(), this->GPUWeights.end(), this->weights.begin());
	thrust::copy(this->device_deltas.begin(), this->device_deltas.end(), this->host_deltas.begin());
	thrust::copy(this->device_deltas.begin(), this->device_deltas.end(), this->host_deltas.begin());
	thrust::copy(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), this->output_values.begin());
}

void RecurrentNeuralNetwork::CopyToDevice(){
	this->device_deltas = this->host_deltas;
	this->GPUMapTo = this->mapTo;
	this->GPUMapFrom = this->mapFrom;
	this->GPUOutput_values = this->output_values;
	this->GPUPreviousOutput_Values = this->output_values;
	this->GPUWeights = this->weights;
}

void  RecurrentNeuralNetwork::cleanNetwork(){
	this->CopyToHost();
	//Free the used memory
	clear_vector::free(this->GPUMapFrom);
	clear_vector::free(this->GPUMapTo);
	clear_vector::free(this->GPUWeights);
	clear_vector::free(this->device_deltas);
	clear_vector::free(this->GPUOutput_values);
	clear_vector::free(this->GPUPreviousOutput_Values);
}

void  RecurrentNeuralNetwork::emptyGPUMemory(){

}
//*********************
//Misc
//*********************
ostream& RecurrentNeuralNetwork::OutputNetwork(ostream &os){
	return os;
}

void RecurrentNeuralNetwork::VisualizeNetwork(){
	cout.precision(20);
	std::cout << "Weight" << "\t" << "In" << "\t" << "Out" << endl;
	for (unsigned int i = 0; i < this->weights.size(); i++){
		std::cout << i << ") " << this->weights[i] << "\t" << this->mapFrom[i] << "\t" << this->mapTo[i] << endl;
	}
	std::cout << endl;
	cout << "Neuron Values" << endl;
	
	for (unsigned int i = this->settings.i_input; i < this->output_values.size(); i++){
		std::cout << i << ") " << this->output_values[i] << endl;
	}

	std::cout << endl;

	std::cout << endl;
	cout << "deltas" << endl;
	for (unsigned int i = 0; i < this->host_deltas.size(); i++){
		std::cout << i << ") " << this->host_deltas[i] << endl;
	}



}