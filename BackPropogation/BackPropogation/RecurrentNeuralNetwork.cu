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
	//This set will also change over time and contains all the output values
	for (int i = this->settings.i_input; i < this->numberNonWeights; i++){
		//Map a weight from each of the input to each of the outputs
		for (int j = 0; j < this->settings.i_input; j++){
			//Each one starts with a random value as the output weight
			this->weights.push_back(RandomClamped());
			this->mapFrom.push_back(j);
			this->mapTo.push_back(i);
		}
		this->positionOfLastWeightToNode.push_back(((i - this->settings.i_input)*this->settings.i_input)+this->settings.i_input - 1);
	}
	this->numberOfNodes = this->settings.i_output;
	this->input_weights = this->weights.size();
}


//***************************
//Modify Structure Of Neuron
//***************************
int RecurrentNeuralNetwork::decideNodeToAttachTo(){
	return RandInt(this->numberNonWeights, this->output_values.size());
}

int RecurrentNeuralNetwork::decideNodeToAttachFrom(){
	return RandInt(this->numberNonWeights, this->output_values.size());
}





weight_type RecurrentNeuralNetwork::getNewWeight(){
	return RandomClamped();
}



void RecurrentNeuralNetwork::addWeight(int numberWeightsToAdd){
	int decideTo;
	int decideFrom;
	host_vector<weight_type>::iterator itWeights;
	host_vector<int>::iterator itMapFrom;
	host_vector<int>::iterator itMapTo;
	for (int i = 0; i < numberWeightsToAdd; i++){
		//A new weight should be added between existing neurons
		//Add one weight from any current node to the new nodes
		decideTo = this->decideNodeToAttachTo();
		decideFrom = this->decideNodeToAttachTo();
		itWeights = this->weights.begin() + (this->positionOfLastWeightToNode[decideTo]);
		itMapFrom = this->mapFrom.begin() + (this->positionOfLastWeightToNode[decideTo]);
		itMapTo = this->mapTo.begin() + (this->positionOfLastWeightToNode[decideTo]);

		this->weights.insert(itWeights,this->getNewWeight());
		this->mapTo.insert(itMapTo,decideTo);
		this->mapFrom.insert(itMapFrom,decideFrom);
		//Add one weight from the new node to any other node

		//Increment the position
		for (int j = decideTo; j < this->positionOfLastWeightToNode.size(); j++){
			this->positionOfLastWeightToNode[j] += 1;
		}
		this->input_weights++;

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
				this->positionOfLastWeightToNode.push_back((long)((this->settings.i_input*j)) + this->settings.i_input - 1);
			}
			//Add new weights from the new nodes to the output nodes
			for (int j = this->settings.i_input; j < this->numberNonWeights; j++){
				for (int k = this->numberNonWeights; k < this->output_values.size(); k++){
					//Create a new weight from the current node to the weight
					//Create a new weight
					this->weights.push_back(RandomClamped());
					//Set a new pointer from the one new node to each of the output nodes
					this->mapFrom.push_back(k);
					//Map the new weights to the output
					this->mapTo.push_back(j);
				}
				this->positionOfLastWeightToNode[j - this->settings.i_input] = this->weights.size()-1;
			}
			this->numberOfNodes += this->settings.i_output;

		}
		else if (true){
			//A new neuron is added
			addNewNeuron++;
		}



	}

	if (addNewNeuron > 0){
		host_vector<weight_type>::iterator it = this->weights.begin() + input_weights;
		host_vector<int>::iterator itInt = this->mapFrom.begin() + input_weights;
		host_vector<int>::iterator itInt2 = this->mapTo.begin() + this->input_weights;
		int total_nodes_weights_before_output_added = 0;//Count the number of weights which are added before the to output nodes are found
		//Insert any new Neurons which were chosen to be created
		//Create connection from input to new node
		for (int i = addNewNeuron - 1; i > -1; i--){
			//Add the new neuron
			this->output_values.push_back(0);
			for (int j = 0; j < this->settings.i_input; j++){
				//Insert the connections to the input
				this->weights.insert(it, getNewWeight());
				this->mapFrom.insert(itInt, this->settings.i_input - j - 1);
				this->mapTo.insert(itInt2, this->output_values.size()-1);
				this->input_weights++;//Increase input_weights end position
				total_nodes_weights_before_output_added++;//Increment the number of weights added
			}
			this->positionOfLastWeightToNode.push_back(this->input_weights);
		}
		
		
		it = this->weights.begin() + this->positionOfLastWeightToNode[0] + total_nodes_weights_before_output_added + 1;
		itInt = this->mapFrom.begin() + this->positionOfLastWeightToNode[0] + total_nodes_weights_before_output_added + 1;
		itInt2 = this->mapTo.begin() + this->positionOfLastWeightToNode[0] + total_nodes_weights_before_output_added + 1;
		//Create connection from new node to output node
		for (int i = 0; i < this->settings.i_output; i++){
			for (int j = addNewNeuron - 1; j > -1; j--){
				//Insert a connection to the output
				this->weights.insert(it, getNewWeight());
				this->mapFrom.insert(itInt, this->output_values.size()-1);
				this->mapTo.insert(itInt2, i + this->settings.i_input);
			}
			it += addNewNeuron + numberOfNodes - this->settings.i_output;
			itInt += addNewNeuron + numberOfNodes - this->settings.i_output;
			itInt2 += addNewNeuron + numberOfNodes - this->settings.i_output;
		}
		this->numberOfNodes += addNewNeuron;

		//Increment the stored position of the last weight
		for (int i = 0; i < this->settings.i_output; i++){
			this->positionOfLastWeightToNode[i] += total_nodes_weights_before_output_added + i + 1;
		}
	}


}



//*********************
//Run The Network
//*********************
template <typename T>
struct multiply : public thrust::unary_function <T,T >{

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return (thrust::get<0>(x) * thrust::get<1>(x));
	}

};

thrust::host_vector<weight_type> RecurrentNeuralNetwork::runNetwork(weight_type* in){
	//Sum all the input values
	device_vector<int> GPUOutput_values = this->output_values;//Copy the output_nodes
	device_vector<int> GPUMapFrom = this->mapFrom;//Copy the map from
	device_vector<int> GPUMapTo = this->mapTo; //Copy the mapTo
	device_vector<weight_type> GPUWeights = this->weights;

	//Copy the input into the GPU memory
	for (int i = 0; i < this->settings.i_input; i++){
		weights[i] = in[i];
	}
	

	//Reduce the input into the sum for each node
	reduce_by_key(GPUMapTo.begin() + settings.i_input, 
		GPUMapTo.begin() + settings.i_input + this->numberOfNodes,
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(GPUWeights.begin() + this->settings.i_input,
		make_permutation_iterator(
		GPUWeights.begin(),
		GPUMapFrom.begin() + this->settings.i_input
		))
		),
		multiply<weight_type>()),
		thrust::make_discard_iterator(),
		GPUOutput_values.begin() + this->settings.i_output
		);


	return host_vector<weight_type>();
}


//*********************
//Misc
//*********************
void RecurrentNeuralNetwork::VisualizeNetwork(){
	std::cout << "Weight" << "\t" << "In" << "\t" << "Out" << endl;
	for (int i = 0; i < this->weights.size(); i++){
		std::cout << i << ") " << this->weights[i] << "\t" << this->mapFrom[i] << "\t" << this->mapTo[i] << endl;
	}
	std::cout << endl;
}