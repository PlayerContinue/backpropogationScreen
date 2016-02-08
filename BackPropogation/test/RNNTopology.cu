#include "RNNTopology.cuh"

//*********************************************
//Constructor/Destructor
//*********************************************
RNNTopology::RNNTopology(){
	this->host_or_device = this->DEVICE;
}
//Constructor Telling if the network is host or device
RNNTopology::RNNTopology(HOST_DEVICE type){
	this->host_or_device = type;
}

//*********************************************
//Network Construction
//Construct the network from the settings
//*********************************************
//Build the Topology of the Network based on the settings provided
bool RNNTopology::buildTopology(NSettings settings){
	this->settings = settings;//Store the settings
	int number_weights = ((1+this->settings.i_input)*this->settings.i_number_start_nodes)
		+ (this->settings.i_output*this->settings.i_number_start_nodes);//Number of weights in the current network
	//Build the RNN Network on the host initially
	//Create the network on the host
	this->host_weights = thrust::host_vector<WEIGHT_TYPE>(number_weights);
	this->host_bias = thrust::host_vector<WEIGHT_TYPE>(this->settings.i_number_start_nodes + this->settings.i_output);
	this->host_from = thrust::host_vector<unsigned int>(number_weights);
	this->host_to = thrust::host_vector<unsigned int>(number_weights);


	//Start by settings the weights
	std::srand((unsigned)(time(NULL)));//Randomize the values using current time as seed
	thrust::generate(thrust::host, this->host_weights.begin(), this->host_weights.end(), RandomClamped);
	//Find the bias
	thrust::generate(thrust::host, this->host_bias.begin(), this->host_bias.end(), RandomClamped);
	//Create the initial to and from list for hidden nodes
	for (int i = 0; i < this->settings.i_number_start_nodes; i++){
		for (int j = 0; j < this->settings.i_input; j++){
			this->host_from[((i)*(this->settings.i_input + 1) + j)] = j;//from list goes from all inputs and itself
			this->host_to[((i)*(this->settings.i_input + 1) + j)] = i + this->settings.i_input;//to list is ordered by number
		}
		this->host_from[((i)*(this->settings.i_input + 1) + this->settings.i_input)] = i + this->settings.i_input;
		this->host_to[((i)*(this->settings.i_input + 1) + this->settings.i_input)] = i + this->settings.i_input;
	}
	
	//Store the number of weights by node type
	this->network_info[INPUT_WEIGHTS] = 0;
	this->network_info[HIDDEN_WEIGHTS] = (1 + this->settings.i_input)*this->settings.i_number_start_nodes;
	this->network_info[OUTPUT_WEIGHTS] = (this->settings.i_output*this->settings.i_number_start_nodes);
	this->network_info[INPUT_NODES] = this->settings.i_input;
	this->network_info[HIDDEN_NODES] = this->settings.i_number_start_nodes;
	this->network_info[OUTPUT_NODES] = this->settings.i_output;


	//Create the initial to and from list for output nodes
	for (int i = 0; i < this->settings.i_output; i++){
		for (int j = 0; j < this->settings.i_number_start_nodes; j++){
			this->host_from[this->network_info[HIDDEN_WEIGHTS] + (i*this->settings.i_number_start_nodes) + j] = j + this->settings.i_input;
			this->host_to[this->network_info[HIDDEN_WEIGHTS] + (i*this->settings.i_number_start_nodes) + j] = i + this->settings.i_input + this->settings.i_number_start_nodes;
		}
	}

	//Create the device_output with one number for each node
	this->host_output = thrust::host_vector<WEIGHT_TYPE>(this->settings.i_input + this->settings.i_number_start_nodes + this->settings.i_output);

	//Move data from host to device
	if (this->host_or_device == DEVICE){
		this->device_from = this->host_from;
		this->device_to = this->host_to;
		this->device_weights = this->host_weights;
		this->device_output = this->host_output;
		this->device_bias = this->host_bias;
		this->cleanTopology();
	}
	
		
	

	return true;
}

//*********************************************
//Network Information
//Retrieves Information about the network to provide 
// to the training and running to allow them to function
//*********************************************
//Return the number of layers in the network
int RNNTopology::numberLayers(){
	return 3;
}

//Get number of nodes in a particular layer
int RNNTopology::InfoInLayer(int layer, INFO_TYPE info_desired){
	return this->network_info[info_desired];
}



//*********************************************
//Device Vectors
//Retrieves the networks layers and positions
//*********************************************


TopologyLayerData RNNTopology::getLayer(int layer){
	return TopologyLayerData(this->device_weights.begin(),this->device_weights.end(),
		this->device_from.begin(),this->device_from.end(),
		this->device_to.begin(),this->device_to.end(),
		this->device_output.begin(),this->device_output.end(),
		this->device_bias.begin(),this->device_bias.end());
}

//*********************************************
//Clean Up Topology
//Remove the current values
//*********************************************

bool RNNTopology::cleanTopology(){
	thrust::fill(this->device_output.begin(), this->device_output.end(), (WEIGHT_TYPE)0);
	return true;
}

bool RNNTopology::emptyTopology(){
	clean_device::free(this->device_from);
	clean_device::free(this->device_to);
	clean_device::free(this->device_weights);
	clean_device::free(this->device_output);
	clean_device::free(this->device_bias);
	return true;
}

//*********************************************
//CheckPoint
//Output Values as a checkpoint
//*********************************************
std::ostream& RNNTopology::createCheckpoint(std::ostream &os){
	if (this->device_weights.size() > this->host_weights.size()){//Copy everything only if size has changed, otherwise only move changed values
		this->host_from.resize(this->device_from.size());
		thrust::copy(this->device_from.begin(), this->device_from.end(), this->host_from.begin());
		this->host_to.resize(this->device_to.size());
		thrust::copy(this->device_to.begin(), this->device_to.end(), this->host_to.begin());
		this->host_bias.resize(this->device_bias.size());
		this->host_weights.resize(this->device_weights.size());
	}
	thrust::copy(this->device_weights.begin(), this->device_weights.end(), this->host_weights.begin());
	thrust::copy(this->device_bias.begin(), this->device_bias.end(), this->host_bias.begin());
	//Output host_to
	os << "From" << endl;
	os << this->host_from.size() << endl;
	thrust::copy(this->host_from.begin(), this->host_from.end(), std::ostream_iterator<int>(os, " "));
	os << endl;
	os << "To" << endl;
	os << this->host_to.size() << endl;
	//Output host_from
	thrust::copy(this->host_to.begin(), this->host_to.end(), std::ostream_iterator<int>(os, " "));
	os << endl;
	//Output Weights
	os << "Weights" << endl;
	os << this->host_weights.size() << endl;
	thrust::copy(this->host_weights.begin(), this->host_weights.end(), std::ostream_iterator<WEIGHT_TYPE>(os, "\n"));
	os << endl;
	os << "Bias" << endl;
	os << this->host_bias.size() << endl;
	thrust::copy(this->host_bias.begin(), this->host_bias.end(), std::ostream_iterator<WEIGHT_TYPE>(os, "\n"));
	os << endl;


	return os;
};
