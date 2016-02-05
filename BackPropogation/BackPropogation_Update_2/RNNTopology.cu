#include "RNNTopology.cuh"

//*********************************************
//Constructor/Destructor
//*********************************************
RNNTopology::RNNTopology(){}
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
	int number_weights = (this->settings.i_input*this->settings.i_number_start_nodes)
		+ (this->settings.i_output*this->settings.i_number_start_nodes) + this->settings.i_number_start_nodes;//Number of weights in the current network
	//Build the RNN Network on the host initially
	//Create the network on the host
	this->host_weights = thrust::host_vector<WEIGHT_TYPE>(number_weights);
	this->host_from = thrust::host_vector<unsigned int>(number_weights);
	this->host_to = thrust::host_vector<unsigned int>(number_weights);

	//Start by settings the weights
	std::srand((unsigned)(time(NULL)));//Randomize the values using current time as seed
	thrust::generate(thrust::host, this->host_weights.begin(), this->host_weights.end(), RandomClamped);
	//Create the to and from list
	for (int i = 0; i < this->settings.i_number_start_nodes; i++){
		for (int j = 0; j < this->settings.i_input; j++){
			this->host_from[((i - 1)*(this->settings.i_input + 1) + j)] = j;
			this->host_to[((i - 1)*(this->settings.i_input + 1) + j)] = j;
		}
		this->host_to[((i - 1)*(this->settings.i_input + 1) + this->settings.i_input)] = i;
	}
	
	//Create the From List
	if (this->host_or_device == DEVICE){
		this->device_from = this->host_from;
		this->device_to = this->host_to;
		this->device_weights = this->host_weights;
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
int RNNTopology::numberNodesInLayer(int layer){
	return 0;
}

//*********************************************
//Device Vectors
//Retrieves the networks layers and positions
//*********************************************


TopologyLayerData RNNTopology::getLayer(int layer){
	return TopologyLayerData(this->device_weights.begin(),this->device_weights.end(),
		this->device_from.begin(),this->device_from.end(),
		this->device_to.begin(),this->device_to.end());
}


