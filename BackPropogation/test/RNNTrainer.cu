#include "RNNTrainer.cuh"

RNNTrainer::RNNTrainer(){

}

//*********************************************
//Create Training Enviornment
//
//*********************************************
//Create the training enviornment before training starts
bool RNNTrainer::createTrainingEnviornment(TopologyBase& topology, NSettings settings){
	this->_topology = &topology;
	this->_topology->buildTopology(settings);
	this->layer_data = this->_topology->getLayer(0);
	this->host_error = thrust::host_vector<WEIGHT_TYPE>(layer_data.device_output_vector_end - layer_data.device_output_vector_begin);
	this->device_error = this->host_error;
	this->host_input = thrust::host_vector<WEIGHT_TYPE>(layer_data.device_output_vector_end - layer_data.device_output_vector_begin);
	this->device_input = this->host_input;
	return true;
}

//*********************************************
//Run Training
//Train the network on the given input, output pair
//*********************************************
void RNNTrainer::train(thrust::host_vector<WEIGHT_TYPE> input, thrust::host_vector<WEIGHT_TYPE> output){
	this->train((thrust::device_vector<WEIGHT_TYPE>)input, (thrust::device_vector<WEIGHT_TYPE>)output);
}
void RNNTrainer::train(thrust::device_vector<WEIGHT_TYPE> input, thrust::device_vector<WEIGHT_TYPE> output){
	thrust::device_vector<WEIGHT_TYPE>::iterator start_of_input = input.begin();
	thrust::device_vector<WEIGHT_TYPE>::iterator end_of_input = input.begin() + this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
	thrust::device_vector<WEIGHT_TYPE>::iterator start_of_output = output.begin();
	thrust::device_vector<WEIGHT_TYPE>::iterator end_of_output = output.begin()+this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES);
	thrust::copy(start_of_input, end_of_input, this->layer_data.device_output_vector_begin);//Copy the input values



	for (int i = 0; i < input.size(); i += this->_topology->InfoInLayer(0, TopologyBase::INPUT_NODES)){
		thrust::copy(this->layer_data.device_output_vector_begin, this->layer_data.device_output_vector_end, this->device_input.begin());//Copy the output of the previous run
		
		forwardRun();

		//Increment the input and output
		start_of_input += this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
		end_of_input += this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
		start_of_output += this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES);
		end_of_output += this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES);
		if (input.size() >(i + this->_topology->InfoInLayer(0, TopologyBase::INPUT_NODES))){
			thrust::copy(start_of_input, end_of_input, this->layer_data.device_output_vector_begin);//Copy the input values
		}
	}

}

void RNNTrainer::forwardRun(){
	//Store where the math should come from
	thrust::device_vector<WEIGHT_TYPE>::iterator previous_start = this->device_input.begin();
	thrust::device_vector<WEIGHT_TYPE>::iterator previous_end = this->device_input.end();
	thrust::device_vector<WEIGHT_TYPE>::iterator output_start = this->layer_data.device_output_vector_begin + this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
	thrust::device_vector<WEIGHT_TYPE>::iterator output_end = this->layer_data.device_output_vector_begin + this->_topology->InfoInLayer(0, TopologyBase::INPUT_NODES) + this->_topology->InfoInLayer(0, TopologyBase::HIDDEN_NODES);
	thrust::device_vector<unsigned int>::iterator from_start = this->layer_data.to_vector_begin;
	thrust::device_vector<unsigned int>::iterator from_end = this->layer_data.to_vector_begin + this->_topology->InfoInLayer(0, TopologyBase::HIDDEN_WEIGHTS);

	//testing::outputToFile<WEIGHT_TYPE>(layer_data.device_output_vector_begin, layer_data.device_output_vector_end, "out1", "tests/output_vector.txt");
	//In RNN there are only three layers, input/hidden/output
	for (int i = 0; i < 2; i++){
		thrust::reduce_by_key(//Find hidden layer values
			from_start,
			from_end,
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			thrust::make_permutation_iterator(//Permute the previous output such that it is in the order of the weights
			previous_start,
			this->layer_data.from_vector_begin
			),
			this->layer_data.weight_vector_begin
			)
			),
			functors::transform_functors::multiplies<WEIGHT_TYPE>()
			),
			thrust::make_discard_iterator(),
			output_start);
		
		//testing::outputToFile<WEIGHT_TYPE>(layer_data.device_output_vector_begin, layer_data.device_output_vector_end, "out1", "tests/output_vector.txt");
		
		thrust::transform(
			output_start,
			output_end,
			layer_data.device_bias_begin,
			output_start,
			functors::transform_functors::bias_sigmoid_functor<WEIGHT_TYPE>());
		
		//testing::outputToFile<WEIGHT_TYPE>(layer_data.device_output_vector_begin, layer_data.device_output_vector_end, "out1", "tests/output_vector.txt");

		previous_start = layer_data.device_output_vector_begin;
		previous_end = layer_data.device_output_vector_end;
		from_start = from_end;
		from_end += this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_WEIGHTS);
		output_start += this->_topology->InfoInLayer(0, TopologyBase::HIDDEN_NODES);
		output_end += this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_NODES);
	}
}

void RNNTrainer::findError(){
	//Find the error of the output
	thrust::transform(this->layer_data.device_output_vector_end - this->_topology->InfoInLayer(TopologyBase::OUTPUT_NODES),
		this->layer_data.device_output_vector_end,
		this->device_error.end() - this->_topology->InfoInLayer(TopologyBase::OUTPUT_NODES),

		)
}

//*********************************************
//Clean Up Topology
//Remove the current values
//*********************************************
//Empty the topology entirely
bool RNNTrainer::cleanTrainer(){
	thrust::fill(this->device_error.begin(), this->device_error.end(), (WEIGHT_TYPE)0);
	thrust::fill(this->device_input.begin(), this->device_input.end(), (WEIGHT_TYPE)0);
	this->_topology->cleanTopology();
	return true;
};

//Clean the network for a run
bool RNNTrainer::emptyTrainer(){
	clean_device::free(this->device_error);
	clean_device::free(this->device_input);
	this->_topology->emptyTopology();
	return true;
};

//*********************************************
//Checkpoint
//Create a Checkpoint of the Training Enviornment
//*********************************************
std::ostream& RNNTrainer::createCheckpoint(std::ostream& os){
	if (this->device_error.size() > this->host_error.size()){
		this->host_error.resize(this->device_error.size());
	}

	this->_topology->createCheckpoint(os);
	return os;
}