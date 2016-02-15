#include "RNNTrainer.cuh"

RNNTrainer::RNNTrainer(){

};

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
	this->host_weight_error = thrust::host_vector<WEIGHT_TYPE>(layer_data.weight_vector_end - layer_data.weight_vector_begin);
	this->host_input = thrust::host_vector<WEIGHT_TYPE>(layer_data.device_output_vector_end - layer_data.device_output_vector_begin);
	this->device_input = this->host_input;
	return true;
};

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
		
		this->forwardRun();//Find the predicted output
		this->findGradiant(start_of_output);//Find the gradiant
		//Increment the input and output
		start_of_input += this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
		end_of_input += this->_topology->InfoInLayer(0,TopologyBase::INPUT_NODES);
		start_of_output += this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES);
		end_of_output += this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES);
		if (input.size() >(i + this->_topology->InfoInLayer(0, TopologyBase::INPUT_NODES))){
			thrust::copy(start_of_input, end_of_input, this->layer_data.device_output_vector_begin);//Copy the input values
		}
	}

};

void RNNTrainer::forwardRun(){
	//Store where the math should come from
	thrust::device_vector<WEIGHT_TYPE>::iterator previous_start = this->device_input.begin();
	thrust::device_vector<WEIGHT_TYPE>::iterator previous_end = this->device_input.end();
	thrust::device_vector<WEIGHT_TYPE>::iterator output_start = this->layer_data.device_output_vector_begin;
	thrust::device_vector<WEIGHT_TYPE>::iterator output_end = this->layer_data.device_output_vector_end;
	thrust::device_vector<unsigned int>::iterator from_start = this->layer_data.to_vector_begin;
	thrust::device_vector<unsigned int>::iterator from_end = this->layer_data.to_vector_end;

	//testing::outputToFile<WEIGHT_TYPE>(layer_data.device_output_vector_begin, layer_data.device_output_vector_end, "out1", "tests/output_vector.txt");
	//In RNN there are only three layers, input/hidden/output
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
		

	thrust::transform(
		this->layer_data.device_output_vector_begin,
		this->layer_data.device_output_vector_end,
		layer_data.device_bias_begin,
		output_start,
		functors::transform_functors::bias_sigmoid_functor<WEIGHT_TYPE>());

};

void RNNTrainer::findGradiant(thrust::device_vector<WEIGHT_TYPE>::iterator start_target){
	thrust::device_vector<unsigned int>::iterator to_vector_start = this->layer_data.to_vector_end - this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_WEIGHTS);
	thrust::device_vector<unsigned int>::iterator from_vector_start = this->layer_data.from_vector_end - this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_WEIGHTS);
	thrust::device_vector<WEIGHT_TYPE>::iterator weight_vector_start = this->layer_data.weight_vector_end - this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_WEIGHTS);
	thrust::device_vector<WEIGHT_TYPE>::iterator error_start = this->device_error.end() - this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_NODES);
	//Find the error of the output
	thrust::transform(this->layer_data.device_output_vector_end - this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES),
		this->layer_data.device_output_vector_end,
		start_target,
		this->device_error.end() - this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES),
		functors::transform_functors::find_output_gradiant<WEIGHT_TYPE>()
		);

	

	//Assumes all hidden nodes connect to output node
	//Will change if better idea becomes apparent
	/*thrust::reduce_by_key(
		Special_Iterator::make_repeat_iterator(thrust::make_counting_iterator((int)0), this->_topology->InfoInLayer(0,TopologyBase::OUTPUT_NODES)),
		Special_Iterator::make_repeat_iterator(thrust::make_counting_iterator((int)0), 
		this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_NODES)) + (this->layer_data.weight_vector_end - weight_vector_start),
		
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		
		Special_Iterator::make_repeat_list_iterator(error_start, this->device_error.end() - error_start),
		Special_Iterator::make_transpose_iterator(weight_vector_start,
		this->layer_data.weight_vector_end, this->_topology->InfoInLayer(0, TopologyBase::HIDDEN_NODES),
		this->_topology->InfoInLayer(0, TopologyBase::OUTPUT_NODES))//Transpose_Iterator_End

		)//Tuple End
		),//Zip Iterator end
		functors::transform_functors::find_hidden_node_gradiant<WEIGHT_TYPE>()
		),//Transform end
		
		thrust::make_discard_iterator(),
		this->device_error.begin()
		);*/

	thrust::reduce_

};

//*********************************************
//Clean Up Topology
//Remove the current values
//*********************************************
//Empty the topology entirely
bool RNNTrainer::cleanTrainer(){
	thrust::fill(this->device_error.begin(), this->device_error.end(), (WEIGHT_TYPE)0);
	thrust::fill(this->device_input.begin(), this->device_input.end(), (WEIGHT_TYPE)0);
	thrust::fill(this->device_weight_error.begin(), this->device_weight_error.end(), (WEIGHT_TYPE)0);
	this->_topology->cleanTopology();
	return true;
};

//Clean the network for a run
bool RNNTrainer::emptyTrainer(){
	clean_device::free(this->device_error);
	clean_device::free(this->device_input);
	clean_device::free(this->device_weight_error);
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
};