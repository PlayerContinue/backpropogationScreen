#include "LongTermShortTermNetwork.cuh"



void LongTermShortTermNetwork::removeNeuron(int position, int layer){
	//The start of the weights to be removed
	int start_of_weights = 0;
	//The start of the nodes to be removed
	int start_of_nodes = 0;
	//the start of weights in the next layer
	int start_of_weights_in_next_layer;
	//The start of nodes in the next layer
	int start_of_nodes_in_next_layer;
	//The number of nodes from the beginning of the next type of nodes to the node to be removed
	int length_of_nodes_to_skip=0;
	//The number of weights from the beginning of the next type of weights to the weights to be removed
	int length_of_weights_to_skip=0;
	//The number of weights to remove
	int length_of_weight_to_remove;
	thrust::device_vector<int>::iterator remove_iterator;
	Memory_Block *mem = &(this->mBlocksLayers[layer][position]);
	//Contains list of pointers to the weights to be deleted

	//Find the start of the weights in the layer
	for (int i = 0; i < layer; i++){
		start_of_weights += this->numberOfWeightsInLayers[i];
		start_of_nodes += this->number_nodes_in_layer[i];
	}

	start_of_nodes_in_next_layer = start_of_nodes + this->number_nodes_in_layer[layer];
	start_of_weights_in_next_layer = start_of_weights + this->numberOfWeightsInLayers[layer];

	//Find the start of the input nodes
	for (int i = 0; i < position; i++){
		length_of_weights_to_skip += this->mBlocksLayers[layer][i].input_weights.size() + 1;
		length_of_nodes_to_skip += 1;
	}

	//Adding nodes to start of memory cells
	for (int i = INPUT_CELL; i < MEMORY_CELL; i++){
		if (i == OUTPUT_CELL){
			//Remove the connections to the next layer
			this->removeOutputConnection(position, layer, 
				start_of_nodes_in_next_layer,
				start_of_weights_in_next_layer, 
				start_of_nodes,
				start_of_weights,
				this->mBlocksLayers[layer][position].input_weights.size() * 2);//Number of weights to delete is 2 times the number of weights
		}
		start_of_weights += this->number_weights_by_type[layer][i];
		start_of_nodes += this->number_nodes_by_type[layer][i];	
	}

	


	//Move to the start of the memory cell weights and nodes
	start_of_weights += NUMBER_WEIGHTS_TO_MEM * length_of_nodes_to_skip;
	start_of_nodes += length_of_nodes_to_skip;

	
	

	//Store the location of the weights/nodes/bias to be removed and store them in a vector
	for (int i = MEMORY_CELL, mem_type = Memory_Block::MEMORY_CELL; i >= INPUT_CELL; i--, mem_type--){
		length_of_weight_to_remove = (mem_type == MEMORY_CELL ? NUMBER_WEIGHTS_TO_MEM : mem->weight_lists[mem_type].size() + 1);
		
		
		
		this->GPUWeights.erase(this->GPUWeights.begin() + start_of_weights, this->GPUWeights.begin() + start_of_weights + length_of_weight_to_remove);
		this->GPUMapTo.erase(this->GPUMapTo.begin() + start_of_weights, this->GPUMapTo.begin() + start_of_weights + length_of_weight_to_remove);
		this->GPUMapFrom.erase(this->GPUMapFrom.begin() + start_of_weights, this->GPUMapFrom.begin() + start_of_weights + length_of_weight_to_remove);
		
		//Remove the bias
		this->GPUBias.erase(this->GPUBias.begin() + start_of_nodes);

		this->number_weights_by_type[layer][i] -= length_of_weight_to_remove;
		this->number_nodes_by_type[layer][i] -= 1;
		this->numberOfNodes--;
		this->number_nodes_in_layer[layer]--;
		this->number_nodes_in_layer[this->number_nodes_in_layer.size() - 1]--;
		this->numberOfWeightsInLayers[layer] -= length_of_weight_to_remove;

		//Remove weights going from the deleted nodes
		remove_iterator = thrust::remove_if(this->positionToSum.begin(), this->positionToSum.end(), this->count.begin(), _1 == start_of_nodes + this->numberNonWeights);
		this->positionToSum.erase(remove_iterator, this->positionToSum.end());

		remove_iterator = thrust::remove_if(this->count.begin(), this->count.end(), _1 == start_of_nodes + this->numberNonWeights);
		this->count.erase(remove_iterator, this->count.end());

		//Remove weights going to the deleted node
		remove_iterator = thrust::remove_if(this->count.begin(), this->count.end(), this->positionToSum.begin(), _1 == start_of_weights);
		this->count.erase(remove_iterator, this->count.end());

		remove_iterator = thrust::remove_if(this->positionToSum.begin(), this->positionToSum.end(), _1 == start_of_weights);
		this->positionToSum.erase(remove_iterator, this->positionToSum.end());
		
		//Transform the values
		thrust::transform_if(this->GPUMapTo.begin() + start_of_weights, this->GPUMapTo.end(), this->GPUMapTo.begin() + start_of_weights, _1 - 1, _1 >= this->numberNonWeights);
		thrust::transform_if(this->GPUMapFrom.begin(), this->GPUMapFrom.end(), this->GPUMapFrom.begin(), _1 - 1, _1 > start_of_nodes + this->numberNonWeights);
		thrust::transform_if(this->positionToSum.begin(), this->positionToSum.end(), this->positionToSum.begin(), _1 - length_of_weight_to_remove, _1 >= start_of_weights);
		thrust::transform_if(this->count.begin(), this->count.end(), this->count.begin(), _1 - 1, _1 > start_of_nodes + this->numberNonWeights);



		if (i > INPUT_CELL){
			//Increment by the number of nodes/weights of the current type
			start_of_weights -= (this->number_weights_by_type[layer][i - 1] + length_of_weight_to_remove - length_of_weights_to_skip);
			start_of_nodes -= (this->number_nodes_by_type[layer][i - 1]);
		}
	}

	//Remove the Cell
	this->mBlocksLayers[layer].erase(this->mBlocksLayers[layer].begin() + position);

	//Resize the output
	this->device_deltas.resize(this->device_deltas.size() - ((MEMORY_CELL - INPUT_CELL)*this->total_number_of_unrolled));
	this->GPUOutput_values.resize(this->GPUOutput_values.size() - ((MEMORY_CELL - INPUT_CELL)*this->total_number_of_unrolled));
	this->GPUPreviousBias.resize(this->GPUBias.size());
	this->GPUPreviousOutput_Values.resize(this->GPUPreviousOutput_Values.size() - (MEMORY_CELL - INPUT_CELL));
	this->GPUPreviousWeights.resize(this->GPUWeights.size());
	this->GPUPreviousTemp.resize(((this->GPUPreviousBias.size() > this->GPUPreviousWeights.size()) ? this->GPUPreviousBias.size() : this->GPUPreviousWeights.size()));
	

}

void LongTermShortTermNetwork::removeOutputConnection(int position, int previous_layer,
	int start_of_nodes_in_layer, int start_of_weights_in_layer, int start_of_nodes, int start_of_weights){
	this->removeOutputConnection(position, previous_layer, start_of_nodes_in_layer, start_of_weights_in_layer, start_of_nodes, start_of_weights,this->settings.i_output);
}

void LongTermShortTermNetwork::removeOutputConnection(int position, int previous_layer, 
	int start_of_nodes_in_layer, int start_of_weights_in_layer, int start_of_nodes,int number_of_weights,  int number_nodes_to_remove){
	thrust::device_vector<weight_type>::iterator remove_weight_iterator;
	thrust::device_vector<int>::iterator remove_int_iterator;
	thrust::device_vector<int>::reverse_iterator reverse_int_iterator;
	//+1 is to skip over the node which will be deleted
	thrust::transform_if(this->GPUMapFrom.begin() + start_of_weights_in_layer,
		this->GPUMapFrom.begin() + start_of_weights_in_layer + this->numberOfWeightsInLayers[previous_layer + 1],
		this->GPUMapFrom.begin() + start_of_weights_in_layer,
		_1 - number_nodes_to_remove,
		_1 > start_of_nodes && _1 < this->numberOfNodes + this->numberNonWeights 
		|| _1 >= start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights);

	reverse_int_iterator = thrust::find_if(
		this->GPUMapFrom.rbegin(),
		this->GPUMapFrom.rend(),
		_1 == start_of_nodes + this->numberNonWeights 
		|| _1 == start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights - number_nodes_to_remove);
	


	//Lower the position of the weights
	do{
		
		
		int temp = (this->GPUMapFrom.size() - (reverse_int_iterator - this->GPUMapFrom.rbegin())) - 1;
		if (reverse_int_iterator != this->GPUMapFrom.rend()){
			//Only perform the operation if the end has not been reached
			thrust::transform_if(this->positionToSum.begin(), 
				this->positionToSum.end(), 
				this->positionToSum.begin(), 
				_1 - 1,
				_1 > (this->GPUMapFrom.size() - (reverse_int_iterator - this->GPUMapFrom.rbegin())) - 1);
		}

		reverse_int_iterator = thrust::find_if(
			reverse_int_iterator+1,
			this->GPUMapFrom.rend(),
			_1 == start_of_nodes + this->numberNonWeights
			|| _1 == start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights - number_nodes_to_remove);

	} while (reverse_int_iterator != this->GPUMapFrom.rend());

	
	//testing::outputToFile(this->positionToSum, "GPUFROM1", "tests/testing1.txt");

	//Remove the Weights 
	remove_weight_iterator = thrust::remove_if(this->GPUWeights.begin() + start_of_nodes_in_layer, this->GPUWeights.end(),
		this->GPUMapFrom.begin() + start_of_nodes_in_layer,
		_1 == start_of_nodes + this->numberNonWeights || _1 == start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights - number_nodes_to_remove);

	this->GPUWeights.erase(remove_weight_iterator, this->GPUWeights.end());

	//Remove the pointer to To
	remove_int_iterator = thrust::remove_if(
		this->GPUMapTo.begin() + start_of_nodes_in_layer,
		this->GPUMapTo.end(),
		this->GPUMapFrom.begin() + start_of_nodes_in_layer,
		_1 == start_of_nodes + this->numberNonWeights || _1 == start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights - number_nodes_to_remove);

	//Subtract the number of weights removed
	this->numberOfWeightsInLayers[previous_layer + 1] -= (this->GPUMapTo.end() - remove_int_iterator);

	for (int i = INPUT_CELL; i < MEMORY_CELL; i++){
		if (this->number_weights_by_type[previous_layer + 1][i] > 0){
			if (this->mBlocksLayers[previous_layer + 1][0].type == Memory_Block::LAYER){
				this->number_weights_by_type[previous_layer + 1][i] -= (this->GPUMapTo.end() - remove_int_iterator) / 4;
			}
			else{
				this->number_weights_by_type[previous_layer + 1][i] -= this->settings.i_output;
			}
		}
	}

	this->GPUMapTo.erase(remove_int_iterator, this->GPUMapTo.end());

	//Remove the pointer to From
	remove_int_iterator = thrust::remove_if(
		this->GPUMapFrom.begin() + start_of_nodes_in_layer,
		this->GPUMapFrom.end(),
		_1 == start_of_nodes + this->numberNonWeights || _1 == start_of_nodes + this->numberNonWeights + this->numberOfNodes + this->numberNonWeights - number_nodes_to_remove);

	this->GPUMapFrom.erase(remove_int_iterator, this->GPUMapFrom.end());

	//Remove the connection in the node
	for (unsigned int i = 0; i < this->mBlocksLayers[previous_layer + 1].size(); i++){
		
			//Remove the connection
			this->mBlocksLayers[previous_layer + 1][i].removeConnection(
				thrust::find_if(host,this->mBlocksLayers[previous_layer + 1][i].mapFrom.begin(),
				this->mBlocksLayers[previous_layer + 1][i].mapFrom.end(),
				_1 == start_of_nodes + this->numberNonWeights + position
				) 
				- this->mBlocksLayers[previous_layer + 1][i].mapFrom.begin());

		
	}

}
