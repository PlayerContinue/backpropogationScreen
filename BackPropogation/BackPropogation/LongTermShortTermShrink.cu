#include "LongTermShortTermNetwork.cuh"



void LongTermShortTermNetwork::removeNeuron(int position, int layer){
	int start_of_weights = 0;
	int start_of_nodes = 0;
	int length_of_nodes_to_skip=0;
	int length_of_weights_to_skip=0;
	int length_of_weight_to_remove;
	Memory_Block *mem = &(this->mBlocksLayers[layer][position]);
	//Contains list of pointers to the weights to be deleted

	//Find the start of the weights in the layer
	for (int i = 0; i < layer; layer++){
		start_of_weights += this->numberOfWeightsInLayers[i];
		start_of_nodes += this->number_nodes_in_layer[i];
	}

	

	//Find the start of the input nodes
	for (int i = 0; i < position; i++){
		start_of_weights += this->mBlocksLayers[layer][i].input_weights.size();
		length_of_nodes_to_skip += this->mBlocksLayers[layer][i].input_weights.size();
		start_of_nodes += 1;//Increment by the input nodes
		length_of_weights_to_skip += 1;
	}

	for (int i = INPUT_CELL; i < MEMORY_CELL; i++){
		start_of_weights += this->number_weights_by_type[layer][i];
		start_of_nodes += this->number_nodes_by_type[layer][i];
	}

	start_of_weights += length_of_weights_to_skip;
	start_of_nodes += length_of_nodes_to_skip;

	//Store the location of the weights/nodes/bias to be removed and store them in a vector
	for (int i = MEMORY_CELL, mem_type = Memory_Block::MEMORY_CELL; i >= INPUT_CELL; i--, mem_type--){
		length_of_weight_to_remove = (mem_type == MEMORY_CELL ? NUMBER_WEIGHTS_TO_MEM : mem->weight_lists[mem_type].size() + 1);
		for (int k = 0; k < length_of_weight_to_remove; k++){
			this->GPUWeights.erase(this->GPUWeights.begin() + start_of_weights);
			this->GPUMapTo.erase(this->GPUMapTo.begin() + start_of_weights);
			this->GPUMapFrom.erase(this->GPUMapFrom.begin() + start_of_weights);
		}

		this->GPUBias.erase(this->GPUBias.begin() + start_of_nodes);


		this->number_weights_by_type[layer][i] -= length_of_weight_to_remove;
		this->number_nodes_by_type[layer][i] -= 1;
		if (i > INPUT_CELL){
			//Increment by the number of nodes/weights of the current type
			start_of_weights -= (this->number_weights_by_type[layer][i - 1] - length_of_weights_to_skip);
			start_of_nodes -= (this->number_nodes_by_type[layer][i - 1] - length_of_nodes_to_skip);
		}
	}



	

}


