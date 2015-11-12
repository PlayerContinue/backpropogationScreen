#include "LongTermShortTermNetwork.cuh"

void LongTermShortTermNetwork::count_weights_in_layers(){
	this->count_weights_in_layers(false);
}

void LongTermShortTermNetwork::count_weights_in_layers(bool running){
	this->numberOfWeightsInLayers.clear();
	if (this->number_weights_by_type.size() < this->mBlocksLayers.size()){
		int length = this->number_weights_by_type.size();
		this->number_weights_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_in_layer.resize(this->mBlocksLayers.size() + 1);

		for (; length < this->number_weights_by_type.size(); length++){
			this->number_weights_by_type[length] = vector<int>(5);
			this->number_nodes_by_type[length] = vector<int>(5);
		}
	}
	else if (this->number_weights_by_type.size() > this->mBlocksLayers.size()){
		this->number_weights_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_in_layer.resize(this->mBlocksLayers.size() + 1);
	}

	std::fill(this->number_nodes_in_layer.begin(), this->number_nodes_in_layer.end(), (int)0);
	int i;
	//Count number of weights found in each layer
	for (i = 0; i < this->mBlocksLayers.size(); i++){
		this->numberOfWeightsInLayers.push_back(0);
		//Number Weights
		this->number_weights_by_type[i][MEMORY_CELL] = 0;
		this->number_weights_by_type[i][INPUT_CELL] = 0;
		this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] = 0;
		this->number_weights_by_type[i][OUTPUT_CELL] = 0;
		this->number_weights_by_type[i][FORGET_CELL] = 0;
		//Number Nodes
		this->number_nodes_by_type[i][MEMORY_CELL] = 0;
		this->number_nodes_by_type[i][INPUT_CELL] = 0;
		this->number_nodes_by_type[i][POTENTIAL_MEMORY_CELL] = 0;
		this->number_nodes_by_type[i][OUTPUT_CELL] = 0;
		this->number_nodes_by_type[i][FORGET_CELL] = 0;

		if (i != 0){
			this->number_nodes_in_layer[this->mBlocksLayers.size()] += this->number_nodes_in_layer[i - 1];
		}

		for (int j = 0; j < this->mBlocksLayers[i].size(); j++){
			this->numberOfWeightsInLayers[i] += this->mBlocksLayers[i][j].number_weights;

			//Add the different memory cell types

			this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] += this->mBlocksLayers[i][j].potential_memory_cell_value.size();
			this->number_nodes_in_layer[i] += 1;
			this->number_nodes_by_type[i][POTENTIAL_MEMORY_CELL] += 1;
			if (this->mBlocksLayers[i][j].getTypeOfMemoryBlock() == Memory_Block::memory_block_type::LAYER){
				this->number_weights_by_type[i][OUTPUT_CELL] += this->mBlocksLayers[i][j].output_weights.size();
				this->number_weights_by_type[i][FORGET_CELL] += this->mBlocksLayers[i][j].forget_weights.size();
				this->number_weights_by_type[i][INPUT_CELL] += this->mBlocksLayers[i][j].input_weights.size();

				this->number_nodes_by_type[i][OUTPUT_CELL] += 1;
				this->number_nodes_by_type[i][FORGET_CELL] += 1;
				this->number_nodes_by_type[i][MEMORY_CELL] += 1;
				this->number_nodes_by_type[i][INPUT_CELL] += 1;
				this->number_nodes_in_layer[i] += 4;
				if (running)
				{
					this->numberOfWeightsInLayers[i] += this->mBlocksLayers[i][j].number_memory_cells * NUMBER_MEM_CELL_WEIGHTS;
					this->number_weights_by_type[i][MEMORY_CELL] += NUMBER_WEIGHTS_TO_MEM;
					this->number_weights_by_type[i][OUTPUT_CELL] += 1;
					this->number_weights_by_type[i][FORGET_CELL] += 1;
					this->number_weights_by_type[i][INPUT_CELL] += 1;
					this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] += 1;
				}
			}
		}
	}
	if (i > 0){
		this->number_nodes_in_layer[this->mBlocksLayers.size()] += this->number_nodes_in_layer[i - 1];
	}


}


//***************************
//Modify Structure Of Neuron
//***************************

void LongTermShortTermNetwork::InitialcreateMemoryBlock(int numberMemoryCells){
	//Initialize the layers
	this->mBlocksLayers = vector<vector<Memory_Block>>();
	this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one hidden layer
	this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one output layer
	this->numberOfWeightsInLayers.resize(2);//set the size of the vector to the number of layer
	this->numberOfWeightsInLayers[0] = 0;//Set the number weights to zero
	this->numberOfWeightsInLayers[1] = 0;
	for (int i = 0; i < this->settings.i_output; i++){//Create the output layer
		this->mBlocksLayers[1].push_back(Memory_Block(numberMemoryCells + this->numberNonWeights, numberMemoryCells, this->settings.i_input, Memory_Block::OUTPUT));
		this->numberOfWeightsInLayers[1] += this->mBlocksLayers[1][i].number_weights;
	}



	//Create the rest of the nodes
	this->createMemoryBlock(numberMemoryCells, 0);
}

void LongTermShortTermNetwork::createMemoryBlock(int numberMemoryCells, int layer_num){
	for (int i = 0; i < numberMemoryCells; i++){
		this->mBlocksLayers[layer_num].push_back(Memory_Block(this->settings.i_input));
		this->numberOfWeightsInLayers[layer_num] += this->mBlocksLayers[layer_num][i].number_weights;
	}

}

void  LongTermShortTermNetwork::addPositionOfWeightChange(int start, int start_weights,int start_nodes, int extension,int number_new_weights){
	//Increment the from values to be the new position of the nodes in the array 
	thrust::transform_if(this->GPUMapFrom.begin(), this->GPUMapFrom.end(), this->GPUMapFrom.begin(), _1 + extension, _1 > start_nodes);
	//Increment the to values to be the new position of the nodes in the array 
	thrust::transform(this->GPUMapTo.begin() + start, this->GPUMapTo.end(), this->GPUMapTo.begin() + start, _1 + extension);
	//Increment the positiontosum values to be the new position in the from-to list
	thrust::transform_if(this->positionToSum.begin(), this->positionToSum.end(), this->positionToSum.begin(), _1 + number_new_weights, _1 >= start_weights);
	//Increment the count for sorting reasons
	thrust::transform_if(this->count.begin(), this->count.end(), this->count.begin(), _1 + extension, _1 > start_nodes);
}

template <typename T>
void LongTermShortTermNetwork::addNewSumCount(int start, int end, thrust::device_vector<T> &key, thrust::device_vector<T> &value, thrust::device_vector<T> insert){
	this->addNewSumCount(start, end, 0,0, key, value, insert);
}

template <typename T>
void LongTermShortTermNetwork::addNewSumCount(int start, int end, int add_to_from,int add_to_value, thrust::device_vector<T> &key, thrust::device_vector<T> &value, thrust::device_vector<T> insert){
	//Add the new positions for the sum list
	key.insert(key.end(),
		thrust::make_transform_iterator(
		insert.begin(),
		_1 + add_to_from)
		+ start,
		
		thrust::make_transform_iterator(
		insert.begin(),
		_1 + add_to_from)
		 + end);

	//Add the position of the new weights
	value.insert(value.end(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)start),
		_1 + add_to_value)
		,
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)start),
		_1 + add_to_value) + end - start
		);
}

void LongTermShortTermNetwork::addNonMemoryCellTOGPU(unsigned int &start_new, unsigned int &start_of_weights_to_insert_on, unsigned int &start_of_nodes_to_insert_on,
	unsigned int &number_new_added,
	unsigned int &number_new_added_total,
	unsigned int layer,
	thrust::device_vector<weight_type>::iterator &weight_iterator,
	thrust::device_vector<int>::iterator &int_iterator,
	thrust::device_vector<int> &key,
	thrust::device_vector<int> &value,
	cell_type type,
	Memory_Block::cell_type memory_type
	){
	int size_to_add;
	int start;
	for (unsigned int j = start_new; j < this->mBlocksLayers[layer].size(); j++){
		size_to_add = this->mBlocksLayers[layer][j].weight_lists[memory_type].size() + this->mBlocksLayers[layer][j].number_memory_cells;
		start = start_of_weights_to_insert_on;
		//Insert the new weights
		weight_iterator = this->GPUWeights.begin() + start_of_weights_to_insert_on;
		this->GPUWeights.insert(weight_iterator, 
			this->mBlocksLayers[layer][j].weight_lists[memory_type].begin(), 
			this->mBlocksLayers[layer][j].weight_lists[memory_type].end());
		weight_iterator = this->GPUWeights.begin() + start_of_weights_to_insert_on + this->mBlocksLayers[layer][j].weight_lists[memory_type].size();
		
		//Insert weights to memory cell
		this->GPUWeights.insert(weight_iterator, 
			thrust::make_constant_iterator((int)1),
			thrust::make_constant_iterator((int)1) + this->mBlocksLayers[layer][j].number_memory_cells);

		int_iterator = this->GPUMapFrom.begin() + start_of_weights_to_insert_on;
		
		//Insert where the value should come from
		this->GPUMapFrom.insert(int_iterator,
			this->mBlocksLayers[layer][j].mapFrom.begin(),
			this->mBlocksLayers[layer][j].mapFrom.end());

		int_iterator = this->GPUMapFrom.begin() + start_of_weights_to_insert_on + this->mBlocksLayers[layer][j].weight_lists[memory_type].size();


		//Temporary measure, will be altered to actual value later
		this->GPUMapFrom.insert(int_iterator,
			thrust::make_constant_iterator((int)-(j-start_new)-1),
			thrust::make_constant_iterator((int)-(j-start_new)-1) + this->mBlocksLayers[layer][j].number_memory_cells);

		

		//Insert where the values should go to
		int_iterator = this->GPUMapTo.begin() + start_of_weights_to_insert_on;
		this->GPUMapTo.insert(int_iterator,
			thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1),
			thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1)
			+ this->mBlocksLayers[layer][j].mapFrom.size() + this->mBlocksLayers[layer][j].number_memory_cells);

		weight_iterator = this->GPUBias.begin() + start_of_nodes_to_insert_on;
		this->GPUBias.insert(weight_iterator, this->mBlocksLayers[layer][j].getBias(memory_type));


		
		
		addNewSumCount(start_of_weights_to_insert_on, start_of_weights_to_insert_on + this->mBlocksLayers[layer][j].weight_lists[memory_type].size() + this->mBlocksLayers[layer][j].number_memory_cells, key, value, this->GPUMapFrom);
		
		//Increase the stored values
		number_new_added += 1;//Note a new node has been added
		start_of_weights_to_insert_on += size_to_add;
		start_of_nodes_to_insert_on += 1;
		this->numberOfWeightsInLayers[layer] += this->mBlocksLayers[layer][j].number_memory_cells;
		this->number_weights_by_type[layer][type] += size_to_add;

		this->addPositionOfWeightChange(start_of_weights_to_insert_on, start,start_of_nodes_to_insert_on, number_new_added, size_to_add);
		
		

		this->number_nodes_by_type[layer][type] += 1;
		this->number_nodes_in_layer[layer] += 1;
		this->number_nodes_in_layer[this->number_nodes_in_layer.size() - 1] += 1;
		this->numberOfNodes += 1;
	}

	
	number_new_added_total += number_new_added;//Note n new nodes have been added
	number_new_added = 0;
}


void LongTermShortTermNetwork::addMemoryCellTOGPU(unsigned int &start_new, unsigned int &start_of_weights_to_insert_on, unsigned int &start_of_nodes_to_insert_on,
	unsigned int &number_new_added,
	unsigned int &number_new_added_total,
	unsigned int layer,
	thrust::device_vector<weight_type>::iterator &weight_iterator,
	thrust::device_vector<int>::iterator &int_iterator,
	thrust::device_vector<int> &key,
	thrust::device_vector<int> &value,
	cell_type type,
	Memory_Block::cell_type memory_type){
	int start_weights;
	int size_to_add = NUMBER_WEIGHTS_TO_MEM;
	for (unsigned int j = start_new; j < this->mBlocksLayers[layer].size(); j++){
		//Insert the new weights
		start_weights = start_of_weights_to_insert_on;
		weight_iterator = this->GPUWeights.begin() + start_of_weights_to_insert_on;
		this->GPUWeights.insert(weight_iterator, thrust::make_constant_iterator((int)1), thrust::make_constant_iterator((int)1) + size_to_add);

		int_iterator = this->GPUMapFrom.begin() + start_of_weights_to_insert_on;
		//Insert where the value should come from
		this->GPUMapFrom.insert(int_iterator,
			thrust::make_constant_iterator((int)0),
			thrust::make_constant_iterator((int)0) + size_to_add
			);

		thrust::transform(this->GPUMapFrom.begin() + start_of_weights_to_insert_on-size_to_add, this->GPUMapFrom.begin() + start_of_weights_to_insert_on, this->GPUMapFrom.begin() + start_of_weights_to_insert_on, _1 + 1);

		//Insert where the values should go to
		int_iterator = this->GPUMapTo.begin() + start_of_weights_to_insert_on;
		this->GPUMapTo.insert(int_iterator,
			thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1),
			thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1)
			+ size_to_add);

		
		addNewSumCount(start_of_weights_to_insert_on,start_of_weights_to_insert_on + size_to_add, key, value, this->GPUMapFrom);


		weight_iterator = this->GPUBias.begin() + start_of_nodes_to_insert_on;
		this->GPUBias.insert(weight_iterator, this->mBlocksLayers[layer][j].getBias(memory_type));
	}

	//Increase the stored values
	number_new_added += 1;//Note a new node has been added
	start_of_weights_to_insert_on += size_to_add;
	start_of_nodes_to_insert_on += 1;
	this->numberOfWeightsInLayers[layer] += size_to_add;
	this->number_weights_by_type[layer][type] += size_to_add;
	this->number_nodes_by_type[layer][type] += 1;
	this->number_nodes_in_layer[layer] += 1;
	this->number_nodes_in_layer[this->number_nodes_in_layer.size() - 1] += 1;
	this->numberOfNodes += 1;

	//Increment From to include the new values
this->addPositionOfWeightChange(start_of_weights_to_insert_on, start_weights,start_of_nodes_to_insert_on, number_new_added, size_to_add);
	number_new_added_total += number_new_added;//Note n new nodes have been added
	number_new_added = 0;

}

void LongTermShortTermNetwork::addConnectionToNewCells(int layer,int start_of_output_layer_weights, int add_length, int start_new, thrust::device_vector<weight_type>::iterator &weight_iterator,
	thrust::device_vector<int>::iterator &int_iterator, 
	thrust::device_vector<int>::iterator &to_iterator, 
	thrust::device_vector<int> &key,
	thrust::device_vector<int> &value,
	vector<Memory_Block>* cell_block){
	if (this->mBlocksLayers[layer][0].getTypeOfMemoryBlock() != Memory_Block::memory_block_type::OUTPUT){
		//Add the interventing nodes

		//Add connection to output layer
		weight_iterator = this->GPUWeights.begin() + start_of_output_layer_weights;
		to_iterator = this->GPUMapTo.begin() + start_of_output_layer_weights;
		int_iterator = this->GPUMapFrom.begin() + start_of_output_layer_weights;
		int pos;
		Memory_Block::cell_type type = Memory_Block::cell_type::POTENTIAL_MEMORY_CELL;
		cell_type block_type = POTENTIAL_MEMORY_CELL;
		//Transform the mapfrom values to include the change to the position of the output
		thrust::transform_if(this->GPUMapFrom.begin(), this->GPUMapFrom.end(),this->GPUMapFrom.begin(), _1 + add_length, _1 > this->numberNonWeights + this->numberOfNodes);
		for (unsigned int i = 0; i < (*cell_block).size(); i++){
			weight_iterator += (*cell_block)[i].weight_lists[type].size() - (add_length);
			int_iterator += (*cell_block)[i].weight_lists[type].size() - (add_length);
			to_iterator += (*cell_block)[i].weight_lists[type].size() - (add_length);
			pos = (int)*(to_iterator - 1);

			this->GPUWeights.insert(weight_iterator, (*cell_block)[i].weight_lists[type].end() - (add_length), (*cell_block)[i].weight_lists[type].end());

			

			this->GPUMapFrom.insert(int_iterator,
				thrust::make_transform_iterator((*cell_block)[i].mapFrom.end() - (add_length),
				_1 + this->numberOfNodes + this->numberNonWeights + add_length),
				thrust::make_transform_iterator((*cell_block)[i].mapFrom.end(),
				_1 + this->numberOfNodes + this->numberNonWeights + add_length));

			this->GPUMapTo.insert(to_iterator, thrust::make_constant_iterator((int)pos),
				thrust::make_constant_iterator((int)pos) + add_length
				);

			key.insert(key.end(),
				(*cell_block)[i].mapFrom.end() - (add_length), 
				(*cell_block)[i].mapFrom.end());

			value.insert(value.end(), thrust::make_counting_iterator(to_iterator - this->GPUMapTo.begin()),
				thrust::make_counting_iterator(to_iterator - this->GPUMapTo.begin()) + add_length);

			this->addNewSumCount(to_iterator - this->GPUMapTo.begin(), to_iterator - this->GPUMapTo.begin() + add_length, -(this->numberOfNodes + this->numberNonWeights), 0, key, value, this->GPUMapFrom);

			testing::outputToFile(key, "key", "tests/test3.txt");
			testing::outputToFile(value, "value", "tests/test3.txt");


			weight_iterator += this->mBlocksLayers[layer].size() - start_new;
			int_iterator += this->mBlocksLayers[layer].size() - start_new;
			to_iterator += this->mBlocksLayers[layer].size() - start_new;

			this->numberOfWeightsInLayers[layer + 1]++;
			this->number_weights_by_type[layer][block_type]++;
		}
	}
}

void LongTermShortTermNetwork::addCellToGPU(unsigned int start_new, unsigned int layer){
	if (layer < this->mBlocksLayers.size()){
		thrust::device_vector<weight_type>::iterator weight_iterator;
		thrust::device_vector<int>::iterator int_iterator;
		thrust::device_vector<int>::iterator to_iterator;
		thrust::device_vector<int> SumOrder_Inserts = thrust::device_vector<int>();
		thrust::device_vector<int> CountOrder_Inserts = thrust::device_vector<int>();
		unsigned int start_of_weights_to_insert_on = 0;
		unsigned int start_of_nodes_to_insert_on = 0;
		unsigned int number_new_added = 0; //nodes
		unsigned int number_new_added_total = 0;//nodes
		unsigned int last_added;
		unsigned int start_of_output_layer_nodes = 0;
		unsigned int start_of_output_layer_weights = 0;
		unsigned int add_length = this->mBlocksLayers[layer].size() - start_new;
		vector<Memory_Block>* cell_block;
		for (int i = 0; i < layer; i++){
			start_of_weights_to_insert_on += this->numberOfWeightsInLayers[i];
			start_of_nodes_to_insert_on += this->number_nodes_in_layer[i];
			start_of_output_layer_nodes += this->number_nodes_in_layer[i];
			start_of_output_layer_weights += this->numberOfWeightsInLayers[i];
		}
		


		for (int type_cell = INPUT_CELL, memory_type = Memory_Block::cell_type::INPUT_CELL; type_cell != MEMORY_CELL && memory_type != Memory_Block::cell_type::MEMORY_CELL; type_cell++, memory_type++){
			//Add New Input Nodes
			start_of_weights_to_insert_on += this->number_weights_by_type[layer][type_cell];
			start_of_nodes_to_insert_on += this->number_nodes_by_type[layer][type_cell];
			this->addNonMemoryCellTOGPU(start_new, start_of_weights_to_insert_on, start_of_nodes_to_insert_on,
				number_new_added, number_new_added_total, layer, weight_iterator, int_iterator, CountOrder_Inserts,SumOrder_Inserts, (cell_type)type_cell, (Memory_Block::cell_type)memory_type);
			if (type_cell == OUTPUT_CELL && this->mBlocksLayers[layer][0].getTypeOfMemoryBlock()!=Memory_Block::memory_block_type::OUTPUT){//Add connection to the succeeding layer
				last_added = start_of_nodes_to_insert_on - (add_length)+1;//Position of the start of new outputs
				cell_block = &(this->mBlocksLayers[layer+1]);
				//Add the link to the output to the output row
				for (int i = 0; i < (*cell_block).size(); i++){
					for (int j = 0; j < add_length; j++){
						(*cell_block)[i].addNewConnection(last_added + j);
					}
				}
			}

		}


		
		start_of_weights_to_insert_on += this->number_weights_by_type[layer][MEMORY_CELL];
		start_of_nodes_to_insert_on += this->number_nodes_by_type[layer][MEMORY_CELL];


		

		//Add the memory cell
		addMemoryCellTOGPU(start_new, start_of_weights_to_insert_on, start_of_nodes_to_insert_on,
			number_new_added, number_new_added_total, layer, weight_iterator, int_iterator, CountOrder_Inserts, SumOrder_Inserts, MEMORY_CELL, Memory_Block::cell_type::MEMORY_CELL);
		
		
		//Replace the information pointing to the Memory Cell values
		//Works due to the start_of_nodes_to_insert_on-add_length being the last original memory_cell value. Thus the new memory cell is just x more than that one
		thrust::transform_if(this->GPUMapFrom.begin(), this->GPUMapFrom.end(), this->GPUMapFrom.begin(), (_1 * -1) + (start_of_nodes_to_insert_on), _1 < 0);

		//Change the values of the keys
		thrust::transform_if(CountOrder_Inserts.begin(), CountOrder_Inserts.end(), CountOrder_Inserts.begin(), (_1 * -1) + (start_of_nodes_to_insert_on), _1 < 0);
		


		//Add a new weight to the connect the output of the new nodes to the input of the next layer
		//Get the start of the next layer
		start_of_output_layer_weights += this->numberOfWeightsInLayers[layer];
		start_of_output_layer_nodes += this->number_nodes_in_layer[layer];
			
		//Connect the new nodes to the input of the next layer
		this->addConnectionToNewCells(layer, start_of_output_layer_weights, add_length, start_new, weight_iterator, int_iterator, to_iterator,CountOrder_Inserts,SumOrder_Inserts, cell_block);
		



		cell_block = NULL;
		delete cell_block;//Remove the cell block pointer
		//Increase the node a connection is mapped from if that node is in another layer
		//This works to fix an issue where a new node is added, but it doesn't make much sense
		
		int previous_count_size = this->count.size();
		

	
		//Remove any which point to the Input
		int_iterator = thrust::remove_if(SumOrder_Inserts.begin(), SumOrder_Inserts.end(), CountOrder_Inserts.begin(), _1 < this->numberNonWeights);
		SumOrder_Inserts.erase(int_iterator, SumOrder_Inserts.end());
	
		int_iterator = thrust::remove_if(CountOrder_Inserts.begin(), CountOrder_Inserts.end(), _1 < this->numberNonWeights);
		CountOrder_Inserts.erase(int_iterator, CountOrder_Inserts.end());
		testing::outputToFile(CountOrder_Inserts, "Count", "tests/test3.txt");
		testing::outputToFile(SumOrder_Inserts, "Count", "tests/test3.txt");
		
		//Sort the keys
		thrust::sort_by_key(CountOrder_Inserts.begin(), CountOrder_Inserts.end(), SumOrder_Inserts.begin());
	
		

		//Merge the two lists together
		thrust::merge_by_key(this->count.begin(), 
			this->count.begin() + previous_count_size,
			CountOrder_Inserts.begin(), 
			CountOrder_Inserts.end(), 
			this->positionToSum.begin(), 
			SumOrder_Inserts.begin(), 
			this->GPUOutput_values.begin(), 
			this->device_deltas.begin());

		//Resize the two list to include room for the new values
		this->count.resize(this->count.size() + CountOrder_Inserts.size());
		this->positionToSum.resize(this->positionToSum.size() + SumOrder_Inserts.size());

		thrust::copy_n(this->GPUOutput_values.begin(), this->count.size(), this->count.begin());
		thrust::copy_n(this->device_deltas.begin(), this->positionToSum.size(), this->positionToSum.begin());


		this->GPUPreviousBias.resize(this->GPUBias.size());
		this->GPUPreviousTemp.resize(this->GPUPreviousTemp.size() + number_new_added_total);
		this->GPUPreviousOutput_Values.resize(this->GPUPreviousOutput_Values.size() + number_new_added_total);
		this->GPUPreviousWeights.resize(this->GPUWeights.size());
		this->device_deltas.resize(this->device_deltas.size() + (number_new_added_total*this->total_number_of_unrolled));
		//Add the new outputs to each row
		this->GPUOutput_values.resize(this->GPUOutput_values.size() + (number_new_added_total*this->total_number_of_unrolled));
		//Clean the extra vectors
		clear_vector::free(CountOrder_Inserts);
		clear_vector::free(SumOrder_Inserts);
	}

}

void LongTermShortTermNetwork::addNeuron(int numberNeuronsToAdd){
	int start_value = this->mBlocksLayers[0].size();
	int layer = 0;
	this->createMemoryBlock(numberNeuronsToAdd, layer);

	this->addCellToGPU(start_value, layer);
}



int LongTermShortTermNetwork::decideNodeToAttachTo(){
	vector<int> notFullyConnected = vector<int>();
	int numberMBlocks = this->mBlocksLayers[0].size();
	//Find how many nodes are not fully connected
	for (int i = 0; i < numberMBlocks; i++){
		//A node is considered not fully connected if it is not connected to at least one other memory block
		if (this->mBlocksLayers[0][i].input_weights.size() < this->numberNonWeights + numberMBlocks){
			notFullyConnected.push_back(i);
		}

	}

	if (notFullyConnected.size() > 0){
		//Return a random number in the set of not fully connected nodes
		//It's the number contained in the positionOfLastWeightToNode which is this->settings.i_input less than its actual position
		return notFullyConnected[RandInt(0, notFullyConnected.size() - 1)];
	}
	else{
		//All nodes are fully connected and no new weights can be added
		return -1;
	}
}

int LongTermShortTermNetwork::decideNodeToAttachFrom(int attachTo){
	vector<int> notConnectedTo = vector<int>();
	bool containsValue = false;
	int start = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo - 1] : 0);
	int end = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo] : this->positionOfLastWeightToNode[attachTo]);
	for (unsigned int k = this->numberNonWeights; k < this->bias.size() - this->settings.i_output; k++){
		for (int i = start; i <= end; i++){
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





weight_type LongTermShortTermNetwork::getNewWeight(){
	srand(time(NULL));
	return RandomClamped();
}



void LongTermShortTermNetwork::addWeight(int numberWeightsToAdd){
	int decideTo = this->decideNodeToAttachTo();
	for (int i = 0; i < numberWeightsToAdd; i++){
		if (decideTo != -1){
			this->mBlocksLayers[0][decideTo].addNewConnection(this->settings.i_input + this->mBlocksLayers[0].size(), this->settings.i_input + (2 * this->mBlocksLayers[0].size()));
		}
		else{
			break;
		}
		decideTo = this->decideNodeToAttachTo();
	}

}






