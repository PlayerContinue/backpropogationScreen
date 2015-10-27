#include "LongTermShortTermNetwork.cuh"
//#define TRAININGTEST
//#define TRAININGTEST2
//#define DELTA_TEST
//#define AVERAGE_TEST
//#define APPLY_DELTA_BIAS
//#define NVIDA_OUTPUT_TEST
//#define NVIDA_OUTPUT_TEST2
//#define DELTA_MAP_TEST
#define NUMBER_MEMORY_WEIGHTS 3
//*********************
//Training the Network
//*********************



void LongTermShortTermNetwork::InitializeLongShortTermMemory(){
	//Store all the values in the device
	//Will later add option for too little memory
	//Copy the information to the device
	this->UnrollNetwork(3);
	this->RealOutput = device_vector<weight_type>(this->settings.i_output);
	this->host_deltas = host_vector<weight_type>(this->GPUOutput_values.size() - this->numberNonWeights);
	this->device_deltas = device_vector<weight_type>(this->GPUOutput_values.size() - (this->settings.i_backprop_unrolled*this->numberNonWeights));
	this->training_previous_number_rows = this->settings.i_backprop_unrolled;
	this->count_weights_in_layers(true);
}

void LongTermShortTermNetwork::averageWeights(){

	thrust::copy(this->GPUOutput_values.begin() + ((this->numberOfNodes + this->numberNonWeights)* this->training_previous_number_rows-2), this->GPUOutput_values.begin() + ((this->numberOfNodes + this->numberNonWeights)* this->training_previous_number_rows-1), this->GPUOutput_values.begin());//Replace the current input with the output from the last run
	thrust::fill(this->GPUOutput_values.begin() + this->numberOfNodes + this->numberNonWeights, this->GPUOutput_values.end(), (weight_type)0);//Reset the rest of the output values


}


//Add the input
void LongTermShortTermNetwork::setInput(weight_type* in){
	//Place the input into the GPU values matrix
	for (int i = 0; i < this->settings.i_input; i++){
		this->GPUOutput_values[i] = in[i];
	}

}

//Add the input
void LongTermShortTermNetwork::setInput(weight_type** in){
	//Place the input into the GPU values matrix

	for (int j = 0; j < this->settings.i_backprop_unrolled; j++){
		for (int i = 0; i < this->numberNonWeights; i++){
			this->GPUOutput_values[i + (j*(this->numberNonWeights + this->numberOfNodes))] = in[j][i];
		}
	}

}

void LongTermShortTermNetwork::StartTraining(weight_type** in, weight_type** out){

	//Reset the weights to the end of the weights
	this->averageWeights();
	//Set the input values
	this->setInput(in);
	this->training_previous_number_rows = this->settings.i_backprop_unrolled;
	this->LongShortTermMemoryTraining(in, out);
	
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the delta 
	this->FindBackPropDelta(out, 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->FindPreviousBias();
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->FindPreviousWeights();
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), (weight_type)0);
}

void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type** in, weight_type** out){
	
	int start = 1;

	//Get the number of weights in the output layer
	//This is needed because the output layer needs to be used only once, so we need to inform the system which weights to skip


	if (this->newSequence){
		//New sequence, so find the initial first layer
		start = 0;
		
	}
	
	unsigned int number_nodes_to_beginning_of_layer = 0;
	unsigned int number_weights_in_layer = this->GPUWeights.size();
	for (int i = start; i < this->settings.i_backprop_unrolled; i++){
	
		thrust::reduce_by_key(
			this->GPUMapTo.begin(),
			this->GPUMapTo.end(),

			//Multiply the weights x output
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin(),
			thrust::make_permutation_iterator(
			this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer,
			this->GPUMapFrom.begin()
			)
			)
			),
			functors::multiply<weight_type>()
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);

#ifdef NVIDA_OUTPUT_TEST2
		testing::outputToFile<weight_type>(this->GPUWeights, "weights", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(this->GPUBias, "weights", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer,8, "GPU", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "PrevBias-3", "tests/prevbias3.txt");
		
#endif

		if (i > 0){//Only increment it by the number of nodes when working from any layer which is not the initial layer
			//This lets the nodes use the previous layer as their input
			number_nodes_to_beginning_of_layer += this->numberOfNodes + this->numberNonWeights;
		}

		//Add the bias to the current value
		thrust::transform(this->GPUBias.begin(),
			this->GPUBias.end(),
			this->GPUPreviousOutput_Values.begin(),
			this->GPUPreviousOutput_Values.begin(),
			_1 + _2
			);


		//Transfer all values from the current to the next row
		thrust::transform(this->GPUPreviousOutput_Values.begin(),
			this->GPUPreviousOutput_Values.end(),
			this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer + this->numberNonWeights, functors::sigmoid_functor<weight_type>());

	}

	

}





//Find the delta gradiant for each of the "layers" of the network
void LongTermShortTermNetwork::FindBackPropDelta(weight_type** out, int current_layer){

	unsigned int number_weights_in_layer = this->GPUWeights.size();

	unsigned int number_nodes_to_end_of_layer = this->GPUOutput_values.size();

	


	unsigned int delta_next_start = this->numberOfNodes * this->settings.i_backprop_unrolled - this->numberOfNodes;//this->device_deltas.size() - this->numberOfNodes;
	unsigned int delta_next_end = this->numberOfNodes * this->settings.i_backprop_unrolled; //this->device_deltas.size();
	
	//Find the deltas of the output
	for (int i = this->settings.i_backprop_unrolled; i > 0; i--){
		//Store the output into a vector
		//Performed each round due to the output changing each round
		for (unsigned int j = 0; j < this->RealOutput.size(); j++){
			this->RealOutput[j] = out[i-1][j];
		}

		//Find the delta of the output for the current layer
		//output * (1-output) * (target - output)
		thrust::transform(
			this->RealOutput.begin(),
			this->RealOutput.end(),
			this->GPUOutput_values.begin() + ((this->numberOfNodes + this->numberNonWeights) * i) - this->settings.i_output,
			this->device_deltas.begin() + delta_next_end - this->settings.i_output,
			functors::find_output_delta<weight_type>());

		number_nodes_to_end_of_layer -= this->settings.i_output;
#ifdef DELTA_TEST
		testing::outputToFile<weight_type>(this->RealOutput, "output", "tests/testing.txt");
		testing::outputToFile<weight_type>(this->RealOutput, "output", "tests/testing.txt");
		testing::outputToFile<weight_type>(this->GPUOutput_values.begin() + ((this->numberOfNodes + this->numberNonWeights) * i) - this->settings.i_output, this->settings.i_output, "pred_out", "tests/testing.txt");
		testing::outputToFile<weight_type>(this->device_deltas.begin() + delta_next_end - this->settings.i_output,this->settings.i_output, "delta_output", "tests/testing.txt");
		testing::outputToFile<weight_type>(this->device_deltas, "PostOutput", "tests/testing.txt");
		
#endif
		if (i != this->settings.i_backprop_unrolled){//Only perform this action when we've gone past the output layer

			thrust::reduce_by_key(
				this->count.begin(),
				this->count.end(),
				thrust::make_permutation_iterator(
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUWeights.begin(),

				thrust::make_permutation_iterator(
				this->device_deltas.begin() + delta_next_end,
				thrust::make_transform_iterator(
				this->GPUMapTo.begin(),
				_1 - this->numberNonWeights
				)
				)
				)
				),
				functors::multiply<weight_type>()
				),
				this->positionToSum.begin()

				),//Transform End
				thrust::make_discard_iterator(),
				this->GPUPreviousOutput_Values.begin()
				);

#ifdef DELTA_TEST
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "PreTransform", "tests/testing.txt");
#endif
			//Add the bias
			/*thrust::transform(
				this->GPUBias.begin(),
				this->GPUBias.end(),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->device_deltas.begin() + end_of_succeeding_layer - this->numberNonWeights - this->numberOfNodes,
				this->GPUPreviousOutput_Values.begin()

				)
				),
				this->GPUPreviousOutput_Values.begin(),
				functors::add_bias<weight_type>(this->settings.i_backprop_unrolled == i)
				);*/
#ifdef DELTA_TEST
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "PostBias", "tests/testing.txt");
#endif
			//Find the new deltas
			thrust::transform(
				this->device_deltas.begin() + delta_next_start,
				this->device_deltas.begin() + delta_next_start + this->numberOfNodes,
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUOutput_values.begin() + ((this->numberOfNodes + this->numberNonWeights) * (i)) - this->numberOfNodes,
				this->GPUPreviousOutput_Values.begin()
				)), functors::find_non_output_delta<weight_type>()),
				this->device_deltas.begin() + delta_next_start,
				_1 + _2
				);
#ifdef DELTA_TEST
			testing::outputToFile<weight_type>(this->device_deltas, "Device","tests/testing.txt");
#endif
		}
		delta_next_end -= this->numberOfNodes;

		
		delta_next_start -= this->numberOfNodes;

		number_nodes_to_end_of_layer = (number_nodes_to_end_of_layer + this->settings.i_output) - this->numberOfNodes - this->numberNonWeights;
		


	}

}

void LongTermShortTermNetwork::FindPreviousBias(){
	int start_mem_cells = this->GPUBias.size() - this->settings.i_output - this->mBlocksLayers[0].size();
	int end_mem_cells = this->GPUBias.size() - this->settings.i_output;
#ifdef APPLY_DELTA_BIAS

	testing::outputToFile<weight_type>(this->GPUBias, "Bias1", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias1", "tests/prevbias2.txt");
#endif
	//Apply momentum to the bias

	if (this->settings.d_alpha != 0){
		//Apply the alpha to bias
		thrust::transform_if(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousBias.begin(),
			thrust::make_constant_iterator(this->settings.d_alpha),
			this->GPUBias.begin(),
			thrust::make_counting_iterator(int(0))
			)
			)
			,
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousBias.begin(),
			thrust::make_constant_iterator(this->settings.d_alpha) + this->GPUPreviousBias.size(),
			this->GPUBias.begin(),
			thrust::make_counting_iterator(int(0))
			)
			) + this->GPUBias.size(),
			this->GPUBias.begin(),
			functors::multiply_add<weight_type>(),
			functors::compare_between<(unsigned int)3, int>(3, 0, start_mem_cells, end_mem_cells)// #bias-#output-#memory_cells < count < #bias - #output
			);
	}



#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUBias, "Bias2", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias2", "tests/prevbias2.txt");
#endif

	//Retrieve the new previous Bias
	//If I remove memory cells, remove permutation around device_delta so it doesn't skip
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		) + ((this->settings.i_backprop_unrolled - 1) * this->GPUBias.size()),

		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->device_deltas.begin() + this->numberOfNodes,
		functors::multiply_by_constant<weight_type>((weight_type)this->settings.d_beta)
		),

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(this->numberOfNodes*(_1 % (this->settings.i_backprop_unrolled - 1))) + (_1 / (this->settings.i_backprop_unrolled - 1))
		)
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousTemp.begin()
		);

	thrust::transform(this->GPUPreviousBias.begin(),this->GPUPreviousBias.end(),this->GPUPreviousTemp.begin(),this->GPUPreviousBias.begin(), _1 + _2);

#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUBias, "Bias3", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias3", "tests/prevbias2.txt");
#endif
}

void LongTermShortTermNetwork::FindPreviousWeights(){

	

#ifdef TRAININGTEST2
	//thrust::sequence(this->GPUOutput_values.begin(), this->GPUOutput_values.end());
	//thrust::sequence(this->device_deltas.begin(), this->device_deltas.end());
	testing::outputToFile<weight_type>(this->device_deltas, "Delta", "tests/test5.txt");
	testing::outputToFile<weight_type>(this->GPUOutput_values, "Output", "tests/test5.txt");
	
#endif
	int length_between_adds = this->GPUWeights.size() + this->numberNonWeights;
	int number_delta_between_add = this->GPUMapTo.size();
#ifdef TRAININGTEST2
	testing::outputToFile<weight_type>(
		thrust::make_permutation_iterator(
		this->device_deltas.begin() + this->numberOfNodes,
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapTo.begin(),
		_1 - this->numberNonWeights
		),//End Transform Iterator (States to start with the 2nd layer)

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)//End Transform

		),//End Perm
		thrust::make_counting_iterator((int)0)
		)//End Tuple
		),//END zip
		functors::extend_value<int>(number_delta_between_add, 0, this->numberOfNodes, false)
		)//End of transform iterator
		), (this->settings.i_backprop_unrolled - 1)  * this->GPUWeights.size(),
		"Intermediate1",

		"tests/test5.txt"
		);

	testing::outputToFile<weight_type>(thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapFrom.begin(),
		_1
		),//End Transform Iterator (States to start with the 2nd layer)
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),//End Perm
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes + this->numberNonWeights, false)//Increase whenever the counter reaches the end
		)//End of transform iterator
		), (this->settings.i_backprop_unrolled - 1)  * this->GPUWeights.size(),
		"Intermediate2",

		"tests/test5.txt"
		);

	testing::outputToFile<weight_type>(thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(

		thrust::make_permutation_iterator(
		this->device_deltas.begin() + this->numberOfNodes,
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapTo.begin(),
		_1 - this->numberNonWeights
		),//End Transform Iterator (States to start with the 2nd layer)

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)//End Transform

		),//End Perm
		thrust::make_counting_iterator((int)0)
		)//End Tuple
		),//END zip
		functors::extend_value<int>(number_delta_between_add, 0, this->numberOfNodes, false)
		)//End of transform iterator
		),

		//Weight_permutation
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapFrom.begin(),
		_1
		),//End Transform Iterator (States to start with the 2nd layer)
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),//End Perm
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes + this->numberNonWeights, false)//Increase whenever the counter reaches the end
		)//End of transform iterator
		)//End Permutation Iterator


		)
		),
		functors::find_previous_weight<weight_type>(this->settings.d_beta)
		),//End Transform Iterator
		thrust::make_transform_iterator(//Weight 1 - 0, Weight 2-0,....
		thrust::make_counting_iterator((int)0),
		(this->GPUMapTo.size()*(_1 % (this->settings.i_backprop_unrolled - 1))) + (_1 / (this->settings.i_backprop_unrolled - 1))
		)

		), (this->settings.i_backprop_unrolled - 1) * this->GPUWeights.size(),
		"Intermediate",

		"tests/test5.txt"
		
		);
	

#endif

	//Apply the alpha
	thrust::transform_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_constant_iterator(this->settings.d_alpha),
		this->GPUPreviousWeights.begin(),
		this->GPUWeights.begin()
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_constant_iterator(this->settings.d_alpha),
		this->GPUPreviousWeights.end(),
		this->GPUWeights.end()
		)
		),
		this->GPUWeights.begin(),
		functors::multiply_add<weight_type>(),
		functors::compare_two<(unsigned int)1, weight_type>(5, 1)
		);

	//Find the previous weights
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		) + (this->settings.i_backprop_unrolled - 1)  * this->GPUWeights.size(),

		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(

		thrust::make_permutation_iterator(
		this->device_deltas.begin() + this->numberOfNodes,
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapTo.begin(),
		_1 - this->numberNonWeights
		),//End Transform Iterator (States to start with the 2nd layer)

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)//End Transform

		),//End Perm
		thrust::make_counting_iterator((int)0)
		)//End Tuple
		),//END zip
		functors::extend_value<int>(number_delta_between_add, 0, this->numberOfNodes, false)
		)//End of transform iterator
		),

		//Weight_permutation
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(//Add the number of nodes when the end of the mapto is reached

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->GPUMapFrom.begin(),
		_1
		),//End Transform Iterator (States to start with the 2nd layer)
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),//End Perm
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes + this->numberNonWeights, false)//Increase whenever the counter reaches the end
		)//End of transform iterator
		)//End Permutation Iterator


		)
		),
		functors::find_previous_weight<weight_type>(this->settings.d_beta)
		),//End Transform Iterator
		thrust::make_transform_iterator(//Weight 1 - 0, Weight 2-0,....
		thrust::make_counting_iterator((int)0),
		(this->GPUMapTo.size()*(_1 % (this->settings.i_backprop_unrolled - 1))) + (_1 / (this->settings.i_backprop_unrolled - 1))
		)

		),//End Permutation Iterator
		thrust::make_discard_iterator(),
		this->GPUPreviousTemp.begin()
		);

	thrust::transform(this->GPUPreviousWeights.begin(),this->GPUPreviousWeights.end(),this->GPUPreviousTemp.begin(),this->GPUPreviousWeights.begin(), _1 + _2);


#ifdef TRAININGTEST2
	testing::outputToFile<weight_type>(this->device_deltas, "Delta2", "tests/test5.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousWeights, "PrevGPUVal", "tests/test5.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousTemp, "PreGPUTemp", "tests/test5.txt");
#endif

}

//Apply the error
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){

#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-1", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUBias, "Bias-1", "tests/prevbias2.txt");


#endif

#ifdef DELTA_MAP_TEST 
	testing::outputToFile<int>(this->GPUMapFrom, "From", "tests/prevbias3.txt");
	testing::outputToFile<int>(this->GPUMapTo, "To", "tests/prevbias3.txt");
#endif
	this->ApplyErrorToBias();


	thrust::transform_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUWeights.begin(),
		this->GPUPreviousWeights.begin()
		
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUWeights.end(),
		this->GPUPreviousWeights.end()
		
		)
		),
		this->GPUWeights.begin(),
		functors::add_and_store<weight_type>(this->settings.i_backprop_unrolled-1),
		functors::compare_two<(unsigned int)0, weight_type>(5, 1));

#ifdef APPLY_DELTA_BIAS

	testing::outputToFile<weight_type>(this->GPUBias, "Bias-5", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-5", "tests/prevbias2.txt");
#endif



}

void LongTermShortTermNetwork::ApplyErrorToBias(){
	int start_mem_cells = this->GPUBias.size() - this->settings.i_output - this->mBlocksLayers[0].size();
	int end_mem_cells = this->GPUBias.size() - this->settings.i_output;
	//Apply the delta to the bias


	//Apply the error
	thrust::transform_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.begin(),
		this->GPUPreviousBias.begin(),
		thrust::make_counting_iterator((int)0)
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.begin(),
		this->GPUPreviousBias.begin(),
		thrust::make_counting_iterator((int)0)
		)
		) + this->GPUBias.size(),
		this->GPUBias.begin(),
		functors::add_and_store<weight_type>(this->settings.i_backprop_unrolled - 1),
		functors::compare_between<(unsigned int)2, int>(3, 0, start_mem_cells, end_mem_cells)
		);

#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUBias, "Bias4", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias4", "tests/prevbias2.txt");
#endif
}

//*********************
//Run The Network
//*********************


void LongTermShortTermNetwork::InitializeLongShortTermMemoryForRun(){
	//Form the delta objects
	this->CopyToDevice();
	this->moveBiasToGPU(false);//Don't create a previous_bias
	this->count_weights_in_layers();
}

device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in, run_type type){
	switch (type){
	case run_type::WITHOUT_MEMORY_CELLS:
		return this->runNetwork(in, 0, this->newSequence);
	case run_type::WITH_MEMORY_CELLS:
		//Pass the number of memory cells into the function so they can be skipped when finding the output
		return this->runNetwork(in, 1, this->newSequence);
	default:
		return this->runNetwork(in, 0, this->newSequence);
	}
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){
	return this->runNetwork(in,0,this->newSequence);
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in, int number_of_extra_weights, bool &newSequence){
	if (newSequence){
		this->runNetwork(in, number_of_extra_weights);
		newSequence = false;
	}

	return this->runNetwork(in, number_of_extra_weights);
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in,int number_of_extra_weights){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	//Number mBlocks in previous layer
	unsigned int previousnumberMBlocks = 0;
	unsigned int numberBlocksToLayer = 0;
	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);

	//unsigned int numberBias = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i+=2){
		
		if (i != 0){
			previousnumberMBlocks += this->numberOfWeightsInLayers[i - 1] + this->numberOfWeightsInLayers[i-2] + number_of_extra_weights;
			numberBlocksToLayer += numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		

		
#ifdef TRAININGTEST

		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPREV_START", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "GPU_START", "tests/PrevOut1.txt");
		}
		
#endif
		
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous
		thrust::reduce_by_key(
			this->GPUMapTo.begin() + previousnumberMBlocks,
			//Start at the beginning of the previous layer
			this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i], // End at the the number of nodes before the current layer + the number of nodes in the current layer
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin() + previousnumberMBlocks,//Start from the beginning of the layer
			thrust::make_permutation_iterator(//Permute the output values such that they start at the correct position
			this->GPUOutput_values.begin(),
			this->GPUMapFrom.begin() + previousnumberMBlocks
			)
			)
			),
			functors::multiply_or_return_zero<weight_type,1>() //Multiply the two together
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);
#ifdef TRAININGTEST
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPreBias1", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "GPUPreBias2", "tests/PrevOut1.txt");
			testing::outputToFile<weight_type>(this->GPUMapTo.begin() + previousnumberMBlocks, this->numberOfWeightsInLayers[i], "Map0", "tests/map1.txt");
			
		}
		
#endif
		//Add Bias to the hidden layers
		thrust::transform(
			this->GPUBias.begin() + numberBlocksToLayer,
			this->GPUBias.begin() + numberBlocksToLayer + (numberMBlocks*4),
			this->GPUPreviousOutput_Values.begin(),
			this->GPUPreviousOutput_Values.begin(),
			_1 + _2
			);


#ifdef TRAININGTEST
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPostBias", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "GPUPostBias", "tests/PrevOut1.txt");
			testing::outputToFile<weight_type>(this->GPUBias.begin() + numberBlocksToLayer, (numberMBlocks * 4), "Map1", "tests/map1.txt");
		}
		
#endif
		//Create a input/output/forget/potential_memory_cell_values/memory_cell_value value
		//Essentially run the gate and get the output value
		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin(), //input values
			this->GPUPreviousOutput_Values.begin() + numberMBlocks,//output values
			this->GPUPreviousOutput_Values.begin() + (2 * numberMBlocks),//forget values
			this->GPUPreviousOutput_Values.begin() + (3 * numberMBlocks),//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + (4 * numberMBlocks),
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 4),//Old Memory Cell Values
			this->GPUOutput_values.begin() + numberBlocksToLayer, //Old Input
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 2), //Old Forget
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 3), //Old Potential
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 1)
			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + numberMBlocks, //input values
			this->GPUPreviousOutput_Values.begin() + (2 * numberMBlocks),//output values
			this->GPUPreviousOutput_Values.begin() + (3 * numberMBlocks),//forget values
			this->GPUPreviousOutput_Values.begin() + (4 * numberMBlocks),//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + (5 * numberMBlocks),
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 5),//Memory Cell Values
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 1), //Old Input
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 3), //Old Forget
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 4), //Old Potential
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 2)
			)),
			functors::run_memory_block_functon<weight_type>());
		
#ifdef TRAININGTEST
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUMid1", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "GPUMid1", "tests/PrevOut1.txt");
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values.begin() + numberMBlocks, numberMBlocks, "Map1", "tests/map1.txt");
		}
		
#endif
			thrust::reduce_by_key(
				this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i],
				this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i] + this->numberOfWeightsInLayers[i + 1],
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUWeights.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i],
				thrust::make_permutation_iterator(
				this->GPUOutput_values.begin(),
				thrust::make_transform_iterator(
				this->GPUMapFrom.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i],
				_1
				)
				)
				)
				),
				functors::multiply<weight_type>()
				),
				thrust::make_discard_iterator(),
				this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5)
				);
		

#ifdef TRAININGTEST
		
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUOut", "tests/PrevOut1.txt");
			testing::outputToFile<weight_type>(this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i], this->numberOfWeightsInLayers[i + 1], "Map2", "tests/map1.txt");
			testing::outputToFile<weight_type>(this->GPUMapFrom.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i], this->numberOfWeightsInLayers[i + 1], "From1", "tests/map1.txt");
		}
		
#endif
		
		//Add the bias to the output
		thrust::transform(
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5) + this->settings.i_output,
			this->GPUBias.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			_1 + _2
			);
		
#ifdef TRAININGTEST
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "outpostbias", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "outpostbias", "tests/PrevOut1.txt");
		}
		
#endif
		
		thrust::transform(
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5) + this->settings.i_output,
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			functors::sigmoid_functor<weight_type>()
			);
		
#ifdef TRAININGTEST
		
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "Final", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "Final", "tests/PrevOut1.txt");
		}
		

#endif
		
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.begin() + (numberMBlocks * 5) + this->settings.i_output, this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer);
		
		thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
		
#ifdef TRAININGTEST
		if (number_of_extra_weights != 0){
			testing::outputToFile(this->GPUPreviousOutput_Values, "GPUNew", "tests/PrevOut1.txt");
			testing::outputToFile(this->GPUOutput_values, "GPUNew", "tests/PrevOut1.txt");
		}
		
#endif
	}

	


	

	thrust::copy(this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + this->numberOfNodes - this->settings.i_output, this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + this->numberOfNodes, toReturn.begin());

	return toReturn;
}
