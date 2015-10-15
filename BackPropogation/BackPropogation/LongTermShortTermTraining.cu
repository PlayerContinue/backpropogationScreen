#include "LongTermShortTermNetwork.cuh"
//#define TRAININGTEST
//#define TRAININGTEST2
//#define DELTA_TEST
//#define AVERAGE_TEST
//#define APPLY_DELTA_BIAS
//#define NVIDA_OUTPUT_TEST
//#define NVIDA_OUTPUT_TEST2
//#define DELTA_MAP_TEST
#define NUMBER_MEMORY_WEIGHTS 4
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
}

void LongTermShortTermNetwork::averageWeights(){

#ifdef AVERAGE_TEST
	testing::outputToFile<weight_type>(this->GPUOutput_values, "initialOutput", "tests/Testing6.txt");
#endif
	/*thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1/this->settings.i_backprop_unrolled
		),

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1 / this->settings.i_backprop_unrolled
		) + this->GPUOutput_values.size() - this->numberNonWeights,

		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin() + this->numberNonWeights,
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->numberOfNodes)* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)

		),

		thrust::make_discard_iterator(),

		this->GPUOutput_values.begin() + this->numberNonWeights
		);*/

#ifdef AVERAGE_TEST
	testing::outputToFile<weight_type>(this->GPUOutput_values, "outputbeforetransform", "tests/Testing6.txt");
#endif

	//Find the average from the sum
	/*thrust::transform(
		this->GPUOutput_values.begin() + this->numberNonWeights,
		this->GPUOutput_values.begin() + this->numberNonWeights + this->numberOfNodes,
		this->GPUOutput_values.begin() + this->numberNonWeights,
		_1 / this->settings.i_backprop_unrolled

		);*/
	int i = 0;
	thrust::copy(this->GPUOutput_values.end() - this->numberOfNodes, this->GPUOutput_values.end(), this->GPUOutput_values.begin() + this->numberNonWeights);//Replace the current input with the output from the last run
#ifdef AVERAGE_TEST
	testing::outputToFile<weight_type>(this->GPUOutput_values, "outputaftertransform", "tests/Testing6.txt");
#endif
	thrust::fill(this->GPUOutput_values.begin() + this->numberOfNodes + this->numberNonWeights, this->GPUOutput_values.end(), (weight_type)0);

#ifdef AVERAGE_TEST
	testing::outputToFile<weight_type>(this->GPUOutput_values, "outputAfterFill", "tests/Testing6.txt");
#endif

}

void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type** in, weight_type** out){
	//Reset the weights to the end of the weights
	this->averageWeights();

	//Get the number of weights in the output layer
	//This is needed because the output layer needs to be used only once, so we need to inform the system which weights to skip


	//Set the input values
	this->setInput(in);
	unsigned int number_nodes_to_beginning_of_layer = 0;
	unsigned int number_weights_in_layer = this->GPUWeights.size();
	for (int i = 1; i < this->settings.i_backprop_unrolled; i++){
	
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
		testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "PrevBias-3", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(thrust::make_permutation_iterator(
			this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer,
			this->GPUMapFrom.begin()
			),this->GPUMapTo.size(), "PrevBias-1", "tests/prevbias3.txt");
		cout << "2";
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

#ifdef NVIDA_OUTPUT_TEST

		cout << "3";
#endif

	}

	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the delta 
	this->FindBackPropDelta(out, 0);

}



//Find the delta gradiant for each of the "layers" of the network
void LongTermShortTermNetwork::FindBackPropDelta(weight_type** out, int current_layer){
#ifdef NVIDA_OUTPUT_TEST

	cout << "4";
#endif
	

	unsigned int number_weights_in_layer = this->GPUWeights.size();

	unsigned int number_nodes_to_end_of_layer = this->GPUOutput_values.size();

	
	unsigned int number_to_output_of_layer;

	unsigned int delta_next_start = this->device_deltas.size() - this->numberOfNodes;
	unsigned int delta_next_end = this->device_deltas.size();
	
	//Find the deltas of the output
	for (int i = this->settings.i_backprop_unrolled; i > 0; i--){
		//Store the output into a vector
		//Performed each round due to the output changing each round
		for (unsigned int j = 0; j < this->RealOutput.size(); j++){
			this->RealOutput[j] = out[i-1][j];
		}
		number_to_output_of_layer = this->settings.i_output + ((this->numberOfNodes + this->numberNonWeights) * (this->settings.i_backprop_unrolled - i));//Number nodes from end of list to end of the layer
		//Find the delta of the output for the current layer
		//output * (1-output) * (target - output)
		thrust::transform(
			this->RealOutput.begin(),
			this->RealOutput.end(),
			this->GPUOutput_values.end() - number_to_output_of_layer,
			this->device_deltas.begin() + delta_next_end - this->settings.i_output,
			functors::find_output_delta<weight_type>());

		number_nodes_to_end_of_layer -= this->settings.i_output;
#ifdef DELTA_TEST
		testing::outputToFile<weight_type>(this->GPUBias.begin(), 1, "test", "test11");
		testing::outputToFile<weight_type>(this->device_deltas, "PostOutput", "tests/testing.txt");
		//thrust::copy(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + this->device_deltas.size(), this->device_deltas.begin());

		/*testing::outputToFile<weight_type>(
			thrust::make_permutation_iterator(
			this->device_deltas.begin() + end_of_succeeding_layer - number_from_delta_start - this->numberOfNodes,
			thrust::make_transform_iterator(
			this->GPUMapTo.begin(),
			_1 - this->numberNonWeights
			)
			), this->GPUWeights.size(), "Results2", "tests/testing.txt");


			testing::outputToFile<weight_type>(
			thrust::make_permutation_iterator(
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin(),

			thrust::make_permutation_iterator(
			this->device_deltas.begin() + end_of_succeeding_layer - number_from_delta_start - this->numberOfNodes,
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

			), this->positionToSum.size(), "Results", "tests/testing.txt");*/
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
			int k = 0;
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
				this->GPUOutput_values.end() - number_to_output_of_layer + this->settings.i_output - this->numberOfNodes,
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



//Apply the error
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){
#ifdef NVIDA_OUTPUT_TEST

	cout << "5";
#endif
#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-1", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUBias, "Bias-1", "tests/prevbias2.txt");


#endif

#ifdef DELTA_MAP_TEST 
	testing::outputToFile<int>(this->GPUMapFrom, "From", "tests/prevbias3.txt");
	testing::outputToFile<int>(this->GPUMapTo, "To", "tests/prevbias3.txt");
#endif
	this->ApplyErrorToBias();


	thrust::device_vector<weight_type> tempPrevBias = thrust::device_vector<weight_type>(this->GPUPreviousBias);
	thrust::device_vector<weight_type> tempBias = thrust::device_vector<weight_type>(this->GPUBias);
	thrust::device_vector<int> tempMapFrom = thrust::device_vector<int>(this->GPUMapFrom);
#ifdef TRAININGTEST2
	//testing::outputToFile<weight_type>(this->device_deltas, "Delta");
	//testing::outputToFile<weight_type>(this->GPUOutput_values, "Output");
#endif
	int length_between_adds = this->GPUWeights.size() + this->numberNonWeights;
	int number_delta_between_add = this->GPUMapTo.size();
#ifdef TRAININGTEST

	testing::outputToFile<weight_type>(
		thrust::make_permutation_iterator(
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
		)
		)

		//Increase whenever the counter reaches the end
		, thrust::make_transform_iterator(//Weight 1 - 0, Weight 2-0,....
		thrust::make_counting_iterator((int)0),
		(this->GPUMapTo.size()*(_1 % (this->settings.i_backprop_unrolled - 1))) + (_1 / (this->settings.i_backprop_unrolled - 1))
		))
		, this->GPUMapTo.size()*(this->settings.i_backprop_unrolled - 1), "testing4", "tests/test4.txt"
		);



	testing::outputToFile<weight_type>(
		thrust::make_permutation_iterator(
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
		),//End Permutation Iterator


		thrust::make_transform_iterator(//Weight 1 - 0, Weight 2-0,....
		thrust::make_counting_iterator((int)0),
		(this->GPUMapTo.size()*(_1 % (this->settings.i_backprop_unrolled - 1))) + (_1 / (this->settings.i_backprop_unrolled - 1))
		)
		),
		this->GPUMapTo.size()*(this->settings.i_backprop_unrolled - 1),
		"weights",
		"tests/test7.txt"
		);

	testing::outputToFile<weight_type>(
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
		functors::extend_value<int>(number_delta_between_add, 0, number_delta_between_add, false)
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
		)),
		this->GPUMapTo.size()*(this->settings.i_backprop_unrolled - 1),
		"test", "tests/test8.txt"
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

#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUBias, "Bias-3", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-3", "tests/prevbias2.txt");
#endif
	
	//Find the previous weights
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / (this->settings.i_backprop_unrolled - 1))
		) + ((this->settings.i_backprop_unrolled - 1) *this->GPUOutput_values.size()),

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
		this->GPUPreviousWeights.begin()
		);

#ifdef DELTA_MAP_TEST 
	testing::outputToFile<int>(this->GPUMapFrom, "From", "tests/prevbias3.txt");
	testing::outputToFile<int>(this->GPUMapTo, "To", "tests/prevbias3.txt");
#endif

#ifdef APPLY_DELTA_BIAS

	testing::outputToFile<weight_type>(this->GPUBias, "Bias-2", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-2", "tests/prevbias2.txt");
#endif
#ifdef TRAININGTEST2
	testing::outputToFile<weight_type>(this->device_deltas, "Delta", "tests/test5.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousWeights, "PrevGPUVal", "tests/test5.txt");
#endif

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
		functors::add_and_store<weight_type>(this->settings.i_backprop_unrolled - 1),
		functors::compare_two<(unsigned int)0, weight_type>(5, 1));

	thrust::copy(tempPrevBias.begin(), tempPrevBias.end(), this->GPUPreviousBias.begin());
	thrust::copy(tempBias.begin(), tempBias.end(), this->GPUBias.begin());
	thrust::copy(tempMapFrom.begin(),tempMapFrom.end(),this->GPUMapFrom.begin());
#ifdef APPLY_DELTA_BIAS

	testing::outputToFile<weight_type>(this->GPUBias, "Bias-5", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias-5", "tests/prevbias2.txt");
#endif

	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), (weight_type)0);


}

void LongTermShortTermNetwork::ApplyErrorToBias(){
	//Apply the delta to the bias
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
			this->GPUBias.begin()
			)
			)
			,
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousBias.end(),
			thrust::make_constant_iterator(this->settings.d_alpha) + this->GPUPreviousBias.size(),
			this->GPUBias.end()
			)
			),
			this->GPUBias.begin(),
			functors::multiply_add<weight_type>(),
			functors::compare_two<(unsigned int)2, weight_type>(5, (weight_type)0)
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
		this->GPUPreviousBias.begin()
		);

#ifdef APPLY_DELTA_BIAS
	testing::outputToFile<weight_type>(this->GPUBias, "Bias3", "tests/prevbias2.txt");
	testing::outputToFile<weight_type>(this->GPUPreviousBias, "PrevBias3", "tests/prevbias2.txt");
#endif

	//Apply the error
	thrust::transform_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.begin(),
		this->GPUPreviousBias.begin()
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.end(),
		this->GPUPreviousBias.end()
		)
		),
		this->GPUBias.begin(),
		functors::add_and_store<weight_type>(this->settings.i_backprop_unrolled - 1),
		functors::compare_two<(unsigned int)0, weight_type>(5, (weight_type)0)
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
		return this->runNetwork(in, 0);
	case run_type::WITH_MEMORY_CELLS:
		return this->runNetwork(in, this->GPUWeights.size() - this->numberOfWeightsInLayers[0]);
	default:
		return this->runNetwork(in, 0);
	}
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){
	return this->runNetwork(in,0);
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in,int number_of_extra_weights){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	//Number mBlocks in previous layer
	unsigned int previousnumberMBlocks = 0;
	unsigned int numberBlocksToLayer = 0;
	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);

	int output_weight_size = ((this->mBlocksLayers[this->mBlocksLayers.size() - 2].size()));

	//unsigned int numberBias = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){

		if (i != 0){
			previousnumberMBlocks += this->numberOfWeightsInLayers[i - 1] + number_of_extra_weights;
			numberBlocksToLayer += numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous

		
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUMapTo, "Start", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUMapFrom, "Start", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUPreviousOutput_Values, "Start", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "Start", "tests/PrevOut1.txt");
#endif
		thrust::reduce_by_key(
			this->GPUMapTo.begin() + previousnumberMBlocks + number_of_extra_weights,
			//Start at the beginning of the previous layer
			this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i] + number_of_extra_weights, // End at the the number of nodes before the current layer + the number of nodes in the current layer
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
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPreBias", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "GPUPreBias", "tests/PrevOut1.txt");
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
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPostBias", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "GPUPostBias", "tests/PrevOut1.txt");
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
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 3) //Old Potential
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
			this->GPUOutput_values.begin() + numberBlocksToLayer + (numberMBlocks * 4) //Old Potential
			)),
			functors::run_memory_block_functon<weight_type>());
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUMid", "tests/PrevOut1.txt");
#endif
		thrust::reduce_by_key(
			this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i] + number_of_extra_weights,
			this->GPUMapTo.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i] + this->numberOfWeightsInLayers[i + 1] + number_of_extra_weights,
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i],
			thrust::make_permutation_iterator(
			this->GPUOutput_values.begin() ,
			this->GPUMapFrom.begin() + previousnumberMBlocks + this->numberOfWeightsInLayers[i]
			)
			)
			),
			functors::multiply<weight_type>()
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks*5)
			);

#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUOut", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUBias, "GPUBiasOut", "tests/PrevOut1.txt");
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
		testing::outputToFile(this->GPUPreviousOutput_Values, "outpostbias", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "outpostbias", "tests/PrevOut1.txt");
#endif

		thrust::transform(
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5) + this->settings.i_output,
			this->GPUPreviousOutput_Values.begin() + numberBlocksToLayer + (numberMBlocks * 5),
			functors::sigmoid_functor<weight_type>()
			);
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "Final", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "Final", "tests/PrevOut1.txt");
#endif

		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.begin() + (numberMBlocks * 5) + this->settings.i_output, this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer);
		thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUNew", "tests/PrevOut1.txt");
		testing::outputToFile(this->GPUOutput_values, "GPUNew", "tests/PrevOut1.txt");
#endif
	}

	


	

	thrust::copy(this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + this->numberOfNodes - this->settings.i_output, this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + this->numberOfNodes, toReturn.begin());

	return toReturn;
}
