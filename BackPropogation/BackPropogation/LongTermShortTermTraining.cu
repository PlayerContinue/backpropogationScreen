#include "LongTermShortTermNetwork.cuh"
//#define TRAININGTEST
//#define TRAININGTEST2
//#define DELTA_TEST
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
	this->device_deltas = device_vector<weight_type>(this->GPUOutput_values.size() - this->numberNonWeights);
	this->RealOutput = device_vector<weight_type>(this->settings.i_output);
}

void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type* in, weight_type* out){
	//Get the number of weights in the output layer
	//This is needed because the output layer needs to be used only once, so we need to inform the system which weights to skip

	//Set the input values
	this->setInput(in);
	unsigned int number_weights_to_beginining_of_layer = 0;
	unsigned int number_nodes_to_beginning_of_layer = 0;
	unsigned int number_weights_in_layer = this->GPUWeights.size();

	//The first row is only for input
	//Thus we only sum the input
	thrust::reduce_by_key(
		this->GPUMapTo.begin(),
		this->GPUMapTo.begin() + (this->numberNonWeights * this->mBlocksLayers[0].size()),
		//Multiply the weights x output
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(

		this->GPUWeights.begin(),
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		this->GPUMapFrom.begin()
		)
		)
		),
		functors::multiply<weight_type>()
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);

	//Transfer all values from the current to the next row
	thrust::transform(this->GPUPreviousOutput_Values.begin(),
		this->GPUPreviousOutput_Values.begin() + (this->numberNonWeights * this->mBlocksLayers[0].size()),
		this->GPUOutput_values.begin() + this->numberNonWeights, functors::sigmoid_functor<weight_type>());

	for (int i = 0; i < this->settings.i_backprop_unrolled; i++){
		thrust::reduce_by_key(
			this->GPUMapTo.begin(),
			this->GPUMapTo.end(),

			//Multiply the weights x output
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(

			this->GPUWeights.begin(),

			thrust::make_permutation_iterator(
			this->GPUOutput_values.begin(),
			thrust::make_transform_iterator(
			this->GPUMapFrom.begin(), functors::add_constant_value<int>(number_nodes_to_beginning_of_layer, this->settings.i_input))
			)
			)
			),
			functors::multiply<weight_type>()
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);

		if (i > 0){//Only increment it by the number of nodes when working from any layer which is not the initial layer
			//This lets the nodes use the previous layer as their input
			number_weights_to_beginining_of_layer += number_weights_in_layer;
			number_nodes_to_beginning_of_layer += this->numberOfNodes;
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

	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the delta 
	this->FindBackPropDelta(out,0);

}


//Find the delta gradiant for each of the "layers" of the network
void LongTermShortTermNetwork::FindBackPropDelta(weight_type* out, int current_layer){

	//Store the output into a vector
	for (unsigned int i = 0; i < this->RealOutput.size(); i++){
		this->RealOutput[i] = out[i];
	}

	unsigned int number_weights_in_layer = this->GPUWeights.size();
	
	unsigned int number_nodes_to_end_of_layer = this->GPUOutput_values.size();
	
	unsigned int number_nodes_to_start_of_layer = number_nodes_to_end_of_layer - this->numberOfNodes;
	
	//Starts at the beginning of the weights for the output
	unsigned int weights_to_start_of_next_layer = this->GPUWeights.size(); //(this->settings.i_output*this->mBlocksLayers[this->mBlocksLayers.size()-2].size()); //Stores info used only on the output
	
	unsigned int on_last_layer = this->numberOfNodes - (this->settings.i_output * this->settings.i_output*this->mBlocksLayers[this->mBlocksLayers.size() - 2].size());//Used to allow for a special situtation for the output without changing the code
	
	unsigned int number_nodes_to_end = this->settings.i_output;
	
	unsigned int end_of_succeeding_layer = number_nodes_to_end_of_layer;//Marks the end of the next layer in the sequence
	
	unsigned int number_weights_to_start_from_end =this->GPUMapTo.size();//this->settings.i_output * this->settings.i_output*this->mBlocksLayers[this->mBlocksLayers.size() - 2].size();
	
	unsigned int last_layer_bias = this->GPUBias.size() - this->settings.i_output;
	//Find the deltas of the output
	for (int i = this->settings.i_backprop_unrolled; i >= 0; i--){
		//Find the delta of the output for the current layer
		//output * (1-output) * (target - output)
		thrust::transform(
			this->RealOutput.begin(),
			this->RealOutput.end(),
			this->GPUOutput_values.begin() + number_nodes_to_end_of_layer - this->settings.i_output,
			this->device_deltas.begin() + number_nodes_to_end_of_layer - this->settings.i_output - this->numberNonWeights,
			functors::find_output_delta<weight_type>());

		number_nodes_to_end_of_layer -= this->settings.i_output;
#ifdef DELTA_TEST
		testing::outputToFile<weight_type>(this->GPUBias.begin(), 1, "test", "test11");
		testing::outputToFile<weight_type>(this->device_deltas, "PostOutput", "tests/testing.txt");
		

		testing::outputToFile<weight_type>(
			thrust::make_permutation_iterator(
			this->device_deltas.begin() + end_of_succeeding_layer - this->numberNonWeights - this->numberOfNodes,
			thrust::make_transform_iterator(
			this->GPUMapTo.end() - number_weights_to_start_from_end,
			_1 - this->numberNonWeights

			)
			), this->GPUWeights.size(), "Results2", "tests/testing.txt");


		testing::outputToFile<weight_type>(
			//thrust::make_permutation_iterator(
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - weights_to_start_of_next_layer,
			
			thrust::make_permutation_iterator(
			this->device_deltas.begin() + end_of_succeeding_layer - this->numberNonWeights - this->numberOfNodes,
			thrust::make_transform_iterator(
			this->GPUMapTo.end() - number_weights_to_start_from_end,
			_1 - this->numberNonWeights
			)
			)
			)
			),
			functors::multiply<weight_type>()
			)//,
			//this->positionToSum.begin()

			//)

			, this->positionToSum.size(), "Results", "tests/testing.txt");
#endif
		if (i != this->settings.i_backprop_unrolled){//Only perform this action when we've gone past the output layer
		
			thrust::reduce_by_key(
				this->count.begin(),
				this->count.end(),
				thrust::make_permutation_iterator(
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUWeights.end() - weights_to_start_of_next_layer,

				thrust::make_permutation_iterator(
				this->device_deltas.begin() + end_of_succeeding_layer - this->numberNonWeights - this->numberOfNodes,
				thrust::make_transform_iterator(
				this->GPUMapTo.end() - number_weights_to_start_from_end,
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
			thrust::transform(
				this->GPUBias.begin(),
				this->GPUBias.end(),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->device_deltas.begin() + end_of_succeeding_layer - this->numberNonWeights - this->numberOfNodes,
				this->GPUPreviousOutput_Values.begin()

				)
				),
				this->GPUPreviousOutput_Values.begin(),
				functors::add_bias<weight_type>(this->settings.i_backprop_unrolled -1 != i)
				);
#ifdef DELTA_TEST
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "PostBias", "tests/testing.txt");
#endif
			//Find the new deltas
			thrust::transform(
				this->device_deltas.begin() + number_nodes_to_start_of_layer - this->numberNonWeights,
				this->device_deltas.begin() + number_nodes_to_end_of_layer + this->settings.i_output - this->numberNonWeights,
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUOutput_values.begin() + number_nodes_to_start_of_layer,
				this->GPUPreviousOutput_Values.begin()
				)), functors::find_non_output_delta<weight_type>()),
				this->device_deltas.begin() + number_nodes_to_start_of_layer - this->numberNonWeights,
				_1 + _2
				);
#ifdef DELTA_TEST
			testing::outputToFile<weight_type>(this->device_deltas, "Device","tests/testing.txt");
#endif
		}

		if (i == this->settings.i_backprop_unrolled - 1){
			//Change the values to the width of every row besides the input
			on_last_layer = 0;//No longer working on the last layer, so no longer needed
			number_nodes_to_end = this->count.size();//Need to go back the whole way now
			last_layer_bias = 0;
			//number_weights_to_start_from_end = this->GPUMapTo.size();
			//weights_to_start_of_next_layer = this->numberOfWeightsInLayers[current_layer];//Set the weights back to the start
		}

		if (i < this->settings.i_backprop_unrolled){
			end_of_succeeding_layer = number_nodes_to_end_of_layer + this->settings.i_output;
			
		}
		
			number_nodes_to_end_of_layer = (number_nodes_to_end_of_layer + this->settings.i_output) - this->numberOfNodes;
			number_nodes_to_start_of_layer = number_nodes_to_end_of_layer - this->numberOfNodes;
			
		
	}


}

//Apply the error
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){
#ifdef TRAININGTEST2
	testing::outputToFile<weight_type>(this->device_deltas, "Delta");
	testing::outputToFile<weight_type>(this->GPUOutput_values, "Output");
#endif

#ifdef TRAININGTEST
	/*thrust::copy(
		
		thrust::make_permutation_iterator(

		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapTo.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapTo.size(), 1, this->numberOfNodes, false)

		), thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)),
		thrust::make_permutation_iterator(
		
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapTo.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapTo.size(), 1, this->numberOfNodes, false)

		), thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)) + this->GPUMapTo.size() * this->settings.i_backprop_unrolled, std::ostream_iterator<int>(std::cout, "\n"));*/

	/*thrust::copy(

		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapFrom.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		)
		,
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes, true)


		),
		
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapFrom.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		)
		,
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes, true)


		) + this->GPUMapTo.size() * this->settings.i_backprop_unrolled, std::ostream_iterator<int>(std::cout, "\n"));*/

/*thrust::copy(
	thrust::make_transform_iterator(
	thrust::make_permutation_iterator(
	thrust::make_permutation_iterator(
	this->GPUMapTo.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapTo.size()
	)
	),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
	)
	),
	_1%this->numberOfNodes
	)
	,
	thrust::make_transform_iterator(
	thrust::make_permutation_iterator(
	thrust::make_permutation_iterator(
	this->GPUMapTo.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapTo.size()
	)
	),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
	)
	),
	_1%this->numberOfNodes
	)
	+ this->GPUMapTo.size() * this->settings.i_backprop_unrolled,
	std::ostream_iterator<int>(std::cout, "\n")
	);*/

testing::outputToFile<weight_type>(thrust::make_permutation_iterator(
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->device_deltas.begin(),

	thrust::make_permutation_iterator(

	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->GPUMapTo.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapTo.size()
	)
	),
	thrust::make_counting_iterator((int)0)
	)
	),
	functors::extend_value<int>(this->GPUMapTo.size(), 1, this->numberOfNodes, false)

	), thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
	))

	),

	thrust::make_permutation_iterator(
	this->GPUOutput_values.begin(),
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->GPUMapFrom.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapFrom.size()
	)
	),
	thrust::make_counting_iterator((int)0)
	)
	)
	,
	functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes, true)


	)

	)
	)
	)
	,
	functors::find_previous_weight<weight_type>(this->settings.d_beta)
	),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)

	)
	), this->GPUMapTo.size() * this->settings.i_backprop_unrolled,
	"Sorted Results"
	);

testing::outputToFile<weight_type>(
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->device_deltas.begin(),

	thrust::make_permutation_iterator(

	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->GPUMapTo.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapTo.size()
	)
	),
	thrust::make_counting_iterator((int)0)
	)
	),
	functors::extend_value<int>(this->GPUMapTo.size(), 1, this->numberOfNodes, false)

	), thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
	))

	),

	thrust::make_permutation_iterator(
	this->GPUOutput_values.begin(),
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	thrust::make_permutation_iterator(
	this->GPUMapFrom.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapFrom.size()
	)
	),
	thrust::make_counting_iterator((int)0)
	)
	)
	,
	functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes, true)


	)

	)
	)
	)
	,
	functors::find_previous_weight<weight_type>(this->settings.d_beta)
	), this->GPUMapTo.size() * this->settings.i_backprop_unrolled,
	"OutputResults"



	);

testing::outputToFile<int>(thrust::make_transform_iterator(
	thrust::make_permutation_iterator(
	thrust::make_permutation_iterator(
	this->GPUMapTo.begin(),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	_1%this->GPUMapTo.size()
	)
	),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
	)
	),
	_1%this->numberOfNodes
	), this->GPUMapTo.size() * this->settings.i_backprop_unrolled, "Order");
	int k = 0;

#endif
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_permutation_iterator(
		thrust::make_permutation_iterator(
		this->GPUMapTo.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)
		),
		_1%this->numberOfNodes
		),
		thrust::make_transform_iterator(
		thrust::make_permutation_iterator(
		thrust::make_permutation_iterator(
		this->GPUMapTo.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)
		),
		_1%this->numberOfNodes
		) + this->GPUMapTo.size() * this->settings.i_backprop_unrolled,
		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->device_deltas.begin(),
		
		thrust::make_permutation_iterator(

		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapTo.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapTo.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		),
		functors::extend_value<int>(this->GPUMapTo.size(), 1, this->numberOfNodes, false)

		), thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		))
		
		),
		
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUMapFrom.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1%this->GPUMapFrom.size()
		)
		),
		thrust::make_counting_iterator((int)0)
		)
		)
		,
		functors::extend_value<int>(this->GPUMapFrom.size(), 0, this->numberOfNodes, true)


		)

		)
		)
		)
		,
		functors::find_previous_weight<weight_type>(this->settings.d_beta)
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->GPUWeights.size())* (_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		
		)
		),
		
		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()

		);

#ifdef TRAININGTEST2
	testing::outputToFile<weight_type>(this->device_deltas, "Delta");
	testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values,"PrevGPUVal");
#endif
	
	int memory_start_position = this->GPUWeights.size() - ((this->settings.i_output*this->mBlocksLayers[this->mBlocksLayers.size() - 2].size())) -
		(this->mBlocksLayers[0].size()*NUMBER_MEMORY_WEIGHTS);
	int memory_end_position = this->GPUWeights.size() - ((this->settings.i_output*this->mBlocksLayers[this->mBlocksLayers.size() - 2].size()));

	
	thrust::transform(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUMapTo.begin()
		),
		this->GPUPreviousWeights.begin(),
		this->GPUWeights.begin(),
		thrust::make_counting_iterator((int)0)
		)
		)
		,
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUMapTo.begin()
		),
		this->GPUPreviousWeights.begin(),
		this->GPUWeights.begin(),
		thrust::make_counting_iterator((int)0)
		)
		) + this->GPUWeights.size(),
		this->GPUWeights.begin(),
		functors::apply_new_error<weight_type>(this->settings.d_alpha, this->settings.i_backprop_unrolled, memory_start_position,
		memory_end_position)
		);

	//Apply to bias
	if (this->settings.d_alpha != 0){
		//Apply the alpha to bias
		thrust::transform(
			this->GPUBias.begin(),
			this->GPUBias.end(),
			this->GPUPreviousBias.begin(),
			this->GPUBias.begin(),
			_1 + (_2 * (weight_type)this->settings.d_alpha)
			);
	}

	//Retrieve the new previous Bias
	//If I remove memory cells, remove permutation around device_delta so it doesn't skip
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->settings.i_backprop_unrolled)
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->settings.i_backprop_unrolled)
		) + this->device_deltas.size(),

		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		this->device_deltas.begin(),
		functors::multiply_by_constant<weight_type>((weight_type)this->settings.d_beta)
		),

		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		((this->numberOfNodes)*(_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousBias.begin()
		);


	//Apply the error
	thrust::transform_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.begin(),
		this->GPUPreviousBias.begin()
	
		))
		,
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUBias.end(),
		this->GPUPreviousBias.end()
		)),
		thrust::make_counting_iterator((int)0),
		this->GPUBias.begin(),
		functors::add_and_store<weight_type>(this->settings.i_backprop_unrolled),
		functors::check_not_between<int>(this->GPUBias.size() - (this->settings.i_output*this->settings.i_backprop_unrolled) - this->mBlocksLayers[0].size(), 
		this->GPUBias.size() - (this->settings.i_output*this->settings.i_backprop_unrolled))
		);

	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), (weight_type)0);
}

//*********************
//Run The Network
//*********************


void LongTermShortTermNetwork::InitializeLongShortTermMemoryForRun(){
	//Form the delta objects
	this->CopyToDevice();
	this->moveBiasToGPU(false);//Don't create a previous_bias
}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	//Number mBlocks in previous layer
	unsigned int previousnumberMBlocks = 0;
	unsigned int numberBlocksToLayer = 0;

	//unsigned int numberBias = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){

		if (i != 0){
			previousnumberMBlocks += this->numberOfWeightsInLayers[i - 1];
			numberBlocksToLayer += numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "Start", "tests/PrevOut1.txt");
#endif
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
			functors::multiply<weight_type>() //Multiply the two together
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPreBias", "tests/PrevOut1.txt");
#endif
		thrust::transform(
			this->GPUBias.begin() + numberBlocksToLayer,
			this->GPUBias.begin() + numberBlocksToLayer + (NUMBER_MEMORY_WEIGHTS * numberMBlocks),
			this->GPUPreviousOutput_Values.begin(),
			this->GPUPreviousOutput_Values.begin(),
			_1 + _2

			);

		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUPostBias", "tests/PrevOut1.txt");
		//Create a input/output/forget/potential_memory_cell_values/memory_cell_value value
		//Essentially run the gate and get the output value
		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin(), //input values
			this->GPUPreviousOutput_Values.begin() +  numberMBlocks,//output values
			this->GPUPreviousOutput_Values.begin() +  (2 * numberMBlocks),//forget values
			this->GPUPreviousOutput_Values.begin() +  (3 * numberMBlocks),//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + (4 * numberMBlocks),
			this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + (numberMBlocks * 4)//Memory Cell Values

			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + numberMBlocks, //input values
			this->GPUPreviousOutput_Values.begin() + (2 * numberMBlocks),//output values
			this->GPUPreviousOutput_Values.begin() + (3 * numberMBlocks),//forget values
			this->GPUPreviousOutput_Values.begin() + (4 * numberMBlocks),//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + (5 * numberMBlocks),
			this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer + (numberMBlocks * 5)//Memory Cell Values
			)),
			functors::run_memory_block_functon<weight_type>());
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUPreviousOutput_Values, "GPUMid", "tests/PrevOut1.txt");
#endif
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.begin() + (numberMBlocks * 5), this->GPUOutput_values.begin() + this->numberNonWeights + numberBlocksToLayer);
		thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
#ifdef TRAININGTEST
		testing::outputToFile(this->GPUOutput_values, "GPUNew", "tests/PrevOut1.txt");
#endif
	}

	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);

	int output_weight_size = ((this->mBlocksLayers[this->mBlocksLayers.size() - 2].size()));


	thrust::reduce_by_key(
		this->GPUMapTo.end() - (output_weight_size * this->settings.i_output),
		this->GPUMapTo.end(),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUWeights.end() - (output_weight_size * this->settings.i_output),
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		this->GPUMapFrom.end() - (output_weight_size * this->settings.i_output)
		)
		)
		),
		functors::multiply<weight_type>()
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);
#ifdef TRAININGTEST
	testing::outputToFile(this->GPUPreviousOutput_Values, "GPUOut", "tests/PrevOut1.txt");
#endif
	//Add the bias
	thrust::transform(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUBias.end() - this->settings.i_output
		)
		),
		functors::add<weight_type>()
		),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUPreviousOutput_Values.begin() + this->settings.i_output,
		this->GPUBias.end()
		)
		),
		functors::add<weight_type>()
		),
		this->GPUOutput_values.end() - this->settings.i_output,
		functors::sigmoid_functor<weight_type>()
		);
#ifdef TRAININGTEST
	testing::outputToFile(this->GPUOutput_values, "Final", "tests/PrevOut1.txt");
#endif
	thrust::copy(this->GPUOutput_values.end() - this->settings.i_output, this->GPUOutput_values.end(), toReturn.begin());

	return toReturn;
}
