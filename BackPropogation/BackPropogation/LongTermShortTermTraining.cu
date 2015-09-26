#include "LongTermShortTermNetwork.cuh"
#include "TestCode.cuh"
//*********************
//Training the Network
//*********************


void LongTermShortTermNetwork::InitializeLongShortTermMemoryForRun(){
	//Form the delta objects
	this->CopyToDevice();
}

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

	//Find the delta 
	this->FindBackPropDelta(out);

}


//Find the delta gradiant for each of the "layers" of the network
void LongTermShortTermNetwork::FindBackPropDelta(weight_type* out){

	//Store the output into a vector
	for (unsigned int i = 0; i < this->RealOutput.size(); i++){
		this->RealOutput[i] = out[i];
	}

	unsigned int number_weights_in_layer = this->GPUWeights.size();
	unsigned int number_nodes_to_end_of_layer = this->GPUOutput_values.size();
	unsigned int number_nodes_to_start_of_layer = number_nodes_to_end_of_layer - this->numberOfNodes;
	//Find the deltas of the output
	for (int i = this->settings.i_backprop_unrolled; i >= 0; i--){
		//Find the delta of the output for the current layer
		//output * (1-output) * (target - output)
		thrust::transform(this->RealOutput.begin(),
			this->RealOutput.end(),
			this->GPUOutput_values.begin() + number_nodes_to_end_of_layer - this->settings.i_output,
			this->device_deltas.begin() + number_nodes_to_end_of_layer - this->settings.i_output - this->numberNonWeights,
			functors::find_output_delta<weight_type>());

		number_nodes_to_end_of_layer -= this->settings.i_output;

		if (i != this->settings.i_backprop_unrolled){//Only perform this action when we've gone past the output layer
			thrust::reduce_by_key(
				//Keeps track of the values being summed
				this->count.begin(),
				this->count.end(),

				thrust::make_permutation_iterator(

				thrust::make_transform_iterator(

				thrust::make_zip_iterator(

				thrust::make_tuple(

				this->GPUWeights.begin(),

				thrust::make_permutation_iterator(//Permutes the deltas such that they match the weights( i.e. if a delta belongs to node 1, then it purmutes to show up whenever
				//one of the nodes going to node one is found
				this->device_deltas.begin(),
				thrust::make_transform_iterator(
				this->GPUMapTo.begin(),
				_1 + number_nodes_to_start_of_layer
				)
				)
				)
				), functors::multiply<weight_type>()),
				//Position to sum is the list of positions of weights which come from a node such that they are all next to each other
				//By using this permutation, we can sum them all without sorting
				this->positionToSum.begin()
				),

				thrust::make_discard_iterator(),

				this->GPUPreviousOutput_Values.begin()
				);

			//Add the bias
			thrust::transform(
				this->GPUBias.begin(),
				this->GPUBias.end(),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->device_deltas.begin() + number_nodes_to_start_of_layer - 1,
				this->GPUPreviousOutput_Values.begin()

				)
				),
				this->GPUPreviousOutput_Values.begin(),
				functors::add_bias<weight_type>()
				);



			//Find the new deltas
			thrust::transform(
				this->device_deltas.begin() + number_nodes_to_start_of_layer - 1,
				this->device_deltas.begin() + number_nodes_to_start_of_layer - 1 + (number_nodes_to_end_of_layer - number_nodes_to_start_of_layer),
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUOutput_values.begin() + number_nodes_to_start_of_layer,
				this->GPUPreviousOutput_Values.begin()
				)), functors::find_non_output_delta<weight_type>()),
				this->device_deltas.begin() + number_nodes_to_start_of_layer - this->numberNonWeights,
				_1 + _2
				);


		}

		number_nodes_to_end_of_layer = (number_nodes_to_end_of_layer + this->settings.i_output) - this->numberOfNodes;
		number_nodes_to_start_of_layer = number_nodes_to_end_of_layer - this->numberOfNodes;

	}


}

//Apply the error
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){
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
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->device_deltas.begin(),

		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),

		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_counting_iterator(int(0)),
		thrust::make_permutation_iterator(
		this->GPUMapFrom.begin(),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator(int(0)),
		_1 % this->GPUMapFrom.size()
		)
		)

		)

		),
		functors::extend_value<int>(this->GPUMapFrom.size())
		)
		)

		)
		),
		functors::find_previous_weight<weight_type>(this->settings.d_beta)),
		thrust::make_transform_iterator(

		thrust::make_counting_iterator((int)0),

		((1 + this->numberOfNodes)*(_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)),

		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);


	testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values);


	thrust::transform(

		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUMapFrom.begin()
		),
		this->GPUPreviousWeights.begin()
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUMapFrom.begin()
		),
		this->GPUPreviousWeights.begin()
		)
		) + this->GPUPreviousWeights.size()

		,
		this->GPUWeights.begin(),
		this->GPUWeights.begin(),
		functors::apply_new_error<weight_type>(this->settings.d_alpha, this->settings.i_backprop_unrolled*this->settings.i_number_in_sequence)
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
		((1 + this->numberOfNodes)*(_1%this->settings.i_backprop_unrolled)) + (_1 / this->settings.i_backprop_unrolled)
		)
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousBias.begin()
		);

	//Apply the error
	thrust::transform(
		this->GPUBias.begin(),
		this->GPUBias.end(),
		this->GPUPreviousBias.begin(),
		this->GPUBias.begin(),
		thrust::plus<weight_type>()
		);

	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), (weight_type)0);
}