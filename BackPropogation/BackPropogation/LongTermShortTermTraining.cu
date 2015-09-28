#include "LongTermShortTermNetwork.cuh"
//#define TRAININGTEST
//#define TRAININGTEST2

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
	

	
	thrust::transform(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUPreviousWeights.begin()
		)
		)
		,
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUPreviousOutput_Values.end(),
		this->GPUPreviousWeights.end()
		)
		),
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
	thrust::transform(
		this->GPUBias.begin(),
		this->GPUBias.end(),
		this->GPUPreviousBias.begin(),
		this->GPUBias.begin(),
		thrust::plus<weight_type>()
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

	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){

		if (i != 0){
			previousnumberMBlocks = numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous
		thrust::reduce_by_key(this->GPUMapTo.begin(), this->GPUMapTo.end(), thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin() + this->numberNonWeights + previousnumberMBlocks, // We don't want to multiply the actual input/out values, so we skip them
			make_permutation_iterator( // Create an iterator which maps the values coming from to those going to
			this->GPUOutput_values.begin(),
			this->GPUMapFrom.begin())
			)
			),
			functors::multiply<weight_type>()), //Multiply the two values then run them through a sigmoid function
			thrust::make_discard_iterator(), // Discard the retrieved order, the order should be constant
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks//Store in the previous in order to not overwrite the saved values
			);

		thrust::transform(
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			this->GPUBias.begin() + previousnumberMBlocks
			)
			),
			functors::add<weight_type>()
			),
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks + (5 * numberMBlocks),
			this->GPUBias.begin() + previousnumberMBlocks
			)
			),
			functors::add<weight_type>()
			)

			

			,
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			functors::sigmoid_functor<weight_type>());

		//Create a input/output/forget/potential_memory_cell_values/memory_cell_value value
		//Essentially run the gate and get the output value
		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks + this->settings.i_input, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks + this->settings.i_input,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input
			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks + this->settings.i_input, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (5 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input
			)),
			functors::run_memory_block_functon<weight_type>());
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->GPUOutput_values.begin());

	}

	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);

	int output_weight_size = ((this->mBlocksLayers[this->mBlocksLayers.size() - 2].size()));

	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1 / output_weight_size // The number of output nodes in layer before the output layer
		),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1 / output_weight_size // The number of output nodes in layer before the output layer
		) + output_weight_size * this->settings.i_output,
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		this->GPUMapFrom.end() - (output_weight_size * this->settings.i_output)),

		thrust::make_permutation_iterator(
		this->GPUWeights.begin(),
		this->GPUMapTo.end() - (output_weight_size*this->settings.i_output)
		)
		)
		),
		functors::multiply<weight_type>()
		),
		thrust::make_discard_iterator(),
		toReturn.begin()
		);

	thrust::transform(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		toReturn.begin(),
		this->GPUBias.end()-this->settings.i_output
		)
		),
		functors::add<weight_type>()
		),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		toReturn.end(),
		this->GPUBias.end()
		)
		),
		functors::add<weight_type>()
		),
		toReturn.begin(),
		functors::sigmoid_functor<weight_type>());

	return toReturn;
}
