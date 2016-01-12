#include "Hessian_Network.cuh"




//*********************
//Initialize the network
//*********************
void Hessian_Network::InitializeHessianNetwork(){
	this->alphas = thrust::device_vector<weight_type>(this->GPUWeights.size() * this->settings.i_backprop_unrolled);
}


//*********************
//Training the Network
//*********************

void Hessian_Network::StartTraining(weight_type** in, weight_type** out){
	//Reset the weights to the end of the weights
	this->averageWeights();
	//Set the input values
	this->setInput(in);
	this->training_previous_number_rows = this->settings.i_backprop_unrolled;
	//Run the network
	this->TrainingRun(in, out);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the delta 
	this->FindBackPropDelta(out, 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find Initial Hessian Values
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->findHessianFreeMatrix();
	//thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the alpha
	//this->findAlpha();
	this->FindPreviousBias();
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->FindPreviousWeights();
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), (weight_type)0);
}

void Hessian_Network::TrainingRun(weight_type** in, weight_type** out){
	int start = 1;

	//Get the number of weights in the output layer
	//This is needed because the output layer needs to be used only once, so we need to inform the system which weights to skip

	unsigned int number_nodes_to_internal_next_layer = 0;//Number nodes to the next "layer" in the current layer
	unsigned int number_weights_to_internal_next_layer = 0; // number weights to the next "layer" in the current layer
	unsigned int number_nodes_to_start_of_alpha_storage = 0;
	//Number of nodes to the start of the current layer to which new numbers will be added
	unsigned int number_nodes_to_start_of_storage_layer = this->numberNonWeights + this->numberOfNodes + this->numberNonWeights;//Two number of non weights to get to the start of the next set of non input values

	//Number nodes to the beginning of the previous layer from which data will be gathered
	unsigned int number_nodes_to_beginning_of_layer = 0;

	unsigned int number_weights_in_layer = this->GPUWeights.size();
	for (int i = start; i < this->settings.i_backprop_unrolled; i++){
		number_nodes_to_internal_next_layer = 0;
		number_weights_to_internal_next_layer = 0;
		for (int j = 0; j < this->mBlocksLayers.size(); j++){//Go through each layer of the current iteration
			thrust::reduce_by_key(
				this->GPUMapTo.begin() + number_weights_to_internal_next_layer,
				this->GPUMapTo.begin() + number_weights_to_internal_next_layer + this->numberOfWeightsInLayers[j],

				//Multiply the weights x output
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUWeights.begin() + number_weights_to_internal_next_layer,
				thrust::make_permutation_iterator(
				this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer,
				this->GPUMapFrom.begin() + number_weights_to_internal_next_layer
				)
				)
				),
				functors::multiply<weight_type>()
				),
				thrust::make_discard_iterator(),
				this->GPUPreviousOutput_Values.begin()
				);

			//Copy the values into the alpha list
			thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->alphas.begin() + number_nodes_to_start_of_storage_layer + number_nodes_to_internal_next_layer);

			//Add the bias to the current value
			thrust::transform(this->GPUBias.begin() + number_nodes_to_internal_next_layer,
				this->GPUBias.begin() + number_nodes_to_internal_next_layer + this->number_nodes_in_layer[j],
				this->GPUPreviousOutput_Values.begin(),
				this->GPUOutput_values.begin() + number_nodes_to_start_of_storage_layer + number_nodes_to_internal_next_layer,//Start + number of nodes to layer with searching values + number of nodes to current layer
				functors::sum_and_sigmoid<weight_type>()
				);


#ifdef NVIDA_OUTPUT_TEST2
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "prevout1" + std::to_string(j) + std::to_string(i), "tests/prevbias3.txt");
#endif



			number_nodes_to_internal_next_layer += this->number_nodes_in_layer[j];
			number_weights_to_internal_next_layer += this->numberOfWeightsInLayers[j];

#ifdef NVIDA_OUTPUT_TEST2
			testing::outputToFile<weight_type>(this->GPUOutput_values, "fullout2" + std::to_string(j) + std::to_string(i), "tests/prevbias3.txt");
			testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "prevout2" + std::to_string(j) + std::to_string(i), "tests/prevbias3.txt");
#endif

		}


#ifdef NVIDA_OUTPUT_TEST2
		testing::outputToFile<weight_type>(this->GPUPreviousOutput_Values, "prevout", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(this->GPUWeights, "weights1", "tests/prevbias3.txt");
		testing::outputToFile<weight_type>(this->GPUBias, "bias1", "tests/prevbias3.txt");

#endif
		//Only increment it by the number of nodes when working from any layer which is not the initial layer
		//This lets the nodes use the previous layer as their input
		number_nodes_to_beginning_of_layer += this->numberOfNodes + this->numberNonWeights;
		number_nodes_to_start_of_storage_layer += this->numberNonWeights + this->numberOfNodes;
		number_nodes_to_start_of_alpha_storage += this->numberOfNodes;

	}



}

//Find the hessian free matrix
void Hessian_Network::findHessianFreeMatrix(){
	//Initial layer has x_i of 0
	// and therefore y_i of 0
	//Find the vector of the x_i
	thrust::reduce_by_key(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size()),
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size()),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_repeat_list_iterator(this->GPUWeights.begin(), this->GPUWeights.size()),
		Unique_Iterator::make_return_iterator(this->alphas.begin(), this->alphas.end(), this->numberOfNodes + this->numberNonWeights,
		this->numberOfNodes + this->numberNonWeights, this->settings.i_input),
		this->GPUOutput_values.begin(),
		thrust::make_constant_iterator((int)1)
		)
		),
		functors::find_forward_x_hessian<weight_type>()
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);

	//Find the forward hessian weight
	thrust::transform(
		this->GPUPreviousOutput_Values.begin(),
		this->GPUPreviousOutput_Values.begin() + this->numberOfNodes,
		this->alphas.begin(),
		this->alphas.begin(),
		functors::find_forward_y_hessian<weight_type>()
		);




}



void Hessian_Network::ApplyError(){

}

void Hessian_Network::findAlpha(){



	int alpha_length = (this->settings.i_backprop_unrolled * (this->GPUWeights.size()));
	thrust::fill(this->device_deltas.begin(), this->device_deltas.begin() + this->numberOfNodes + this->numberNonWeights, 1);
	thrust::device_vector<weight_type> temp(this->alphas.size());
	/*testing::outputToFile(this->GPUOutput_values, "testing", "tests/GPUO.txt");
	testing::outputToFile(this->device_deltas, "testing", "tests/deltas.txt");
	testing::outputToFile<weight_type>(thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(

	thrust::make_permutation_iterator(
	Unique_Iterator::make_return_iterator(this->device_deltas.begin(), this->device_deltas.begin(),
	0,
	(this->numberOfNodes + this->numberNonWeights), this->settings.i_input),
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
	)
	)
	),
	functors::add<int>()
	)
	),
	thrust::make_permutation_iterator(
	this->GPUOutput_values.begin(),
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
	)
	)
	),
	functors::add<int>()
	)
	),
	Unique_Iterator::make_repeat_list_iterator(
	this->GPUWeights.begin(),
	this->GPUWeights.size()
	)
	)
	),
	functors::find_top_alpha<weight_type>()
	), (this->settings.i_backprop_unrolled * (this->GPUWeights.size())), "testing", "tests/return.txt");*/
	thrust::fill(this->alphas.begin(), this->alphas.end(), (weight_type)0);
	thrust::reduce_by_key(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size()),
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size())
		+ (alpha_length),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(

		thrust::make_permutation_iterator(
		Unique_Iterator::make_return_iterator(this->device_deltas.begin(), this->device_deltas.begin(),
		0,
		(this->numberOfNodes + this->numberNonWeights), this->settings.i_input),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
		)
		)
		),
		functors::add<int>()
		)
		),
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
		)
		)
		),
		functors::add<int>()
		)
		),
		Unique_Iterator::make_repeat_list_iterator(
		this->GPUWeights.begin(),
		this->GPUWeights.size()
		)
		)
		),
		functors::find_top_alpha<weight_type>()
		),
		thrust::make_discard_iterator(),
		this->alphas.begin()
		);



	//Find the denominator
	thrust::reduce_by_key(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size()),
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapTo.begin(), this->GPUMapTo.size())
		+ (this->settings.i_backprop_unrolled * (this->GPUWeights.size())),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(

		thrust::make_permutation_iterator(
		Unique_Iterator::make_return_iterator(this->device_deltas.begin(), this->device_deltas.begin(),
		0,
		(this->numberOfNodes + this->numberNonWeights), this->settings.i_input),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
		)
		)
		),
		functors::add<int>()
		)
		),
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_repeat_list_iterator(this->GPUMapFrom.begin(), this->GPUMapFrom.size()),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		(_1 / this->GPUMapFrom.size())* (this->numberOfNodes + this->numberNonWeights)
		)
		)
		),
		functors::add<int>()
		)
		)
		)
		),
		functors::find_alpha_denominator<weight_type>()
		),
		thrust::make_discard_iterator(),
		temp.begin()
		);

	thrust::transform(this->alphas.begin(),
		this->alphas.begin() + (alpha_length),
		temp.begin(),
		this->alphas.begin(),
		functors::special_divide<weight_type>()
		);

	/*thrust::reduce_by_key(
	Unique_Iterator::make_repeat_iterator(thrust::make_counting_iterator((int)0), this->settings.i_backprop_unrolled),
	Unique_Iterator::make_repeat_iterator(thrust::make_counting_iterator((int)0), this->settings.i_backprop_unrolled) + alpha_length,
	thrust::make_permutation_iterator(
	thrust::make_transform_iterator(
	thrust::make_zip_iterator(
	thrust::make_tuple(
	Unique_Iterator::make_repeat_list_iterator(this->alphas.begin(), alpha_length),
	Unique_Iterator::make_repeat_list_iterator(this->device_deltas.begin(), this->device_deltas.size()),
	)
	),
	functors::multiply<weight_type>()
	),
	thrust::make_transform_iterator(
	thrust::make_counting_iterator((int)0),
	(_1 * (this->numberOfNodes)) + ((_1*(this->numberOfNodes)) % (alpha_length))
	)

	),
	thrust::make_discard_iterator(),
	this->GPUPreviousWeights.begin()

	);
	thrust::transform(
	thrust::make_transform_iterator(
	this->GPUPreviousWeights.begin(),
	_1 / this->settings.i_backprop_unrolled
	),
	thrust::make_transform_iterator(
	this->GPUPreviousWeights.end(),
	_1 / this->settings.i_backprop_unrolled
	),
	this->GPUWeights.begin(),
	this->GPUWeights.begin(),
	thrust::plus<weight_type>()

	);*/
	testing::outputToFile(this->alphas, "alpha", "tests/alpha.txt");
	testing::outputToFile(this->GPUPreviousWeights, "prev", "tests/prev.txt");

}