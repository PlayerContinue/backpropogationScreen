#include "Hessian_Network.cuh"




//*********************
//Initialize the network
//*********************
void Hessian_Network::InitializeHessianNetwork(){
	this->alphas = thrust::device_vector<weight_type>(this->device_deltas.size());
	this->hessian = thrust::device_vector<weight_type>(this->GPUWeights.size());
	int i = 0;
}

void Hessian_Network::InitializeTraining(){
	this->InitializeLongShortTermMemory();
	this->InitializeHessianNetwork();
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
	int temp = this->GPUPreviousOutput_Values.size();
	//Find Initial Hessian Values
	this->GPUPreviousOutput_Values.resize(this->numberOfNodes + this->numberNonWeights);
	thrust::copy(this->GPUOutput_values.begin(), this->GPUOutput_values.begin() + this->numberOfNodes + this->numberNonWeights,
		this->GPUPreviousOutput_Values.begin());
	this->findHessianFreeMatrixForward();
	this->GPUPreviousOutput_Values.resize(temp);
	//Find the delta 
	this->FindBackPropDelta(out, 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->findWeightDeltas();
	//Find backward
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	this->GPUPreviousOutput_Values.resize(this->numberOfNodes + this->numberNonWeights);
	this->findHessianFreeMatrixBackward();
	
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	//Find the alpha value
	this->findAlpha();

	this->GPUPreviousOutput_Values.resize(temp);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	
	//this->FindPreviousBias();
	//thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	
	//this->FindPreviousWeights();
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


			//Add the bias to the current value
			thrust::transform(this->GPUBias.begin() + number_nodes_to_internal_next_layer,
				this->GPUBias.begin() + number_nodes_to_internal_next_layer + this->number_nodes_in_layer[j],
				this->GPUPreviousOutput_Values.begin(),
				this->GPUOutput_values.begin() + number_nodes_to_start_of_storage_layer + number_nodes_to_internal_next_layer,//Start + number of nodes to layer with searching values + number of nodes to current layer
				functors::sum_and_sigmoid<weight_type>()
				);



			number_nodes_to_internal_next_layer += this->number_nodes_in_layer[j];
			number_weights_to_internal_next_layer += this->numberOfWeightsInLayers[j];



		}



		//Only increment it by the number of nodes when working from any layer which is not the initial layer
		//This lets the nodes use the previous layer as their input
		number_nodes_to_beginning_of_layer += this->numberOfNodes + this->numberNonWeights;
		number_nodes_to_start_of_storage_layer += this->numberNonWeights + this->numberOfNodes;
		number_nodes_to_start_of_alpha_storage += this->numberOfNodes;

	}



}

//Find the hessian free matrix
void Hessian_Network::findHessianFreeMatrixForward(){
	unsigned int start = 0;//Start point of the hessian
	unsigned int start_output = 0;//The start of the current output
	unsigned int start_node_internal = 0;//The nodes to skip in the current layer
	unsigned int end_node_internal = 0;//The number of nodes to the end of the current iterator
	unsigned int start_weight_internal = 0;//The number of weights to the start of the current iteration
	unsigned int end_weight_internal = 0;//Number of weights to the end of the current iteration
#ifdef HESSIAN_FORWARD_OUT
	testing::outputToFile(this->GPUPreviousOutput_Values, "input", "tests/alphas.txt");
	testing::outputToFile<weight_type>(this->GPUOutput_values, "output", "tests/alphas.txt");
#endif
	//thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), (weight_type).6);
	for (int i = 0; i < this->settings.i_backprop_unrolled-1; i++){
		for (int j = 0; j < this->mBlocksLayers.size(); j++){
		end_weight_internal += this->numberOfWeightsInLayers[j];
		end_node_internal += this->number_nodes_in_layer[j];
		
#ifdef HESSIAN_FORWARD_OUT
		testing::outputToFile<weight_type>(
			thrust::make_permutation_iterator(
			this->GPUPreviousOutput_Values.begin(),
			thrust::make_transform_iterator(
			this->GPUMapFrom.begin() + start_weight_internal,
			_1 - ((this->numberOfNodes + this->numberNonWeights)*(_1 / this->GPUPreviousOutput_Values.size()))//Subtract the nodes if they are greater than the current layer
			)
			),
			end_weight_internal - start_weight_internal,
			"test",
			"tests/alphas.txt"
			);
		testing::outputToFile<weight_type>(
			thrust::make_permutation_iterator(
			this->GPUOutput_values.begin() + start_output,
			this->GPUMapFrom.begin() + start_weight_internal
			),
			end_weight_internal - start_weight_internal,
			"output2",
			"tests/alphas.txt"
			);
#endif
		//Initial layer has x_i of 0
		// and therefore y_i of 0
		//Find the vector of the x_i
		//w_ji * R(Y_j) + v_j_i*y_j)
		thrust::reduce_by_key(
			this->GPUMapTo.begin() + start_weight_internal,
			this->GPUMapTo.begin() + end_weight_internal,
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(

			this->GPUWeights.begin() + start_weight_internal,

			thrust::make_permutation_iterator(
			this->GPUPreviousOutput_Values.begin(),
			thrust::make_transform_iterator(
			this->GPUMapFrom.begin() + start_weight_internal,
			_1 - ((this->numberOfNodes + this->numberNonWeights)*(_1/this->GPUPreviousOutput_Values.size()))//Subtract the nodes if they are greater than the current layer
			)
			),

			thrust::make_permutation_iterator(
			this->GPUOutput_values.begin() + start_output,
			this->GPUMapFrom.begin() + start_weight_internal
			),

			thrust::make_constant_iterator((int)1)
			)
			),
			functors::find_forward_x_hessian<weight_type>()
			),
			thrust::make_discard_iterator(),
			this->alphas.begin() + start + start_node_internal
			);

#ifdef HESSIAN_FORWARD_OUT
		testing::outputToFile(this->alphas, "R(X)", "tests/alphas.txt");
#endif
		start_output += this->numberOfNodes + (2*this->numberNonWeights);//Get the current layers output rather than the previous layer
		thrust::transform(
			this->alphas.begin() + start + start_node_internal,
			this->alphas.begin() + start + end_node_internal,
			this->GPUOutput_values.begin() + start_output + start_node_internal,
			this->GPUPreviousOutput_Values.begin() + start_node_internal + this->numberNonWeights,
			functors::find_forward_y_hessian<weight_type>()
			);
		
#ifdef HESSIAN_FORWARD_OUT
		testing::outputToFile(this->GPUPreviousOutput_Values, "R(Y)", "tests/alphas.txt");
#endif
		start_output -= this->numberOfNodes + (2 * this->numberNonWeights);
		start_node_internal += this->number_nodes_in_layer[j];
		start_weight_internal += this->numberOfWeightsInLayers[j];
		
		}
		end_node_internal = 0;
		end_weight_internal = 0;
		start_weight_internal = 0;
		start_node_internal = 0;
		start += this->numberOfNodes;
		start_output += this->numberOfNodes + this->numberNonWeights;
		thrust::copy(this->GPUOutput_values.begin() + start_output, this->GPUOutput_values.begin() + start_output + this->numberNonWeights, this->GPUPreviousOutput_Values.begin());
	}
	




}

void Hessian_Network::findWeightDeltas(){
	//Sum the deltas
	thrust::reduce_by_key(
		Unique_Iterator::make_repeat_iterator(thrust::make_counting_iterator(0), this->settings.i_backprop_unrolled),
		Unique_Iterator::make_repeat_iterator(thrust::make_counting_iterator(0), this->settings.i_backprop_unrolled) + (this->settings.i_backprop_unrolled * this->numberOfNodes),

		thrust::make_permutation_iterator(
		Unique_Iterator::make_repeat_list_iterator(this->device_deltas.begin(), (this->settings.i_backprop_unrolled * this->numberOfNodes)),
		thrust::make_transform_iterator(
		thrust::make_counting_iterator(0),
		(_1 / this->settings.i_backprop_unrolled) + ((_1%this->settings.i_backprop_unrolled) * this->numberOfNodes) + ((_1 / this->settings.i_backprop_unrolled) * this->numberOfNodes * this->settings.i_backprop_unrolled)

		)
		),
		thrust::make_discard_iterator(),
		this->GPUPreviousBias.begin()
		);
	testing::outputToFile(this->device_deltas, "testing", "tests/device_deltas.txt");
	testing::outputToFile(this->GPUPreviousBias, "testing", "tests/GPUPREVBIAS.txt");
	testing::outputToFile(this->GPUOutput_values, "testing", "tests/GPUOUTPUTVALUES.txt");
	//Find the deltas of the weights
	thrust::transform(
		this->GPUWeights.begin(),
		this->GPUWeights.end(),
		thrust::make_permutation_iterator(
		this->GPUPreviousBias.begin(),
		this->GPUMapTo.begin()
		),
		this->GPUPreviousWeights.begin(),
		thrust::multiplies<weight_type>()
		);
}

void Hessian_Network::findHessianFreeMatrixBackward(){
	int start_node_internal = ((this->settings.i_backprop_unrolled-1) * (this->numberOfNodes)) - this->numberOfNodes;
	int end_node_internal = ((this->settings.i_backprop_unrolled -1) * (this->numberOfNodes));
	int start_output = (this->settings.i_backprop_unrolled - 1) * (this->numberOfNodes + this->numberNonWeights) - this->numberNonWeights - this->numberOfNodes;
	int end_output = (this->settings.i_backprop_unrolled) * (this->numberOfNodes + this->numberNonWeights);
	int number_weights_start = 0;
	int number_weights_end = this->hessian.size();
	bool is_second = false;
	
	//Find R(dE/dx_i) and R(dE/dy_i) of the outputs
	//R(dE/dx_i) is stored in device_deltas, R(dE/dy_i) is stored in alphas
	
	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_skip_iterator(this->GPUOutput_values.begin() 
		+ this->numberOfNodes + this->numberNonWeights - this->settings.i_output, this->settings.i_output, this->numberOfNodes + this->numberNonWeights - this->settings.i_output),
		Unique_Iterator::make_skip_iterator(this->device_deltas.begin() 
		+ this->numberOfNodes - this->settings.i_output, this->settings.i_output, this->numberOfNodes - this->settings.i_output),
		Unique_Iterator::make_skip_iterator(this->alphas.begin() + this->numberOfNodes
		- this->settings.i_output, this->settings.i_output, this->numberOfNodes - this->settings.i_output)
		)
		),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		Unique_Iterator::make_skip_iterator(this->GPUOutput_values.begin() 
		+ this->numberOfNodes + this->numberNonWeights - this->settings.i_output, this->settings.i_output, this->numberOfNodes + this->numberNonWeights - this->settings.i_output),
		Unique_Iterator::make_skip_iterator(this->device_deltas.begin() 
		+ this->numberOfNodes - this->settings.i_output, this->settings.i_output, this->numberOfNodes - this->settings.i_output),
		Unique_Iterator::make_skip_iterator(this->alphas.begin() + this->numberOfNodes
		- this->settings.i_output, this->settings.i_output, this->numberOfNodes - this->settings.i_output)
		)
		) + ((this->settings.i_backprop_unrolled -1 ) * this->settings.i_output),
		functors::find_output_x_hessian_backward<weight_type>()
		);

	//W_ok is not calculated as there are no outgoing weights from the output and therefore have no changes to the weights can occur

	for (int i = this->settings.i_backprop_unrolled - 2; i >= 0; i--){
		
		

		for (int j = this->mBlocksLayers.size() - 2; j >= 0; j--){
			//Find R(dE/dy_i)
				thrust::reduce_by_key(
					this->count.begin(),
					this->count.end(),
					thrust::make_permutation_iterator(
					
					
					thrust::make_transform_iterator(
					thrust::make_zip_iterator(
					thrust::make_tuple(
					//W * R(dE/dx_j) + e'(y_i) * R(dE/dY_i) + v_i

					this->GPUWeights.begin(),//W
					thrust::make_permutation_iterator(
					this->device_deltas.begin() + start_node_internal,//R(dE/dx_j)
					this->GPUMapTo.begin()
					),

					thrust::make_permutation_iterator(
					this->alphas.begin() + start_node_internal,//R(dE/dY_i)
					this->GPUMapFrom.begin()
					),
					
					thrust::make_permutation_iterator(
					this->GPUOutput_values.begin() + start_output,
					this->GPUMapTo.begin()
					),
					thrust::make_constant_iterator((weight_type)1),
					
					thrust::make_permutation_iterator(
					
					this->GPUOutput_values.begin() + start_output,//y_i
					this->GPUMapFrom.begin()
					)

					)
					),
					functors::find_backward_partial_y_hessian<weight_type>(is_second)
					),
					this->positionToSum.begin()
					),
					thrust::make_discard_iterator(),
					this->GPUPreviousOutput_Values.begin()
					);

				//Find w_i
				thrust::transform(
					thrust::make_zip_iterator(
					thrust::make_tuple(
					this->GPUOutput_values.begin() + start_output + this->numberNonWeights,
					this->GPUPreviousOutput_Values.begin(),
					this->alphas.begin() + start_node_internal + this->numberOfNodes
					)
					),
					thrust::make_zip_iterator(
					thrust::make_tuple(
					this->GPUOutput_values.begin() + start_output + this->numberOfNodes + this->numberNonWeights,
					this->GPUPreviousOutput_Values.end(),
					this->alphas.begin() + start_node_internal + this->numberOfNodes
					)
					),
					this->hessian.begin(),
					this->hessian.begin(),
					functors::find_backward_partial_w_hessian<weight_type>()
					);

				//Find the value of R(dE/dx_i)
				thrust::transform(
					thrust::make_zip_iterator(
					thrust::make_tuple(
					this->GPUOutput_values.begin() + start_output - this->numberOfNodes,
					this->GPUPreviousOutput_Values.begin(),
					this->alphas.begin() + start_node_internal,
					this->device_deltas.begin() + start_node_internal
					)
					),
					thrust::make_zip_iterator(
					thrust::make_tuple(
					this->GPUOutput_values.begin() + start_output,
					this->GPUPreviousOutput_Values.begin(),
					this->alphas.begin() + end_node_internal,
					this->device_deltas.begin() + end_node_internal
					)
					),
					this->GPUPreviousOutput_Values.begin(),
					functors::find_backward_partial_x_hessian<weight_type>()
					);
			
				
				
			
		}
		start_node_internal -=this->numberOfNodes;
		end_node_internal -= this->numberOfNodes;
		start_output -= this->numberOfNodes + this->numberNonWeights;
		end_output -= this->numberOfNodes + this->numberNonWeights;
		is_second = false;
			 
		
	}

	//Find the derivative of y

}

void Hessian_Network::ApplyError(){
	//this->ApplyLongTermShortTermMemoryError();
}

void Hessian_Network::findAlpha(){
	
	testing::outputToFile(this->GPUWeights, "testing", "tests/weights.txt");
	testing::outputToFile(this->hessian, "training_output", "tests/hessian.txt");
	thrust::device_vector<weight_type> temp = thrust::device_vector<weight_type>(this->GPUPreviousWeights);
	thrust::fill(this->GPUPreviousWeights.begin() + this->numberOfWeightsInLayers[0] + (this->mBlocksLayers[0].size()), this->GPUPreviousWeights.end(), (weight_type)0);
	testing::outputToFile(this->GPUPreviousWeights, "training_output", "tests/deltas.txt");
	weight_type top;//Value of the top half
	weight_type bottom;
	for (int i = 0; i < this->settings.i_output; i++){
		//Find the top half
		top = thrust::reduce(
			
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(//Hessian, Delta of weights, delta x
			this->hessian.begin(),
			this->GPUPreviousWeights.begin(),
			this->alphas.begin()
			)
			),
			functors::find_top_alpha<weight_type>()
			),

			
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(//Hessian, Delta of weights, delta x
			this->hessian.begin(),
			this->GPUPreviousWeights.begin(),
			this->alphas.begin()
			)
			),
			functors::find_top_alpha<weight_type>()
			) + this->GPUWeights.size()
			);

		bottom = thrust::reduce(
			
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(//Hessian, Delta of weights, delta x
			this->hessian.begin(),
			this->GPUPreviousWeights.begin()
			)
			),
			functors::find_bottom_alpha<weight_type>()
			),

			
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(//Hessian, Delta of weights, delta x
			this->hessian.begin(),
			this->GPUPreviousWeights.begin()
			)
			) + this->GPUWeights.size(),
			functors::find_bottom_alpha<weight_type>()
			)
			);


		//Apply the error
		thrust::transform(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin(),
			this->GPUPreviousWeights.begin(),
			thrust::make_constant_iterator(top / bottom)
			)
			),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin(),
			this->GPUPreviousWeights.begin(),
			thrust::make_constant_iterator(top / bottom)
			)
			),
			this->GPUWeights.begin(),
			functors::apply_hessian_alpha<weight_type>()
			);
		if (i < this->settings.i_output - 1){
			//Find the new delta
			top = thrust::reduce(
				Unique_Iterator::make_skip_iterator(
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(//Hessian, Delta of weights, delta x
				this->hessian.begin(),
				this->GPUPreviousWeights.begin(),
				temp.begin()
				)
				),
				functors::find_beta<weight_type>()
				),
				this->numberOfWeightsInLayers[0], this->mBlocksLayers[0].size()),

				Unique_Iterator::make_skip_iterator(
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(//Hessian, Delta of weights, delta x
				this->hessian.begin(),
				this->GPUPreviousWeights.begin(),
				temp.begin()
				)
				),
				functors::find_beta<weight_type>()
				),
				this->numberOfWeightsInLayers[0] + ((i+1)* this->mBlocksLayers[0].size()), this->mBlocksLayers[0].size()) + this->GPUWeights.size()
				);

			thrust::transform(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUPreviousWeights.begin(),
				temp.begin(),
				thrust::make_constant_iterator(top / bottom)
				)
				),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				this->GPUPreviousWeights.begin(),
				temp.begin(),
				thrust::make_constant_iterator(top / bottom)
				)
				) + this->GPUPreviousBias.size(),
				this->GPUPreviousWeights.begin(),
				functors::find_new_beta<weight_type>()
				);
		}//End if Statement
	}
	//testing::outputToFile(this->alphas, "testing", "tests/alphas.txt");
	testing::outputToFile(this->GPUWeights, "testing", "tests/weights.txt");

}