#include "CGraphicNetwork.cuh"


CGraphicsNetwork::CGraphicsNetwork()
{
}

CGraphicsNetwork::CGraphicsNetwork(vector<int> &sizes)
{
	//Set the number of inputs
	this->I_input = sizes.at(0);

	//Set the number of outputs
	this->I_output = sizes.back();

	//Get the number of layers
	this->v_num_layers = sizes.size();



	//Set the number of layers
	this->v_layers.resize(this->v_num_layers);

	//Create a temporary location for new neuron
	SNeuron tempNeuron;

	//Seed the random
	srand((unsigned)(time(NULL)));

	//Assign the number of Neuron Layers

	//Create each layer
	for (int i = 0; i < this->v_num_layers; i++){//Travel through layers


		//Create a new Layer
		this->v_layers[i] = SNeuronLayer();

		//Create area to store delta values
		this->v_layers[i].delta = thrust::host_vector<double>(sizes.at(i));

		//Set the number nuerons in the current layer
		this->v_layers[i].number_per_layer = sizes.at(i);

		//Set the size of the output array
		this->v_layers[i].output.resize(sizes.at(i));

		//Randomly create a bias for each of the neurons
		for (int j = 0; j < sizes.at(i); j++){//Travel through neurons
			this->v_layers[i].neurons.push_back(SNeuron());

			//Increase the count on the total number of nodes
			this->total_num_nodes++;

			//Add the bias (Random Number between 0 and 1)
			this->v_layers[i].neurons[j].bias = RandomClamped();


			if (i > 0){//Only add weights to non-input layers
				//Add the weights
				for (int k = 0; k < sizes.at(i - 1); k++){//Number of neurons in next layer used as number of outgoing outputs
					this->v_layers[i].neurons[j].weights.push_back(RandomClamped());//Add a random weight between 0 and 1
					this->v_layers[i].neurons[j].previousWeight.push_back(0);//Set previous weight to 0
				}
			}
			else{//The input layer
				this->v_layers[i].neurons[j].weights.push_back(RandomClamped());//Add a random weight between 0 and 1
				this->v_layers[i].neurons[j].previousWeight.push_back(0);//Set previous weight to 0
			}

			//Set the initial delta to 0
			this->v_layers[i].neurons[j].delta = 0;

			//Set the initial previousbias to 0
			this->v_layers[i].neurons[j].previousBias = 0;

		}
	}
}

CGraphicsNetwork::CGraphicsNetwork(vector<int> &sizes, double beta, double alpha) :CGraphicsNetwork(sizes){
	this->beta = beta;
	this->alpha = alpha;
}

//Needs Testing
//TODO Use up a tiny bit of memory to create a pointer to the different objects which are used multiple times


void CGraphicsNetwork::feedForward(double *in){

	//Store the sumation from the previous layer


	//Store the input in the input layer
	//Allows future calculations to be performed easier
	for (int i = 0; i < this->v_layers[0].number_per_layer; i++){
		this->v_layers[0].output[i] = in[i];
	}
#ifdef TRIAL5
	//Define a holder for the output
	thrust::host_vector<double> output;
	int failure;
#endif

	//Perform the following actions on each hidden layer
	for (int i = 1; i < this->v_num_layers; i++){
#ifdef TRIAL5
		double sum;
		output.resize(this->v_layers[i].number_per_layer);

		//For each neuron in the current layer
		//take the output of the previous layer
		//and perform the calculation on it
		for (int j = 0; j < this->v_layers[i].number_per_layer; j++){
			if (!checkNeuronRemoved(this->v_layers[i].neurons[j])){//The current node has not been removed, use it
				sum = 0.0;//Reset the sum
				//For input from each neuron in the preceding layer
				for (int k = 0; k < this->v_layers[i - 1].number_per_layer; k++){
					if (!checkNeuronRemoved(this->v_layers[i - 1].neurons[k])){//The neuron in the previous layer has not been removed, add it
						//Add the output from the nodes from the previous layer times the weights for that neuron on the current layer
						sum += this->v_layers[i - 1].output[k]*this->v_layers[i].neurons[j].weights[k];
						cout << this->v_layers[i - 1].output[k] * this->v_layers[i].neurons[j].weights[k] << endl;
					}
				}

				//Apply the bias
				sum += this->v_layers[i].neurons[j].bias;
				cout << sum << endl;
				cout << "___________________" << endl;
				//Apply the sigmoid function
				output[j] = CGraphicsNetwork::sigmoid(sum);
				//Possibly Temporary
				//States if the neuron has been activated
				if (isNeuronActivated(this->v_layers[i].neurons[j])){
					this->v_layers[i].neurons[j].activated += 1;
				}
			}
		}

#endif
		//Get the feedforward value
		feedForwardGPU(this->v_layers[i], this->v_layers[i - 1]);

#ifdef TRIAL5
		failure = 0;
		//Test the output
		for (int m = 0; m < this->v_layers[i].output.size(); m++){
			if (this->v_layers[i].output[m] != output[m]){
				cout.precision(20);
				cout << this->v_layers[i].output[m] << " | " << output[m] << endl;
				failure++;
			}

		}
		cout << failure << endl;
#endif

	}



}





void CGraphicsNetwork::backprop(double *in, double *tgt){


	//Host_Vector containing the current target
	thrust::host_vector<double> target_vector;

	//Vector containing output of the results
	thrust::host_vector<double> output_vector;

	thrust::host_vector<double> delta_vector;

	//Set the size of the target vector
	target_vector = thrust::host_vector<double>(this->I_output);

	//Perform the feedforward algorithm to retrieve the output of 
	//each node in the network
	this->feedForward(in);



	//Check if results were successful
	updateSuccess(tgt);

	//Copy the target values into the target vector for use in processing by the GPU
	for (int i = 0; i < this->I_output; i++){
		target_vector[i] = tgt[i];
	}

#if defined(TRIAL2) || defined(TRIAL1)|| defined(TRIAL3) || defined(TRIAL4)
	double sum;
	int count_fail = 0;
	thrust::host_vector<double> weights;
	//Stores the current neuron
	SNeuron *currentNeuron;
	thrust::host_vector<double> delta = thrust::host_vector<double>(this->v_layers[this->v_num_layers - 1].delta.size());
#endif

#ifdef TRIAL1

	//Find Delta for the output Layer
	//The required change to have the correct answer
	for (int i = 0; i < this->v_layers[this->v_num_layers - 1].number_per_layer; i++){

		//Store a pointer to the variable
		currentNeuron = &(this->v_layers[this->v_num_layers - 1].neurons[i]);
		delta[i] = this->v_layers[this->v_num_layers - 1].output[i] * (1 - this->v_layers[this->v_num_layers - 1].output[i]) * (tgt[i] - this->v_layers[this->v_num_layers - 1].output[i]);


	}//Removed due to better implementation
#endif

	//Find the delta of the output layer
	findOutputDelta(this->v_layers[this->v_num_layers - 1].getOutput(), target_vector);

#ifdef TRIAL1

	for (int i = 0; i < delta.size(); i++){
		if (delta[i] != target_vector[i]){
			count_fail++;
		}
	}
	cout << count_fail << endl;
#endif

	this->v_layers[this->v_num_layers - 1].delta = target_vector;

	//Find the delta for the hidden layers
	for (int layerPosition = this->v_num_layers - 2; layerPosition > 0; layerPosition--){


#ifdef TRIAL2
		count_fail = 0;
		delta.resize(this->v_layers[layerPosition].neurons.size());
		for (int j = 0; j < this->v_layers[layerPosition].number_per_layer; j++){
			sum = 0.0;

			//Find the delta for the current neuron
			for (int k = 0; k < this->v_layers[layerPosition + 1].number_per_layer; k++){

				//Delta * each weight of the neuron
				sum += this->v_layers[layerPosition + 1].delta[k] *
					this->v_layers[layerPosition + 1].neurons[k].weights[j];
				cout.precision(20);
				cout << this->v_layers[layerPosition + 1].delta[k] *
					this->v_layers[layerPosition + 1].neurons[k].weights[j] << endl;

			}
			cout << "___________________" << endl;
			cout << sum << endl;
			delta[j] = this->v_layers[layerPosition].output[j] * (1 - this->v_layers[layerPosition].output[j]) * sum;
		}
		cout << "_____________________" << endl;
#endif

		//Retrieve the changed delta and store in the output vector
		findHiddenDelta(this->v_layers[layerPosition + 1], this->v_layers[layerPosition]);

#ifdef TRIAL2

		for (int i = 0; i < delta.size(); i++){
			if (delta[i] != this->v_layers[layerPosition].delta[i]){
				std::cout.precision(20);
				cout << delta[i] << " | " << this->v_layers[layerPosition].delta[i] << endl;
				count_fail++;
			}
		}
		cout << count_fail << endl;

#endif
	}

	//Find Delta for the hidden layers
	//The change needed to recieve the correct answer
	//All Layers except input and output
	/*for (int layerPosition = this->v_num_layers - 2; layerPosition > 0; layerPosition--){

	}*/

	//Apply the momentum
	//Does nothing if alpha = 0;
	if (this->alpha != 0){
		for (int layerPos = 1; layerPos < this->v_num_layers; layerPos++){

#ifdef TRIAL3
			weights.clear();
			for (int neuronPos = 0; neuronPos < this->v_layers[layerPos].number_per_layer; neuronPos++){


				//Apply the alpha to each weight
				for (int weightPos = 0; weightPos < this->v_layers[layerPos].neurons[0].weights.size(); weightPos++){
					weights.push_back(this->v_layers[layerPos].neurons[neuronPos].weights[weightPos] + (this->alpha * this->v_layers[layerPos].neurons[neuronPos].previousWeight[weightPos]));
				}

				//Add the bias
				weights.push_back(this->v_layers[layerPos].neurons[neuronPos].bias + (this->alpha * this->v_layers[layerPos].neurons[neuronPos].previousBias));
			}

#endif

			applyMomentum(this->v_layers[layerPos], this->alpha);

#ifdef TRIAL3 
			int position2 = 0;
			int wrong_count = 0;
			for (int neuronPos = 0; neuronPos < this->v_layers[layerPos].number_per_layer; neuronPos++){
				for (int weightPos = 0; weightPos < this->v_layers[layerPos - 1].number_per_layer; weightPos++){
					if (this->v_layers[layerPos].neurons[neuronPos].weights[weightPos] != weights[position2]){
						cout << this->v_layers[layerPos].neurons[neuronPos].weights[weightPos] << " | " << weights[position2] << endl;
						wrong_count++;
					}
					position2++;

				}

				if (this->v_layers[layerPos].neurons[neuronPos].bias != weights[position2]){
					cout << this->v_layers[layerPos].neurons[neuronPos].bias << " | " << weights[position2] << endl;
					wrong_count++;
				}
				position2++;

			}
			cout << wrong_count << endl;
#endif
		}

	}

#ifdef TRIAL4

	thrust::host_vector<double> previousWeights;
	double previousWeights3;

	//Apply the correction
	for (int layerNum = 1; layerNum < this->v_num_layers; layerNum++){
		weights.clear();
		previousWeights.clear();

		for (int neuronPos = 0; neuronPos < this->v_layers[layerNum].number_per_layer; neuronPos++){
			if (!checkNeuronRemoved(this->v_layers[layerNum].neurons[neuronPos]) && !checkNeuronLocked(this->v_layers[layerNum].neurons[neuronPos])){//Check if Neuron is temp removed

				for (int weightPos = 0; weightPos < this->v_layers[layerNum - 1].number_per_layer; weightPos++){

					//Check if the weight should be updated based on previous layer neuron availability
					if (!checkNeuronRemoved(this->v_layers[layerNum - 1].neurons[weightPos])){

						//BETA * delta * output
						previousWeights3 = this->beta * this->v_layers[layerNum].delta[neuronPos] *
							this->v_layers[layerNum - 1].output[weightPos];
						previousWeights.push_back(previousWeights3);

						weights.push_back(this->v_layers[layerNum].neurons[neuronPos].weights[weightPos] +
							previousWeights3);
					}
				}

				previousWeights3 = this->beta * this->v_layers[layerNum].delta[neuronPos];
				previousWeights.push_back(this->beta * this->v_layers[layerNum].delta[neuronPos]);

				weights.push_back(this->v_layers[layerNum].neurons[neuronPos].bias +
					previousWeights3);
			}


		}


		applyCorrection(this->v_layers[layerNum], this->v_layers[layerNum - 1].getOutput(1, 1), this->beta);
		int position = 0;
		count_fail = 0;
		for (int j = 0; j < this->v_layers[layerNum].number_per_layer; j++){
			for (int i = 0; i < this->v_layers[layerNum].neurons[j].previousWeight.size(); i++){
				if (this->v_layers[layerNum].neurons[j].previousWeight[i] != previousWeights[position]
					|| this->v_layers[layerNum].neurons[j].weights[i] != weights[position]
					){
					cout.precision(10);
					cout << " " << this->v_layers[layerNum].neurons[j].previousWeight[i] << " | " << previousWeights[position] << endl
						<< this->v_layers[layerNum].neurons[j].weights[i]
						<< " | " << weights[position] << endl << layerNum << "," << j << "," << i << endl;
					count_fail++;
				}

				position++;
			}

			if (this->v_layers[layerNum].neurons[j].bias != weights[position] || this->v_layers[layerNum].neurons[j].previousBias != previousWeights[position]){
				count_fail++;
			}
			position++;
		}

		cout << count_fail << endl;

	}
#else
	for (int layerNum = 1; layerNum < this->v_num_layers; layerNum++){
		applyCorrection(this->v_layers[layerNum], this->v_layers[layerNum - 1].getOutput(1, 1), this->beta);
	}

#endif


}

//**********************************************
//Add and Remove Layers and Neurons
//**********************************************

//Add a new neuron which causes will not activate until after
// it is taught at least once
//By keeping the neuron non active, the neural network should be able to better 
//update the values
void CGraphicsNetwork::addNeuronToLayer(int layerPositionStart, int layerPositionEnd, int numToAdd){
	int layerPosition = 1;
	int minNeurons = 0;
	//Can't add neurons to non-hidden layers
	//Changing the number of inputs would change the value to greatly
	//as would changing the number of outputs
	//Special version later maybe
	if (layerPositionStart < 1 || layerPositionStart >= (int) this->v_layers.size() - 1){
		//Change the position to the layer below the output
		layerPositionStart = this->v_layers.size() - 1;
	}

	//Add the new Neurons to the one which can get the most use out of them
	//TODO figure out a good algorithm
	for (int i = layerPositionStart; i < layerPositionEnd; i++){
		if (this->v_layers[i].number_per_layer < minNeurons || minNeurons == 0){
			minNeurons = this->v_layers[i].number_per_layer;
			layerPosition = i;
		}
	}

	//Add a new delta
	this->v_layers[layerPosition].delta.resize(this->v_layers[layerPosition].delta.size() + numToAdd);

	//Add new output
	this->v_layers[layerPosition].output.resize(this->v_layers[layerPosition].output.size() + numToAdd);

	//Add the new Neuron
	for (int i = 0; i < numToAdd; i++){
		SNeuron tempNeuron = SNeuron();
		//Add the weights
		//TODO find a better algorithm for deciding the weight
		for (int k = 0; k < this->v_layers[layerPosition - 1].number_per_layer; k++){//Number of neurons in next layer used as number of outgoing outputs
			tempNeuron.weights.push_back(this->v_layers[layerPosition].neurons[i].weights[k] / 2);//Add a random weight between 0 and 1
			this->v_layers[layerPosition].neurons[i].weights[k] = this->v_layers[layerPosition].neurons[i].weights[k] / 2;//Set the weight to half so that it takes the results
			tempNeuron.previousWeight.push_back(0);//Set previous weight to 0
		}

		//Add the bias (Random Number between 0 and 1)
		tempNeuron.bias = RandomClamped();

		//Set the initial delta to 0
		tempNeuron.delta = 0;

		//Set the initial previousbias to 0
		tempNeuron.previousBias = 0;

		//Add the new neuron
		this->v_layers[layerPosition].neurons.push_back(tempNeuron);

		//Add one neuron to the count
		this->v_layers[layerPosition].number_per_layer += 1;


	}

	//Add a new weight for the new node on the next level
	this->v_layers[layerPosition + 1].addNewWeights(numToAdd);
	//Increase the count on the total number of nodes
	this->total_num_nodes += numToAdd;
}

//Create a new layer with no effect on the current output of the network
//By utilizing a no change new layer, the system can learn new values while 
//leaving the previous layer unchanged
void CGraphicsNetwork::addLayer(int position, int neuronPerLayer){

	//Create iterator for insertion
	vector<SNeuronLayer>::iterator it;

	//Add a new layer below the output layer
	//Used to deal with negative values and overly large values
	if (position < 0 || position >= (int) this->v_layers.size()){
		//Change the position to the output layer position
		position = this->v_layers.size() - 1;
	}

	//Add a new layer at the given position
	it = this->v_layers.begin() + position;

	//Insert the new layer
	this->v_layers.insert(it, SNeuronLayer());

	//Create area to store delta values
	this->v_layers[position].delta = thrust::host_vector<double>(neuronPerLayer);

	//Add new output
	this->v_layers[position].output = thrust::host_vector<double>(neuronPerLayer);

	//Set the number nuerons in the current layer
	this->v_layers[position].number_per_layer = neuronPerLayer;




	//Create a temporary location for new neuron
	SNeuron tempNeuron;

	//Randomly create a bias for each of the neurons
	for (int j = 0; j < neuronPerLayer; j++){//Travel through neurons

		//Create a new Neuron
		tempNeuron = SNeuron();

		//Add the weights
		if (position > 0){
			for (int k = 0; k < this->v_layers[position - 1].number_per_layer; k++){//Number of neurons in next layer used as number of outgoing outputs
				tempNeuron.weights.push_back(RandomClamped());//Add a random weight between 0 and 1
				tempNeuron.previousWeight.push_back(0);//Set previous weight to 0
			}
		}

		this->v_layers[position + 1].keepXWeights(1);

		//Add the bias (Random Number between 0 and 1)
		tempNeuron.bias = RandomClamped();

		//Set the initial delta to 0
		tempNeuron.delta = 0;

		//Set the initial previousbias to 0
		tempNeuron.previousBias = 0;

		//Create a new neuron with a provided bias
		this->v_layers[position].neurons.push_back(tempNeuron);

		//Reset the number of layer
		this->v_num_layers = this->v_layers.size();
	}

	//Increase the count on the total number of nodes
	this->total_num_nodes += neuronPerLayer;
}

//TODO - Update neuron removal to actually remove a neuron
void CGraphicsNetwork::removeNeuron(int layerPosition, int neuronPosition){

	if (layerPosition >= this->v_num_layers || layerPosition < 0){//Layer doesn't exist
		throw 20;//Out of bound layer
	}
	else if (neuronPosition >= this->v_layers[layerPosition].number_per_layer || neuronPosition < 0){
		throw 21; //Out of bound Neuron
	}
	else{



		SNeuron &temp_neuron = this->v_layers[layerPosition].neurons[neuronPosition];//Retrive the current neuron

		if (temp_neuron.removed == 0){
			temp_neuron.removed = 1;
		}
		else{
			//TODO - add permanent removal function
		}
	}

}

void CGraphicsNetwork::removeLayer(int layerPosition){

}