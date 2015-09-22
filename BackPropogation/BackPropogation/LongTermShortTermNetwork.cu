#include "LongTermShortTermNetwork.cuh"
#define NUMBER_NODES_IN_MEMORY_CELL 5
#define TEST_DEBUG

//#define _DEBUG_WEIGHTS
LongTermShortTermNetwork::LongTermShortTermNetwork(){
	this->settings = CSettings();
	LongTermShortTermNetwork(this->settings);
}

LongTermShortTermNetwork::LongTermShortTermNetwork(CSettings& settings){
	this->settings = settings;
	this->initialize_network();
}

//***************************
//Initialize Network
//***************************

void LongTermShortTermNetwork::initialize_network(){
	this->weights = host_vector<weight_type>();
	this->mapTo = host_vector<int>();
	this->mapFrom = host_vector<int>();
	this->bias = host_vector<weight_type>();

	//Initialize the layers
	this->mBlocksLayers = vector<vector<Memory_Block>>();


	positionOfLastWeightToNode = vector<long>();
	this->numberNonWeights = this->settings.i_input;
	srand(time(NULL));
	this->createMemoryBlock(2);

}


//***************************
//Modify Structure Of Neuron
//***************************
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
	for (int k = this->numberNonWeights; k < this->bias.size() - this->settings.i_output; k++){
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
	if (decideTo != -1){
		this->mBlocksLayers[0][decideTo].addNewConnection(this->numberNonWeights - 1, this->mBlocksLayers[0].size() + this->numberNonWeights);
	}
	else{

	}

}

void LongTermShortTermNetwork::addNeuron(int numberNeuronsToAdd){
	this->createMemoryBlock(numberNeuronsToAdd);
}


void LongTermShortTermNetwork::createMemoryBlock(int numberMemoryCells){
	if (this->mBlocksLayers.size() == 0){
		this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one hidden layer
		this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one output layer
		for (unsigned int i = 0; i < this->settings.i_output; i++){
			this->mBlocksLayers[1].push_back(Memory_Block(numberMemoryCells + this->numberNonWeights, numberMemoryCells,Memory_Block::OUTPUT));
		}
	}

	for (int i = 0; i < numberMemoryCells; i++){
		this->mBlocksLayers[0].push_back(Memory_Block(this->settings.i_input));
	}

}

void LongTermShortTermNetwork::InitialcreateMemoryBlock(int numberMemoryCells){
	if (this->mBlocksLayers.size() == 0){
		this->mBlocksLayers.push_back(vector<Memory_Block>());
	}
	this->mBlocksLayers[0].push_back(Memory_Block(this->settings.i_input));
}



//*********************
//Run The Network
//*********************
//Multiply two values
template <typename T>
struct multiply : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple x) const{
		return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x));
	}

};

template <typename T>
struct subtract_tuple : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(const Tuple &x){
		return (thrust::get<0>(x) -thrust::get<1>(x));
	}
};

template <typename T>
struct run_memory_block_functon : public::unary_function < T, T > {


	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple &x){//Received Tuple is in the form input, output, forget, potential memory cell, memory cell value
		weight_type memory_value = sigmoid_function(thrust::get<0>(x) * thrust::get<3>(x));//Multiply the input by the potential_memory_value

		weight_type forget = (weight_type)thrust::get<2>(x);

		thrust::get<2>(x) = sigmoid_function((weight_type)thrust::get<2>(x) * (weight_type)thrust::get<4>(x)); //Multiply the forget * the old memory cell value
		thrust::get<4>(x) = thrust::get<4>(x) + memory_value + forget; //Sum the forget,input, and old cell value to get the new value the new potential memory cell value
		thrust::get<1>(x) = sigmoid_function((weight_type)thrust::get<4>(x) * (weight_type)thrust::get<1>(x)); //Multiply the new memory_cell value by the new output value 

	}

	__host__ __device__
	weight_type sigmoid_function(weight_type value){
		return (weight_type)1 / ((weight_type)1 + thrust::exp(((weight_type)-1 * value)));
	}

};

//Perform Sigmoid Operation of a Tuple
template <typename T>
struct sigmoid_tuple_functor : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>

	__host__ __device__
		T operator()(Tuple x) const{
		T z = (T)((T)thrust::get<0>(x)*(T)thrust::get<1>(x));
		z = thrust::exp(((T)-1) * z);
		return (T)1 / ((T)1 + z);
	}

};




//Perform a sigmoid function
template <typename T>
struct sigmoid_functor : public thrust::unary_function < T, T > {
	sigmoid_functor(){};

	__host__ __device__
		T operator()(const T &x) const{
		T z = thrust::exp(((T)-1) * x);
		return (T)1 / ((T)1 + z);
	}

};

void LongTermShortTermNetwork::setInput(weight_type* in){
	//Place the input into the GPU values matrix
	for (int i = 0; i < this->settings.i_input; i++){
		this->GPUOutput_values[i] = in[i];
		this->GPUPreviousOutput_Values[i] = in[i];
	}

}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	unsigned int previousnumberMBlocks = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size(); i++){

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
			multiply<weight_type>()), //Multiply the two values then run them through a sigmoid function
			thrust::make_discard_iterator(), // Discard the retrieved order, the order should be constant
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks//Store in the previous in order to not overwrite the saved values
			);

		thrust::transform(this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks + (5 * numberMBlocks),
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			sigmoid_functor<weight_type>());

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
			run_memory_block_functon<weight_type>());
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->GPUOutput_values.begin());
		
	}

	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);
	//Transform the output of the current
	thrust::transform(this->GPUOutput_values.end() - 
		(5 * this->mBlocksLayers[this->mBlocksLayers.size() - 1].size()) + this->settings.i_output,
		this->GPUOutput_values.end() - 
		(5 * this->mBlocksLayers[this->mBlocksLayers.size() - 1].size())+ this->settings.i_output + this->settings.i_output,
		toReturn.begin(), _1 * (weight_type)1);

	return toReturn;
}
//*********************
//Training the Network
//*********************
template <typename T>
struct find_error : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return thrust::pow((thrust::get<0>(x) -thrust::get<1>(x)), (T)2);
	}

};

//sum a value
template <typename T>
struct add_constant_value : public thrust::unary_function < T, T > {
	const T c;
	const unsigned int input;
	add_constant_value() : c(0), input(0){};

	add_constant_value(T _c, unsigned int _input) :c(_c), input(_input){};


	__host__ __device__
		T operator()(const T &x)const{

		if (x >= input){//The value is not an input
			return ((T)x + (T)c);
		}
		else{//The value is an input
			return x;
		}
	}


};

struct compare_plus : public thrust::unary_function < int, int > {
	int input;
	int numberIncrease;//Number to increase the number by 
	compare_plus(int max_input, int numberIncrease){
		this->input = max_input;
		this->numberIncrease = numberIncrease;
	}


	__host__ __device__
		int operator()(int &x) const{
		if (x < this->input){//Returns this directly, as it is an input
			return x;
		}
		else{
			return (x + numberIncrease);
		}
	}
};

template <typename T>
struct add_one_when_equal_to : public thrust::unary_function < T, T > {
	const T equal_to;
	const T divide_by;
	add_one_when_equal_to(T _divide_by, T _equal_to) :equal_to(_equal_to), divide_by(_divide_by){}
	__host__ __device__
		T operator()(const T &x){
		if (x >= equal_to){
			return  x;
		}
		return (x / divide_by);
	}

};

//Function is _add_to, _greater_than_this
template <typename T>
struct add_when_greater_than : public thrust::unary_function < T, T > {
	const T greater_than_this;
	const T add_to;
	add_when_greater_than(T _add_to, T _greater_than_this) :greater_than_this(_greater_than_this), add_to(_add_to){}
	__host__ __device__
		T operator()(const T &x){
		if (x >= _equal_to){
			return (x + add_to);
		}
		else{
			return x;
		}
	}

};

//Apply the error from the delta and the weight
template <typename T>
struct apply_error: public thrust::binary_function<T,T,T>{
	const T alpha;
	const T beta;
	const T divide;
	apply_error(T _alpha, T _beta, T _divide) : alpha(_alpha), beta(_beta),divide(_divide){

	}

	//w = weight, d = delta, beta * (d/(number summed) + (w + (w*alpha)
	__host__ __device__
		T operator()(const T &d, const T &w)const{
		return (beta * (d / divide)) + (w + (w*alpha));
	}
};

template <typename T>
struct find_non_output_delta : public thrust::unary_function < T, T > {


	find_non_output_delta(){};

	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &t){
		return (T)thrust::get<0>(t) * ((T)1 - (T)thrust::get<0>(t)) * (T)thrust::get<1>(t);

	}

};

void LongTermShortTermNetwork::InitializeLongShortTermMemoryForRun(){
	//Form the delta objects
	this->CopyToDevice();
}

void LongTermShortTermNetwork::InitializeLongShortTermMemory(){
	//Store all the values in the device
	//Will later add option for too little memory
	//Copy the information to the device
	this->UnrollNetwork(3);
	this->host_deltas = host_vector<weight_type>(this->GPUOutput_values.size());
	this->device_deltas = device_vector<weight_type>(this->GPUOutput_values.size());
	this->RealOutput = device_vector<weight_type>(this->settings.i_output);
}

template <typename T>
void testCopy(thrust::device_vector<T> vector,int start,int end){
	thrust::copy(vector.begin() + start, vector.begin() + end, std::ostream_iterator<T>(std::cout, "\n"));
}

void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type* in, weight_type* out){
	//Get the number of weights in the output layer
	//This is needed because the output layer needs to be used only once, so we need to inform the system which weights to skip
	
	//Set the input values
	this->setInput(in);
	unsigned int number_weights_to_beginining_of_layer = 0;
	unsigned int number_nodes_to_beginning_of_layer = 0;
	unsigned int number_weights_in_layer = this->GPUWeights.size();
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
			this->GPUMapFrom.begin(), add_constant_value<int>(number_nodes_to_beginning_of_layer, this->settings.i_input))
			)
			)
			),
			multiply<weight_type>()
			),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);

		if (i > 0){//Only increment it by the number of nodes when working from any layer which is not the initial layer
			//This lets the nodes use the previous layer as their input
			number_weights_to_beginining_of_layer += number_weights_in_layer;
			number_nodes_to_beginning_of_layer += this->numberOfNodes;
		}
		
		

		if (true){
			//Transfer all values from the current to the next row
			thrust::transform(this->GPUPreviousOutput_Values.begin(),
				this->GPUPreviousOutput_Values.end(),
				this->GPUOutput_values.begin() + number_nodes_to_beginning_of_layer + this->numberNonWeights, sigmoid_functor<weight_type>());
		}

		
	}

}

//Find the delta gradiant for each of the "layers" of the network
void LongTermShortTermNetwork::FindBackPropDelta(weight_type* out){
	//Retrieve the length of the output
	unsigned int numberCellsInLayers = this->mBlocksLayers[this->mBlocksLayers.size() - 1].size();
	unsigned int lengthOfOutput = (this->mBlocksLayers[this->mBlocksLayers.size() - 1].size() * 4) + this->getNumberMemoryCells(this->mBlocksLayers.size() - 1);
	unsigned int numberInLayers = this->mBlocksLayers[this->mBlocksLayers.size() - 1].size();
	unsigned int numberOfWeightsOfInputType = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, INPUT_CELL);
	unsigned int numberOfWeightsOfOutputType = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, OUTPUT_CELL);
	unsigned int numberOfWeightsOfForgetType = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, FORGET_CELL);
	unsigned int numberOfWeightsOfPotentialMemoryCellType = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, POTENTIAL_MEMORY_CELL);
	unsigned int numberOfWeightsOfMemoryCellType = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, MEMORY_CELL);
	unsigned int numberOfWeightsInLayer = numberOfWeightsOfInputType + numberOfWeightsOfOutputType + numberOfWeightsOfForgetType + numberOfWeightsOfPotentialMemoryCellType + numberOfWeightsOfMemoryCellType;
	unsigned int numberNodesOfSingleType = this->mBlocksLayers[this->mBlocksLayers.size() - 1].size();//Number of non memory cells in a node



	//Find the output delta
	//Start from the begining + the number of input nodes
	thrust::transform(this->RealOutput.begin(), this->RealOutput.end(), 
		this->GPUOutput_values.end() - lengthOfOutput + numberNodesOfSingleType, 
		this->device_deltas.end() - lengthOfOutput + numberNodesOfSingleType,
		_2 * (((weight_type)1) - _2) * (_1 - _2));//Output * (1- output) * (target-output)
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
	
	thrust::fill(this->device_deltas.begin(), this->device_deltas.end(), 1);

	//n,k used to form the sequence
	//placed here to make the function easier to read
	unsigned int n = numberCellsInLayers;
	unsigned int k = numberOfWeightsOfOutputType;
	//Backpropogate to the input of the output memory cells

	//Multiply the output weights by their deltas
	//deltas.end() - totalLengthOfTheOutput + #input nodes, deltas.end() - totalLengthOfTheOutput + #input nodes + #output nodes
	//The output layer has a special feature wherin the number of weights to each output is equal, and the only different value is the last one for each node
	// as such a formula can be made to place each one next to each other
	thrust::reduce_by_key(
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
		add_one_when_equal_to<int>((int)(numberOfWeightsOfOutputType / numberCellsInLayers), numberOfWeightsOfOutputType - numberCellsInLayers)),
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
		add_one_when_equal_to<int>((int)(numberOfWeightsOfOutputType / numberCellsInLayers), numberOfWeightsOfOutputType - numberCellsInLayers)) + numberOfWeightsOfOutputType,

		thrust::make_permutation_iterator(
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		this->GPUWeights.end() - numberOfWeightsInLayer + numberOfWeightsOfInputType,//Beginning of the output of the output layer memory cells
		
		thrust::make_permutation_iterator(
		this->device_deltas.begin(), //Beginning of the deltas of the output
		this->GPUMapTo.end() - numberOfWeightsInLayer + numberOfWeightsOfInputType
		)
		)
		),
		multiply<weight_type>()
		),
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (((_1%n)*k) + (_1/n)))),
		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);





	//Multiply the memory cells by their memory to get the needed values
	thrust::transform(this->GPUPreviousOutput_Values.begin(),
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size() + (numberOfWeightsOfMemoryCellType / numberCellsInLayers),
		this->GPUOutput_values.end() - (numberOfWeightsOfMemoryCellType / 3),
		this->device_deltas.end() - (numberOfWeightsOfMemoryCellType / 3),
		_2*((weight_type)1 - _2)*_1);

	//Copy the weights from the memory cell to the input/forget/potential, since all connections to the memory cell are always weight one
	// and 1 * n =n

	thrust::transform(
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size(),
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size() + (numberOfWeightsOfMemoryCellType / 3),
		this->GPUOutput_values.end() - numberOfWeightsInLayer,
		this->device_deltas.end() - lengthOfOutput,
		_2*((weight_type)1 - _2)*_1
		);


	//Forget Nodes
	thrust::transform(
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size(),
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size() + (numberOfWeightsOfMemoryCellType / 3),
		this->GPUOutput_values.end() - numberOfWeightsInLayer + numberOfWeightsOfInputType + numberOfWeightsOfOutputType,
		this->device_deltas.end() - lengthOfOutput + (2 * numberCellsInLayers),
		_2*((weight_type)1 - _2)*_1
		);

	//Potential Nodes
	thrust::transform(
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size(),
		this->GPUPreviousOutput_Values.begin() + this->mBlocksLayers[this->mBlocksLayers.size() - 2].size() + (numberOfWeightsOfMemoryCellType / 3),
		this->GPUOutput_values.end() - numberOfWeightsInLayer + numberOfWeightsOfInputType + numberOfWeightsOfOutputType + numberOfWeightsOfForgetType,
		this->device_deltas.end() - lengthOfOutput + (3 * numberCellsInLayers),
		_2*((weight_type)1 - _2)*_1
		);



	//Lengths of weights in the next layer. I.e. output layer if second layer from top
	unsigned int numberOfWeightsOfInputTypeInNextLayer = numberOfWeightsOfInputType;
	unsigned int numberOfWeightsOfOutputTypeInNextLayer = numberOfWeightsOfOutputType;
	unsigned int numberOfWeightsOfForgetTypeInNextLayer = numberOfWeightsOfForgetType;
	unsigned int numberOfWeightsOfPotentialMemoryCellTypeInNextLayer = numberOfWeightsOfPotentialMemoryCellType;
	unsigned int numberOfWeightsOfMemoryCellTypeInNextLayer = numberOfWeightsOfMemoryCellType;
	unsigned int numberOfWeightsInNextLayer = numberOfWeightsInLayer;
	unsigned int numberCellsInNextLayer = numberCellsInLayers;


	//Rempty the delta * weight holder
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);

	numberCellsInLayers = this->mBlocksLayers[this->mBlocksLayers.size() - 2].size();
	unsigned int length_to_previous_output = lengthOfOutput;
	unsigned int length_to_previous_weight = numberOfWeightsInNextLayer;
	unsigned int delta_start = lengthOfOutput;
	unsigned int end_of_count = numberOfWeightsInNextLayer;
	//Find the delta from the gradiant of each other layer in the unrolled network
	for (int i = this->settings.i_backprop_unrolled; i > 0; i--){
		
		//Lengths of weights in the next layer. I.e. output layer if second layer from top
		
		thrust::reduce_by_key(
			this->count.end() - length_to_previous_weight,
			this->count.end() - length_to_previous_weight + end_of_count,//Sum over start of layer to end of layer

			thrust::make_permutation_iterator(
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			//Permute the Output_values such that each one occurs with it's particular weight
			thrust::make_permutation_iterator(
			this->GPUOutput_values.end(),
			this->GPUMapFrom.end() - length_to_previous_weight - this->numberOfNodes),

			//Weight x Delta
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - length_to_previous_weight,//End - number of weights in next layer + the number of values which have no weight
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)))
			),
			multiply<weight_type>()
			)//Permute the deltas such that it matches the weights
			)),
			find_non_output_delta<weight_type>()),
			this->positionToSum.end() - length_to_previous_weight),
			thrust::make_discard_iterator(),
			this->GPUPreviousOutput_Values.begin()
			);



#ifdef _DEBUG_WEIGHTS		
		thrust::copy(thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)), 
			
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)) + end_of_count, std::ostream_iterator<weight_type>(std::cout, "\n"));
		
		std::cout << "______________________";

		//Weight x Delta
		thrust::copy(thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - length_to_previous_weight,//End - number of weights in next layer + the number of values which have no weight
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)))
			),
			multiply<weight_type>()
			), thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - length_to_previous_weight,//End - number of weights in next layer + the number of values which have no weight
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)))
			),
			multiply<weight_type>()
			) + end_of_count, std::ostream_iterator<weight_type>(std::cout, "\n"));

		std::cout << "______________________";

		//Weight x Delta
		thrust::copy(
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			//Permute the Output_values such that each one occurs with it's particular weight
			thrust::make_permutation_iterator(
			this->GPUOutput_values.end() - length_to_previous_weight + this->numberOfNodes,
			this->GPUMapFrom.end() - length_to_previous_weight + this->numberOfNodes),

			//Weight x Delta
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - length_to_previous_weight,//End - number of weights in next layer + the number of values which have no weight
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)))
			),
			multiply<weight_type>()
			)//Permute the deltas such that it matches the weights
			)),
			find_non_output_delta<weight_type>()),
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			//Permute the Output_values such that each one occurs with it's particular weight
			thrust::make_permutation_iterator(
			this->GPUOutput_values.end() - length_to_previous_weight + this->numberOfNodes,
			this->GPUMapFrom.end() - length_to_previous_weight + this->numberOfNodes),

			//Weight x Delta
			thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.end() - length_to_previous_weight,//End - number of weights in next layer + the number of values which have no weight
			thrust::make_permutation_iterator(
			this->device_deltas.end() - delta_start, //Start from the beginning of the previous layer
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0), _1 / numberCellsInLayers)))
			),
			multiply<weight_type>()
			)//Permute the deltas such that it matches the weights
			)),
			find_non_output_delta<weight_type>()) + end_of_count, std::ostream_iterator<weight_type>(std::cout, "\n"));

#endif

		//Increase the position of the weights
		length_to_previous_output += this->numberOfNodes;
		length_to_previous_weight += numberOfWeightsInLayer;
		delta_start += this->numberOfNodes;
		end_of_count = this->numberOfNodes;
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.begin() + this->numberOfNodes, this->device_deltas.end() - delta_start);

	}


}

//Apply the error
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	unsigned int lengthOfOutput = (this->mBlocksLayers[this->mBlocksLayers.size() - 1].size() * 4) + this->getNumberMemoryCells(this->mBlocksLayers.size() - 1);


	unsigned int numberOfWeightsInLayer = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, MEMORY_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, POTENTIAL_MEMORY_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, FORGET_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, OUTPUT_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 1, INPUT_CELL);
	unsigned int numberOfWeightsInCurrentLayer = getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 2, MEMORY_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 2, POTENTIAL_MEMORY_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 2, FORGET_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 2, OUTPUT_CELL) + getNumberTypeWeightsInLayer(this->mBlocksLayers.size() - 2, INPUT_CELL);
	unsigned int numberMemoryCellsOutput = this->getNumberMemoryCells(this->mBlocksLayers.size() - 1);
	unsigned int numberMemoryCellsLayers = this->getNumberMemoryCells(this->mBlocksLayers.size() - 2);
	//Average the deltas
	thrust::reduce_by_key(
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
		_1 / this->settings.i_backprop_unrolled
		),
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
		_1 / this->settings.i_backprop_unrolled
		) + this->device_deltas.size() - lengthOfOutput,

		make_permutation_iterator(this->device_deltas.begin() + this->numberNonWeights,
		thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
		((_1%this->settings.i_backprop_unrolled) * this->numberOfNodes) + (_1/this->settings.i_backprop_unrolled))
		),

		thrust::make_discard_iterator(),
		this->GPUPreviousOutput_Values.begin()
		);

	//Subtract the deltas from the weights from each non-output nodes
	thrust::transform(
		this->GPUWeights.begin() + this->numberNonWeights,
		this->GPUWeights.begin() + this->numberNonWeights + numberOfWeightsInCurrentLayer - numberMemoryCellsLayers,
		make_permutation_iterator(
		this->GPUPreviousOutput_Values.begin(), 
		this->GPUMapTo.begin() + this->numberNonWeights
		),
		this->GPUWeights.begin() + this->numberNonWeights,
		//Beta * (average of delta) + (weights + (weights * alpha))
		apply_error<weight_type>((weight_type)this->settings.d_alpha, (weight_type)this->settings.d_beta, (weight_type)this->settings.i_backprop_unrolled)
		);


	//Subtract the deltas from the weights of the output
	thrust::transform(
		this->GPUWeights.end() - numberOfWeightsInLayer ,
		this->GPUWeights.end() - numberMemoryCellsOutput,
		make_permutation_iterator(this->device_deltas.begin(),
		this->GPUMapTo.end() - numberOfWeightsInLayer),
		this->GPUWeights.end() - numberOfWeightsInLayer, 
		apply_error<weight_type>((weight_type)this->settings.d_alpha, (weight_type) this->settings.d_beta, (weight_type)1)
		);




}

//*********************
//Perform Functionality
//*********************

void LongTermShortTermNetwork::UnrollNetwork(int numLayers){
	vector<vector<Memory_Block>> Unrolled_Layers = vector<vector<Memory_Block>>();//Storage of the memory blocks as new layers
	this->numberOfNodes = 0;
	//Add room for the intial input values
	this->GPUOutput_values.resize(this->numberNonWeights);
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){
		this->loadUnrolledToDevice(2, i);
	}

	//Unroll the output layer only once
	//The output layer will contain only n node (n is the number of output) and will merely sum all input passed into it
	//This makes performing analysis far easier than using a extra layer of memory cells
	this->loadUnrolledToDevice(2, this->mBlocksLayers.size()-1);
	//Expand the output container
	
	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size() - this->numberNonWeights);


	int GPUOutput_values_size = this->GPUOutput_values.size();

	//Resize the network to contain locations for the other layer
	this->GPUOutput_values.resize(this->GPUOutput_values.size() + ((this->settings.i_backprop_unrolled - 1)*(this->GPUOutput_values.size() - this->numberNonWeights)));
	
	this->getSumPermutation();



	//Create an empty array for the current values in the network
	this->ResetSequence();
}

void LongTermShortTermNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
}

template <typename T>
void copyValuesToHost(int start, device_vector<T> &GPU_Vector, host_vector<T> &local_host_Vector){
	//Copy the values into the network
	thrust::copy(GPU_Vector.begin() + start, GPU_Vector.begin() + local_host_Vector.size() + start, local_host_Vector.begin());
}



void LongTermShortTermNetwork::CopyToHost(){
	//Copy the device memory to local
	this->output_bias.resize(this->GPUOutput_values.size());
	this->bias.resize(this->GPUPreviousOutput_Values.size());
	//Copy the output
	thrust::copy(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), this->output_bias.begin());
	//Copy the secondary output
	thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->bias.begin());
	int start = this->settings.i_input;
	for (int j = 0; j < this->mBlocksLayers.size(); j++){
		//Copy back the input
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].input_weights);
			start += this->mBlocksLayers[j][i].input_weights.size();
		}
		//Copy back to output
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].output_weights);
			start += this->mBlocksLayers[j][i].output_weights.size();
		}

		//Copy back to forget
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].forget_weights);
			start += this->mBlocksLayers[j][i].forget_weights.size();
		}

		//Copy back to potential memory cell
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].potential_memory_cell_value);
			start += this->mBlocksLayers[j][i].potential_memory_cell_value.size();
		}

		//Get the new memory_cell_weight
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][i].memory_cell_weights.size() > 0){
			this->mBlocksLayers[j][i].memory_cell_weights[0] = 0;
			}
		}
	}
}

template <typename T>
void LongTermShortTermNetwork::copyNodesToDevice(device_vector<T> &GPU_Vector, device_vector<int> &fromPosition, host_vector<T> &local_host_Vector, host_vector<int> host_from_vector){
	int GPU_VecSize = GPU_Vector.size();
	int GPUFromPos = fromPosition.size();
	GPU_Vector.resize(GPU_VecSize + local_host_Vector.size());
	fromPosition.resize(GPUFromPos + local_host_Vector.size());
	//Copy the values into the network
	thrust::copy(local_host_Vector.begin(), local_host_Vector.end(), GPU_Vector.begin() + GPU_VecSize);
	thrust::copy(host_from_vector.begin(), host_from_vector.end(), fromPosition.begin() + GPUFromPos);
}

//Copies only the input and not the device
template <typename T>
void LongTermShortTermNetwork::specialCopyToNodes(int start_output, int number_output, device_vector<T> &GPUWeightVector, device_vector<int> &toPosition, device_vector<int> &fromPosition, host_vector<T> &local_weights, host_vector<int> map){
	int GPU_VecSize = GPUWeightVector.size();
	int GPUFromPos = fromPosition.size();

	//We need to store the number a special copy of a map, such that it has input from both the input of the sequence and the output of the previous layer
	GPUWeightVector.resize(GPU_VecSize + local_weights.size());
	fromPosition.resize(GPUFromPos + map.size());
	thrust::copy(map.begin(), map.end(), fromPosition.begin() + GPUFromPos);
	thrust::copy(local_weights.begin(), local_weights.end(), GPUWeightVector.begin() + GPU_VecSize);
	GPU_VecSize = GPU_VecSize + local_weights.size();

	for (unsigned int i = start_output; i < start_output + number_output; i++){
		fromPosition.push_back(i);
		GPUWeightVector.push_back(1);
	}

	toPosition.resize(fromPosition.size());
	thrust::fill(toPosition.end() - (local_weights.size() + number_output), toPosition.end(), this->GPUOutput_values.size());

}

void LongTermShortTermNetwork::loadUnrolledToDevice(int type_of_row, unsigned int j){
	//We need to keep track of the end of the number of inputs in order to add in a connection to the outputs for the next level
	unsigned int start_output_position = 0;
	unsigned int number_output_to_add = 0;
	unsigned int* input_nodes = new unsigned int[this->mBlocksLayers[j].size() * 3];
	host_vector<int> memory_cell_from = host_vector<int>(4);
	host_vector<weight_type> memory_cell_weights = host_vector<weight_type>(4);
	memory_cell_weights[0] = 1;//From the input
	memory_cell_weights[1] = 1;//From the potential
	memory_cell_weights[2] = 1;//From the forget
	memory_cell_weights[3] = 1;//From itself

	if (type_of_row == 0){//Is not an output row
		number_output_to_add = this->mBlocksLayers[j].size();
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			start_output_position = this->mBlocksLayers[j][i].input_weights.size();
		}



		//Increment it by the input numbers
		start_output_position += this->settings.i_input - 1;
	}

	//Set all the input values
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
			input_nodes[i] = this->GPUOutput_values.size();//Get the position of an input node
			if (type_of_row == 2){
				this->numberOfNodes++;
			}
			this->GPUOutput_values.push_back(0);
		}
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
			//Add a connection to the memory cell
			this->weights.push_back(1);//Push back a 1 for the multiplication value
			input_nodes[i + this->mBlocksLayers[j].size()] = this->GPUOutput_values.size() + (this->mBlocksLayers[j].size() * 2) + this->mBlocksLayers[j].size();//Store the position of the memory cell for the forget node
			this->GPUMapFrom.push_back(input_nodes[i + this->mBlocksLayers[j].size()]);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
			if (type_of_row == 2){
				this->numberOfNodes++;
			}
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			if (type_of_row == 2){
				this->numberOfNodes++;
			}

			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
			this->weights.push_back(1);//Push back a 1 for the multiplication value
			this->GPUMapFrom.push_back(input_nodes[i + this->mBlocksLayers[j].size()]);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
			input_nodes[i + this->mBlocksLayers[j].size()] = this->GPUOutput_values.size();//Set position of the forget blocks
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the potential_output nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		
		input_nodes[i + (this->mBlocksLayers[j].size() * 2)] = this->GPUOutput_values.size();//Store the position of the 
		
		if (type_of_row == 2){
			this->numberOfNodes++;
		}

		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		
		this->GPUOutput_values.push_back(0);
	}

	//Set the values of the Memory Cells
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			memory_cell_from[0] = input_nodes[i];//Get input in
			memory_cell_from[1] = input_nodes[i + this->mBlocksLayers[j].size()];//The potential input
			memory_cell_from[2] = input_nodes[i + (this->mBlocksLayers[j].size() * 2)];
			memory_cell_from[3] = this->GPUOutput_values.size();//itself

			if (type_of_row == 2){
				this->numberOfNodes++;
			}

			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, memory_cell_weights, memory_cell_from);
			this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
			this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
		}
	}

	free(input_nodes);
}


void LongTermShortTermNetwork::loadLayerToDevice(unsigned int j){

	//Add a place for the input
	if (j == 0){
		this->GPUOutput_values.push_back(0);
		this->GPUPreviousOutput_Values.push_back(0);
	}

	//Set all the input values
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].input_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].input_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].output_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].output_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].forget_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].forget_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	//Set all the potential_output nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].potential_memory_cell_value.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].potential_memory_cell_value.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		
		this->GPUOutput_values.push_back(0);
	}

	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size());
	//Set the values of the Memory Cells
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
			this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
		}
	}
}

void LongTermShortTermNetwork::CopyToDevice(){
	this->device_deltas = device_vector<weight_type>();
	this->GPUMapTo = device_vector<int>();
	this->GPUMapFrom = device_vector<int>();
	this->GPUOutput_values = device_vector<weight_type>();
	this->GPUPreviousOutput_Values = device_vector<weight_type>();
	this->GPUWeights = device_vector<weight_type>();

	//Set the input values to 0
	this->GPUOutput_values.resize(this->numberNonWeights);
	for (unsigned int j = 0; j < this->mBlocksLayers.size(); j++){
		this->loadLayerToDevice(j);
	}

}

unsigned int LongTermShortTermNetwork::getNumberMemoryCells(unsigned int layer){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}
	unsigned int memory_cell_count = 0;
	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		memory_cell_count += this->mBlocksLayers[layer][i].number_memory_cells;
	}
	return memory_cell_count;

}

unsigned int LongTermShortTermNetwork::getNumberWeightsInLayer(unsigned int layer){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}

	unsigned int weights_count = 0;
	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		weights_count += this->mBlocksLayers[layer][i].input_weights.size();
		weights_count += this->mBlocksLayers[layer][i].forget_weights.size();
		weights_count += this->mBlocksLayers[layer][i].output_weights.size();
		weights_count += (this->mBlocksLayers[layer][i].number_memory_cells * 2);//2 is because there is a weight between the memory cell, the forget node, and the output node
		weights_count += (this->mBlocksLayers[layer][i]).number_memory_cells * 3; //The number of weights from the input,potential, and the forget node
	}
	return weights_count;
}

//Returns number of weights going to the type of node in the layer
unsigned int LongTermShortTermNetwork::getNumberTypeWeightsInLayer(unsigned int layer, cell_type cell){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}
	unsigned int number_types = 0;

	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		switch (cell){
		case MEMORY_CELL:
			number_types += (this->mBlocksLayers[layer][i].number_memory_cells) * 3;
			break;
		case POTENTIAL_MEMORY_CELL:
			number_types += (this->mBlocksLayers[layer][i].potential_memory_cell_value.size());
			break;
		case FORGET_CELL:
			number_types += (this->mBlocksLayers[layer][i].forget_weights.size()) + this->mBlocksLayers[layer][i].number_memory_cells;
			break;
		case INPUT_CELL:
			number_types += this->mBlocksLayers[layer][i].input_weights.size();
			break;
		case OUTPUT_CELL:
			number_types += this->mBlocksLayers[layer][i].output_weights.size() + this->mBlocksLayers[layer][i].number_memory_cells;//+1 for the number of memory cell connections
			break;
		}

	}



	return number_types;

}
void LongTermShortTermNetwork::getSumPermutation(){
	//Create a permutation list containing a list of object
	this->positionToSum = thrust::device_vector<int>();
	this->count = thrust::device_vector<int>();
	unsigned int weights[5];
	unsigned int start = 0;
	unsigned int counter = 0;
	unsigned int length = 0;
#ifdef  _DEBUG
	vector<int> temp = vector<int>();
#endif
	for (int k = 0; k < this->mBlocksLayers.size(); k++){//For Each Layer
		weights[0] = getNumberTypeWeightsInLayer(k, INPUT_CELL);
		weights[1] = getNumberTypeWeightsInLayer(k, OUTPUT_CELL);
		weights[2] = getNumberTypeWeightsInLayer(k, FORGET_CELL);
		weights[3] = getNumberTypeWeightsInLayer(k, POTENTIAL_MEMORY_CELL);
		weights[4] = getNumberTypeWeightsInLayer(k, MEMORY_CELL);
		if (k == this->mBlocksLayers.size() - 1){
			length = (weights[0] + weights[1] + weights[2] + weights[3] + weights[4]);
			start = this->GPUMapFrom.size() - (weights[0] + weights[1] + weights[2] + weights[3] + weights[4]);
		}
		else{
			length = this->numberOfNodes;
		}

		
			for (int i = start; i < start + length ; i++){
				for (int j = i; j < start + length ; j++){
					if (this->GPUMapFrom[i] == this->GPUMapFrom[j]){
						this->positionToSum.push_back(j - start);
						this->count.push_back(counter);
#ifdef _DEBUG
						temp.push_back(j - start);
#endif
						
					}
				}
				
				counter++;
			}
			start += length;
		}
		
	

}

void  LongTermShortTermNetwork::cleanNetwork(){
	this->CopyToHost();
	//Free the used memory
	this->emptyGPUMemory();
}

void LongTermShortTermNetwork::emptyGPUMemory(){
	clear_vector::free(this->GPUMapFrom);
	clear_vector::free(this->GPUMapTo);
	clear_vector::free(this->GPUWeights);
	clear_vector::free(this->device_deltas);
	clear_vector::free(this->GPUOutput_values);
	clear_vector::free(this->GPUPreviousOutput_Values);
	clear_vector::free(this->positionToSum);
	clear_vector::free(this->count);
}
//*********************
//Misc
//*********************
void LongTermShortTermNetwork::VisualizeNetwork(){
	cout << *this;
}

ostream& LongTermShortTermNetwork::OutputNetwork(ostream& os){
	os << *this;
	return os;
}