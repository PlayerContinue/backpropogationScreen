#include "modifyNetwork.cuh"

modifyNetwork::modifyNetwork(){
	this->buildNetwork();
}

double* modifyNetwork::getLayerOutputUse(){
	return this->getLayerOutput();
}

double modifyNetwork::getLayerForValue(int i){
	return this->getValueForLayer(i);
}


void modifyNetwork::buildNetwork(){
		CSettings settings = CSettings();
		modifyNetwork temp1 = modifyNetwork();
		settings.d_alpha = 0;
		settings.d_beta = .1;
		vector<int> vec = vector<int>();
		vec.push_back(2);
		vec.push_back(3);
		vec.push_back(3);
		vec.push_back(3);
		this->testNetwork = CGraphicsNetwork(vec, &settings);
	}



double* modifyNetwork::getLayerOutput(){
		double* temp = new double(2);
		temp[0] = 1;
		temp[1] = 1;
		double* output1 = new double[3];
		//Modify the layer values
		output1 = this->testNetwork.getRootMeanSquareErrorForAllLayer(temp);
		return output1;
	}

	//Check if the subtract function works correctly
double modifyNetwork::getValueForLayer(int layer){
	thrust::host_vector<double> temp = this->testNetwork.v_layers[layer].getOutput();
		double k = 0;
		for (int i = 0; i < temp.size(); i++){
			for (int j = i; j < temp.size(); j++){
				k += (temp[i] - temp[j]) * (temp[i] - temp[j]);
			}
		}
		return sqrt(k);
	}


