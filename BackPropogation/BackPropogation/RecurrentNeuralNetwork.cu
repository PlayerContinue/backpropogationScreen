#include "RecurrentNeuralNetwork.cuh"

RecurrentNeuralNetwork::RecurrentNeuralNetwork(CSettings settings){
	this->settings = settings;
	this->initialize_network();
}