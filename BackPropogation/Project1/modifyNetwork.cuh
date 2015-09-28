#include "../BackPropogation/CGraphicNetwork.cuh"
#include "../BackPropogation/CSettings.h"
#include "../BackPropogation/structures_cuda.cuh"
#include "../BackPropogation/main.cu"

class modifyNetwork{
public:


	modifyNetwork();

	double* getLayerOutputUse();

	double getLayerForValue(int i);

private:
	CGraphicsNetwork testNetwork;
	void buildNetwork();



	double* getLayerOutput();

	//Check if the subtract function works correctly
	double getValueForLayer(int layer);





};