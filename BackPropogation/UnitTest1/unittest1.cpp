#include "stdafx.h"
#include "CppUnitTest.h"
#include <vector>
#include "modifyNetwork.cuh"
#include <math.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;



namespace UnitTest1
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			modifyNetwork temp = modifyNetwork();

			double* output1 = temp.getLayerOutputUse();

			for (int i = 0; i < 3; i++){
				Assert::AreEqual(temp.getLayerForValue(i), output1[i]);
			}
		
		}

	};
}