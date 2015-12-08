#pragma once
#ifdef PIPEDLL_EXPORTS
#define PIPEFUNCDLL_EXPORTS_API __declspec(dllexport) 
#else
#define PIPEFUNCDLL_EXPORTS_API __declspec(dllimport) 
#endif

#include <string>
#include <Windows.h>
#include <tchar.h>
#include <queue>

using namespace std;


class PIPEFUNCDLL_EXPORTS_API Connection_Pipe
{
private:
	HANDLE hPipeIn;
	HANDLE hPipeOut;
	std::queue<const wchar_t*> SendQueue;
	std::queue<string> RecieveQueue;
public:
	Connection_Pipe();
	Connection_Pipe(string pipe_name);
	
	//Open the pipe connection
	void Open();

	//Checks if a new message is available to read
	bool has_new_message();

	//Close the pipe connection
	void Close();

	int write(string toWrite);
	int write(wstring toWrite);
	string read();
private:
	//Create a new Pipe
	HANDLE create_pipe(wstring pipe_name);
	wstring convert_to_wstring(string convert);
	//Add a new message to the outgoing pipe
	bool add_to_write_queue(string insert);

	//Retrieve a message from the queue
	string read_from_queue();
	
};


