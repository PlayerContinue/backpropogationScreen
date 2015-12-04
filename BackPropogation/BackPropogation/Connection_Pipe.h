#pragma once
#include <string>
#include <Windows.h>
#include <tchar.h>
#include<queue>

using namespace std;

class Connection_Pipe
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

	//Close the pipe connection
	void Close();

	//Add a new message to the outgoing pipe
	bool add_to_write_queue(string insert);


	//Retrieve a message from the queue
	string read_from_queue();

private:
	//Create a new Pipe
	HANDLE create_pipe(wstring pipe_name);
	wstring convert_to_wstring(string convert);
	//Write to the pipe
	int write(wstring toWrite);
	string read();
};


