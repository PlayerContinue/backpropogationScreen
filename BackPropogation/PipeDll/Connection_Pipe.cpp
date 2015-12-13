#include "stdafx.h"
#include "Connection_Pipe.h"
#include <iostream>

Connection_Pipe::Connection_Pipe()
{

	this->hPipeIn = create_pipe(L"Hyper_Pipe_IN");
	this->hPipeOut = create_pipe(L"Hyper_Pipe_OUT");
}

Connection_Pipe::Connection_Pipe(string pipe_name)
{

	//Create the pipe
	this->hPipeIn = create_pipe(convert_to_wstring(pipe_name + "_IN"));
	this->hPipeOut = create_pipe(convert_to_wstring(pipe_name + "_OUT"));
}

void Connection_Pipe::create_read_thread(){
	while (false){
		string data;
		DWORD numRead;
		if (ReadFile(this->hPipeIn, &data, 1024, &numRead, NULL)){//If something was read
			if (numRead > 0){
				this->RecieveQueue.push(data);
			}
		}
	}
}

void Connection_Pipe::Open(){
	using namespace std::placeholders;
	ConnectNamedPipe(this->hPipeIn, NULL);
	ConnectNamedPipe(this->hPipeOut, NULL);
	auto bound_member_fn = std::bind(&Connection_Pipe::create_read_thread, this);
	this->read_thread = new thread(bound_member_fn);
}

void Connection_Pipe::Close(){
	DisconnectNamedPipe(this->hPipeIn);
	DisconnectNamedPipe(this->hPipeOut);
	CloseHandle(this->hPipeIn);
	CloseHandle(this->hPipeOut);
}

bool Connection_Pipe::add_to_write_queue(string insert){
	this->SendQueue.push(this->convert_to_wstring(insert).c_str());
	return true;
}

string Connection_Pipe::read_from_queue(){
	string front = this->RecieveQueue.front();
	this->RecieveQueue.pop();
	return front;
}

int Connection_Pipe::write(string toWrite){
	return this->write(this->convert_to_wstring(toWrite).c_str());
}

int Connection_Pipe::write(wstring toWrite){
	DWORD bytesWritten = 0;
	const wchar_t* data = toWrite.c_str();
	size_t temp = wcslen(data)* sizeof(wchar_t);
	bool fail = WriteFile(this->hPipeOut, data, wcslen(data) * sizeof(wchar_t), &bytesWritten, NULL);

	FlushFileBuffers(this->hPipeOut);
	return (int)bytesWritten;
}





HANDLE Connection_Pipe::create_pipe(wstring pipe_name){
	HANDLE hPipe;
	pipe_name = TEXT("\\\\.\\pipe\\") + pipe_name;//Combine the name with the general name of a pipe

	hPipe = CreateFile(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
		NULL, OPEN_EXISTING, 0, NULL);

	if (hPipe == INVALID_HANDLE_VALUE){
		hPipe = ::CreateNamedPipe(pipe_name.c_str(),
			PIPE_ACCESS_DUPLEX,
			PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE,
			PIPE_UNLIMITED_INSTANCES,
			4096,
			4096,
			0,
			NULL);
	}

	if (hPipe == INVALID_HANDLE_VALUE){
		cout << pipe_name.c_str();

	}

	return hPipe;

}

wstring Connection_Pipe::convert_to_wstring(string convert){
	//Convert the string to wstring 
	std::wstring ws;
	ws.assign(convert.begin(), convert.end());
	return ws;
}

string Connection_Pipe::read(){
	return this->read_from_queue();
}

bool Connection_Pipe::has_new_message(){
	return (this->RecieveQueue.size() > 0);
}